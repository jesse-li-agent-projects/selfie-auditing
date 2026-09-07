"""Generation and GTE-large embedding-retrieval scoring -- the headline
metric: can a description decoded from one activation be retrieved back to
its topic, against an index of the whole topic corpus?

Neither the injection path nor the ranking maths is reimplemented here:
generation goes through `interpret.generate_interpretations_batch` (the same
path interpretation-time use takes), and the index and recall@k come from
`adapter_training.topic_retrieval_eval` (vendored from upstream's
`evals/embedding_retrieval/topic_retrieval_eval.py`). What this module adds
is the two things that do not fit as-is: an index built without the
reference's Hub-only `load_dataset`, and the query-vector selection for the
pangram style's `count`-many positions per topic.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

from adapter_training.dataset import GroupRecord, Topic, TopicRecord
from interpret import Adapter, generate_interpretations_batch

try:
    import datasets as _datasets  # noqa: F401
except ImportError:
    # The reference module imports `datasets` at top level only for its own
    # Hub-based TopicRetrievalIndex.load_dataset(), which build_index()
    # below deliberately never calls (we build the index from a local
    # topics list instead). Stub it rather than require a heavy, unused,
    # possibly network-unreachable dependency; a real call surfaces loudly.
    import importlib.machinery
    import types

    _stub = types.ModuleType("datasets")
    # A real ModuleSpec, not just a sys.modules entry: transformers probes
    # datasets' availability via importlib.util.find_spec("datasets"), which
    # raises ValueError (not just "not found") on a module with __spec__=None.
    _stub.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)

    def _unexpected_load_dataset(*_args, **_kwargs):
        raise RuntimeError(
            "datasets.load_dataset() was actually called, but only a stub "
            "is installed (real `datasets` package is not available here). "
            "This means something now uses TopicRetrievalIndex.load_dataset() "
            "instead of this module's own build_index()."
        )

    _stub.load_dataset = _unexpected_load_dataset  # type: ignore[attr-defined]
    sys.modules["datasets"] = _stub

from adapter_training.topic_retrieval_eval import (  # noqa: E402
    IndexStrategy,
    TopicRetrievalConfig,
    TopicRetrievalIndex,
    evaluate_labels,
)

# The paper's own choice for contrastive-vector retrieval; the reference's
# own default (TITLE_ONLY) is a different index and would make our numbers
# incomparable to the published 94%/1%.
DEFAULT_INDEX_STRATEGY = IndexStrategy.TITLE_PLUS_ALL_LABELS
DEFAULT_EMBEDDING_MODEL = "thenlper/gte-large"


def check_sentence_transformers_available() -> None:
    """Fail loudly, by package name, before any generation or index build --
    `sentence_transformers` cannot be installed on the egress-free remote, so
    a missing import must be caught before GPU time is spent, not after.
    """
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "sentence_transformers is required for the retrieval eval "
            "(pip install sentence-transformers) -- checked before any "
            "generation or index build so a missing dependency never costs "
            "GPU time"
        ) from exc


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding settings; identical across arms or the comparison is void."""

    max_new_tokens: int = 30
    temperature: float = 0.7
    n_samples: int = 1
    seed: int = 42


def build_index(
    topics: list[Topic],
    *,
    strategy: IndexStrategy = DEFAULT_INDEX_STRATEGY,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str,
) -> TopicRetrievalIndex:
    """Build a `TopicRetrievalIndex` over `topics`, without the reference's
    own `load_dataset` (which calls the Hub).

    Pre-populating `titles`/`labels` before `build_index()` is what makes it
    skip `load_dataset` -- `TopicRetrievalIndex.build_index` only calls it
    `if not self.titles`.

    :param topics: every topic the index should cover (the full corpus, not
        just the query population -- a bigger index is a harder, fixed task,
        the same for every arm)
    :param strategy: what to embed per topic; record it beside any score
    :param embedding_model: passed straight to `SentenceTransformer`
    :param device: embedding device
    :return: the built index, ready for `evaluate_labels` / `score`
    """
    config = TopicRetrievalConfig(
        embedding_model=embedding_model, device=device, index_strategy=strategy
    )
    index = TopicRetrievalIndex(config)
    index.titles = [topic.title for topic in topics]
    index.labels = [list(topic.labels) for topic in topics]
    index.build_index()
    return index


class _ProjectionAdapter(Adapter):
    """Adapts a bare projection module (`checkpoints.load_projection` /
    `checkpoints.untrained_projection`'s return) to the `.transform`
    interface `generate_interpretations_batch` expects -- the same operation
    `adapter_training.inference.SelfIEAdapter.transform` does, done here because
    `load_projection` intentionally returns the bare module (loss.py calls
    it directly too).
    """

    def __init__(self, projection: torch.nn.Module):
        self.projection = projection

    def transform(self, vector: torch.Tensor) -> torch.Tensor:
        return self.projection(vector.float()).to(vector.dtype)


def generate_descriptions(
    model,
    tokenizer,
    adapter: Adapter,
    vectors: torch.Tensor,
    config: GenerationConfig,
    device: str,
) -> list[str]:
    """One sampled description per row of `vectors`, via
    `generate_interpretations_batch`.

    Sampling draws from the process-global `torch` RNG stream, so a row's
    draws depend on how many other rows were generated before it: results are
    reproducible only for a fixed row ordering and a fixed generation batch
    size. Never vary either between two calls meant to be compared.

    :param vectors: `[n, hidden]`, already centred or not as the caller chose
    :param config: decoding settings; `n_samples` beyond 1 all sample the
        same row, and only the first is kept -- fix `n_samples=1` unless a
        caller has a specific reason to draw more
    :return: one description per row, in row order
    """
    torch.manual_seed(config.seed)
    hidden_vectors = {i: vectors[i] for i in range(vectors.shape[0])}
    results = dict(
        generate_interpretations_batch(
            model,
            tokenizer,
            adapter,
            hidden_vectors,
            n_samples=config.n_samples,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            device=device,
        )
    )
    return [results[i][0] for i in range(vectors.shape[0])]


def score(
    index: TopicRetrievalIndex,
    descriptions: list[str],
    ground_truth_titles: list[str],
    k_values: list[int],
) -> dict:
    """Recall@k etc. for `descriptions` against `index`, via the reference's
    own `evaluate_labels` -- do not reimplement the ranking math here.

    :param ground_truth_titles: `descriptions[i]`'s correct topic title; must
        appear in `index.titles`
    :return: `evaluate_labels`'s result dict (`recalls`, `mrr`, ...)
    """
    title_to_index = {title: i for i, title in enumerate(index.titles)}
    ground_truth_indices = [title_to_index[title] for title in ground_truth_titles]
    return evaluate_labels(index, descriptions, ground_truth_indices, k_values=k_values)


def split_segments(description: str) -> list[str]:
    """Split a generation on `;`, stripping whitespace and dropping empty
    segments -- the segment count is never forced to k (bg_think_many D11).
    """
    return [segment.strip() for segment in description.split(";") if segment.strip()]


def score_sets(
    index: TopicRetrievalIndex,
    descriptions: list[str],
    true_topic_titles: list[tuple[str, ...]],
    k_values: list[int],
    batch_size: int = 512,
) -> dict:
    """Set-level recall (bg_think_many D11, `step5_set_retrieval_eval.md`):
    for each query's true topic set, the best rank any of its generated
    segments gives that topic, then recall@N over every (query, true topic)
    pair.

    With k=1 and a generation with no `;`, this reduces *exactly* to `score`
    / `evaluate_labels` on the same inputs: one segment (the whole
    description) scored against the sole true topic, ranked by the same
    "count of strictly-higher-similarity others, plus one" convention --
    tested in `tests/test_set_retrieval.py` so the k=1 slice of this
    experiment stays comparable to `bg_think`'s existing numbers.

    Extra segments are not penalised (a query's segment count need not equal
    its k) -- the known precision hole D11 accepts, made visible by the
    `segments` block rather than hidden in the recall number.

    :param descriptions: one generation per query, in query order
    :param true_topic_titles: query's true topic titles, in prompt order;
        every query must carry the same number of titles (D12 -- never pool
        across k)
    :param k_values: recall@N cutoffs to report
    :param batch_size: segments per embedding/similarity batch (memory, not
        correctness -- mirrors `evaluate_labels`' own batching)
    :return: `{"k", "n_queries", "n_true_topics", "recalls", "mrr",
        "segments": {"mean", "histogram", "fraction_not_equal_k"},
        "per_query"}`
    """
    n_queries = len(descriptions)
    assert len(true_topic_titles) == n_queries
    ks = {len(titles) for titles in true_topic_titles}
    if len(ks) > 1:
        raise ValueError(f"score_sets: mixed k across queries is not supported: {ks}")
    k = ks.pop() if ks else 0

    title_to_index = {title: i for i, title in enumerate(index.titles)}
    true_topic_indices = [
        [title_to_index[title] for title in titles] for titles in true_topic_titles
    ]

    segments_per_query = [split_segments(description) for description in descriptions]
    flat_segments: list[str] = []
    segment_query: list[int] = []
    for query_index, segments in enumerate(segments_per_query):
        for segment in segments:
            flat_segments.append(segment)
            segment_query.append(query_index)

    best_ranks: list[list[int | None]] = [[None] * k for _ in range(n_queries)]

    for start in range(0, len(flat_segments), batch_size):
        end = min(start + batch_size, len(flat_segments))
        batch_embeddings = index._embed_texts(
            flat_segments[start:end], show_progress=False
        )
        batch_similarities = torch.mm(batch_embeddings, index.topic_embeddings.T)
        for row in range(end - start):
            query_index = segment_query[start + row]
            sim_row = batch_similarities[row]
            for position, topic_index in enumerate(true_topic_indices[query_index]):
                # "Other topics with strictly higher similarity, plus one" --
                # the same rank convention as evaluate_labels, and `topic_index`
                # excludes itself automatically since x > x is never true.
                rank = int((sim_row > sim_row[topic_index]).sum().item()) + 1
                current = best_ranks[query_index][position]
                if current is None or rank < current:
                    best_ranks[query_index][position] = rank
        del batch_similarities

    n_true_topics = n_queries * k
    recalls = {
        n: (
            sum(
                1
                for ranks in best_ranks
                for rank in ranks
                if rank is not None and rank <= n
            )
            / n_true_topics
            if n_true_topics
            else 0.0
        )
        for n in k_values
    }
    reciprocal_ranks = [
        (1.0 / rank) if rank is not None else 0.0
        for ranks in best_ranks
        for rank in ranks
    ]
    mrr = sum(reciprocal_ranks) / n_true_topics if n_true_topics else 0.0

    segment_counts = [len(segments) for segments in segments_per_query]
    histogram: dict[str, int] = {}
    for count in segment_counts:
        key = str(count) if count < 4 else "4+"
        histogram[key] = histogram.get(key, 0) + 1
    fraction_not_equal_k = (
        sum(1 for count in segment_counts if count != k) / n_queries
        if n_queries
        else 0.0
    )

    per_query = [
        {
            "description": descriptions[i],
            "true_topics": list(true_topic_titles[i]),
            "n_segments": segment_counts[i],
            "best_ranks": best_ranks[i],
        }
        for i in range(n_queries)
    ]

    return {
        "k": k,
        "n_queries": n_queries,
        "n_true_topics": n_true_topics,
        "recalls": recalls,
        "mrr": mrr,
        "segments": {
            "mean": sum(segment_counts) / n_queries if n_queries else 0.0,
            "histogram": histogram,
            "fraction_not_equal_k": fraction_not_equal_k,
        },
        "per_query": per_query,
    }


def resolve_position_offsets(
    records: list[TopicRecord], positions: str
) -> list[int] | None:
    """Parse `--positions` into the offsets to score and average over, or
    `None` for `"last"` (a per-topic offset, not a fixed one -- see
    `query_pairs_for_last`).

    :param positions: `"all"`, `"last"`, or a comma-separated list of ints
    :return: fixed offsets (`"all"` -> `range(max count)`), or `None`
    """
    if positions == "last":
        return None
    if positions == "all":
        max_count = max((record.count for record in records), default=0)
        return list(range(max_count))
    return [int(p) for p in positions.split(",")]


def query_pairs_for_offset(
    records: list[TopicRecord], offset: int
) -> list[tuple[int, str]]:
    """`(vector_index, title)` for every topic whose `count` covers `offset`.

    A one-vector-per-topic directory (baseline, or pooled) has
    `count == 1` for every record, so only `offset == 0` ever yields
    anything -- `--positions` is effectively ignored there, with no special
    case needed.
    """
    return [
        (record.start + offset, record.title)
        for record in records
        if record.count > offset
    ]


def query_pairs_for_last(records: list[TopicRecord]) -> list[tuple[int, str]]:
    """`(vector_index, title)` for each topic's own last vector --
    `start + count - 1`, not a fixed offset, since `count` varies per topic
    in the pangram style (10 with the trailing full stop, 9 without).
    """
    return [(record.start + record.count - 1, record.title) for record in records]


def evaluate_positions(
    index: TopicRetrievalIndex,
    model,
    tokenizer,
    adapter: Adapter,
    vectors: torch.Tensor,
    records: list[TopicRecord],
    positions: str,
    generation_config: GenerationConfig,
    k_values: list[int],
    device: str,
) -> dict:
    """The full generate-then-score pass for one `--positions` spec.

    `"last"` scores one query set (each topic's own last vector). `"all"` or
    an explicit list scores each offset separately and reports both the
    per-offset breakdown and the mean recall over offsets -- the mean is the
    number that corresponds to training on every position as an equal
    example, and the breakdown answers which position carries the topic.

    :param vectors: the full extraction directory's vectors (any centring
        the caller already applied)
    :return: `{"mode": "last", ...score...}` or `{"mode": "per_position",
        "per_position": {offset: score, ...}, "recalls": mean recall@k,
        "best_position": offset with the best recall@min(k_values)}`
    """
    offsets = resolve_position_offsets(records, positions)

    if offsets is None:
        pairs = query_pairs_for_last(records)
        vector_indices, titles = zip(*pairs) if pairs else ((), ())
        descriptions = generate_descriptions(
            model,
            tokenizer,
            adapter,
            vectors[list(vector_indices)],
            generation_config,
            device,
        )
        return {
            "mode": "last",
            "n_queries": len(pairs),
            **score(index, descriptions, list(titles), k_values),
        }

    primary_k = min(k_values)
    per_position: dict[int, dict] = {}
    for offset in offsets:
        pairs = query_pairs_for_offset(records, offset)
        if not pairs:
            continue
        vector_indices, titles = zip(*pairs)
        descriptions = generate_descriptions(
            model,
            tokenizer,
            adapter,
            vectors[list(vector_indices)],
            generation_config,
            device,
        )
        per_position[offset] = score(index, descriptions, list(titles), k_values)

    mean_recalls = {
        k: sum(result["recalls"][k] for result in per_position.values())
        / len(per_position)
        for k in k_values
    }
    mean_mrr = sum(result["mrr"] for result in per_position.values()) / len(
        per_position
    )
    best_position = max(
        per_position, key=lambda offset: per_position[offset]["recalls"][primary_k]
    )
    return {
        "mode": "per_position",
        "positions": list(per_position.keys()),
        "n_queries": sum(len(query_pairs_for_offset(records, o)) for o in per_position),
        "per_position": per_position,
        "recalls": mean_recalls,
        "mrr": mean_mrr,
        "best_position": best_position,
    }


def query_pairs_for_offset_grouped(
    records: list[GroupRecord], offset: int
) -> list[tuple[int, tuple[str, ...]]]:
    """`(vector_index, titles)` for every group whose `count` covers `offset`
    -- the `--grouped` counterpart to `query_pairs_for_offset`."""
    return [
        (record.start + offset, record.titles)
        for record in records
        if record.count > offset
    ]


def query_pairs_for_last_grouped(
    records: list[GroupRecord],
) -> list[tuple[int, tuple[str, ...]]]:
    """`(vector_index, titles)` for each group's own last vector -- the
    `--grouped` counterpart to `query_pairs_for_last`."""
    return [(record.start + record.count - 1, record.titles) for record in records]


def evaluate_grouped_positions(
    index: TopicRetrievalIndex,
    model,
    tokenizer,
    adapter: Adapter,
    vectors: torch.Tensor,
    records: list[GroupRecord],
    positions: str,
    generation_config: GenerationConfig,
    k_values: list[int],
    device: str,
) -> dict:
    """The `--grouped` counterpart to `evaluate_positions`: a group's true
    topic set is `GroupRecord.titles`, scored by `score_sets` (bg_think_many
    D11) instead of the single-topic `score`.

    Shape mirrors `evaluate_positions` exactly -- see its docstring.
    """
    offsets = resolve_position_offsets(records, positions)

    if offsets is None:
        pairs = query_pairs_for_last_grouped(records)
        vector_indices, titles = zip(*pairs) if pairs else ((), ())
        descriptions = generate_descriptions(
            model,
            tokenizer,
            adapter,
            vectors[list(vector_indices)],
            generation_config,
            device,
        )
        return {
            "mode": "last",
            **score_sets(index, descriptions, list(titles), k_values),
        }

    primary_k = min(k_values)
    per_position: dict[int, dict] = {}
    for offset in offsets:
        pairs = query_pairs_for_offset_grouped(records, offset)
        if not pairs:
            continue
        vector_indices, titles = zip(*pairs)
        descriptions = generate_descriptions(
            model,
            tokenizer,
            adapter,
            vectors[list(vector_indices)],
            generation_config,
            device,
        )
        per_position[offset] = score_sets(index, descriptions, list(titles), k_values)

    mean_recalls = {
        k: sum(result["recalls"][k] for result in per_position.values())
        / len(per_position)
        for k in k_values
    }
    mean_mrr = sum(result["mrr"] for result in per_position.values()) / len(
        per_position
    )
    best_position = max(
        per_position, key=lambda offset: per_position[offset]["recalls"][primary_k]
    )
    return {
        "mode": "per_position",
        "positions": list(per_position.keys()),
        "n_queries": sum(
            len(query_pairs_for_offset_grouped(records, o)) for o in per_position
        ),
        "k": next(iter(per_position.values()))["k"] if per_position else None,
        "per_position": per_position,
        "recalls": mean_recalls,
        "mrr": mean_mrr,
        "best_position": best_position,
    }
