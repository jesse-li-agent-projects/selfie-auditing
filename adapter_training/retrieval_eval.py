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

from adapter_training.dataset import Topic, TopicRecord
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
