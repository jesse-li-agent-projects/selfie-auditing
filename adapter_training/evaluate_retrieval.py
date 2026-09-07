"""Generate a description per held-out vector and score it by GTE-large
embedding retrieval against a topic index -- the experiment's headline
number.

    python -m adapter_training.evaluate_retrieval \\
        --vectors vectors/bg_think_l19 --split val \\
        --checkpoint runs/bg_think/best.pt \\
        --dataset-file <jsonl> --center \\
        --positions all --report eval/bg_think_retrieval.json

`--checkpoint untrained` scores the floor comparator instead of a file.
"""

import argparse
import json
import random
from pathlib import Path

# Light import: config.py pulls in no heavy dependencies, so --help stays fast.
from config import BASE_MODEL_8B


def parse_vectors_k(values: list[str]) -> dict[int, Path]:
    """Parse repeated `--vectors-k K=DIR` values into `{k: outputs/DIR}`.

    Duplicated from `train_adapter.parse_vectors_k` rather than imported --
    importing `train_adapter` here would pull in its unconditional heavy
    imports (`torch`, `numpy`) and defeat this module's own `--help`-stays-fast
    discipline.

    :param values: raw `"K=DIR"` strings, one per `--vectors-k` occurrence
    :raises ValueError: on a malformed entry or a repeated k
    """
    directories: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--vectors-k expects K=DIR, got {value!r}")
        key, _, raw_dir = value.partition("=")
        k = int(key)
        if k in directories:
            raise ValueError(f"--vectors-k given twice for k={k}")
        directories[k] = Path("outputs") / raw_dir
    return directories


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a projection checkpoint by embedding retrieval "
        "over the generated descriptions of its held-out vectors."
    )
    parser.add_argument(
        "--vectors",
        type=lambda value: Path("outputs") / value,
        default=None,
        help="extraction output dir, written under outputs/ (implicitly prepended). "
        "Mutually exclusive with --vectors-k",
    )
    parser.add_argument(
        "--vectors-k",
        action="append",
        default=None,
        metavar="K=DIR",
        help="one extraction dir per k, e.g. '--vectors-k 1=bg_think_l19 "
        "--vectors-k 2=bg_think_many_l19_k2 --vectors-k 3=bg_think_many_l19_k3'. "
        "Queries every k given, each scored by set-level recall (score_sets), "
        "and reports per k (bg_think_many D12) in one invocation so the pooled "
        "centring reference and the retrieval index are each built once. "
        "Mutually exclusive with --vectors and --grouped",
    )
    parser.add_argument(
        "--pool-vectors-k",
        action="append",
        default=None,
        metavar="K=DIR",
        help="--vectors-k only: the directories the pooled centring reference "
        "(D13/D14) is computed from, if different from --vectors-k's own -- "
        "e.g. querying only the k=1 directory while pooling against all three, "
        "the condition the adapter actually trained under. Defaults to "
        "--vectors-k's own directories when omitted",
    )
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="'untrained', a local .pt/.safetensors path, or a 'repo_id:filename' Hub pair",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="local JSONL topic dataset (the index needs the full corpus, "
        "not just this run's own topics.json); omit to read the Hub",
    )
    parser.add_argument(
        "--center",
        dest="center",
        action="store_true",
        help="subtract per-position means -- the training-time condition, and "
        "the paper-comparable one (default here)",
    )
    parser.add_argument(
        "--no-center",
        dest="center",
        action="store_false",
        help="raw vectors -- the interpretation-time condition",
    )
    parser.set_defaults(center=True)
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="score a grouped (groups.json) extraction directory: the true "
        "topic set per query is a GroupRecord's titles, scored by set-level "
        "recall (score_sets, bg_think_many D11/step5) instead of the "
        "single-topic recall a topics.json directory gets",
    )
    parser.add_argument(
        "--positions",
        default="all",
        help="'all' (mean over every position, the trained adapter's primary number), "
        "'last' (each topic's own last vector), or a comma-separated list "
        "of offsets; ignored for a one-vector-per-topic directory",
    )
    parser.add_argument(
        "--restrict-topics-to",
        type=lambda value: Path("outputs") / value,
        default=None,
        help="intersect --vectors' topics with this directory's own topic set "
        "before querying, so a recall difference cannot be a topic-population "
        "difference (e.g. restrict the baseline adapter's topics to the pangram "
        "style's compliant ones)",
    )
    parser.add_argument(
        "--limit-topics",
        type=int,
        default=None,
        help="score a fixed random subsample of the query topics instead of "
        "the whole split -- a cheaper first pass",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed for --limit-topics")
    parser.add_argument("--k-values", default="1,5,10")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gen-seed", type=int, default=42)
    parser.add_argument("--embedding-model", default="thenlper/gte-large")
    parser.add_argument(
        "--index-cache",
        type=lambda value: Path("outputs") / value,
        default=None,
        help="build the index once and reuse it here, and across other arms' "
        "runs (same corpus, same strategy, same embedding model)",
    )
    parser.add_argument("--model", default=BASE_MODEL_8B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--report", type=Path, default=None, help="write the JSON report here"
    )
    parsed = parser.parse_args()
    if (parsed.vectors is None) == (parsed.vectors_k is None):
        parser.error("exactly one of --vectors or --vectors-k is required")
    if parsed.vectors_k is not None:
        if parsed.grouped or parsed.restrict_topics_to is not None:
            parser.error(
                "--grouped and --restrict-topics-to are not supported with "
                "--vectors-k -- --vectors-k always scores by set-level recall, "
                "per k, and --restrict-topics-to's title-intersection has no "
                "meaning for a group's plural topic set"
            )
        try:
            parsed.vectors_k = parse_vectors_k(parsed.vectors_k)
            parsed.pool_vectors_k = (
                parse_vectors_k(parsed.pool_vectors_k)
                if parsed.pool_vectors_k is not None
                else parsed.vectors_k
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif parsed.pool_vectors_k is not None:
        parser.error("--pool-vectors-k requires --vectors-k")
    elif parsed.grouped and parsed.restrict_topics_to is not None:
        parser.error("--restrict-topics-to is not supported with --grouped")
    return parsed


# Parsed before the heavy imports below, so `--help` costs no torch import.
args = parse_args() if __name__ == "__main__" else None

from adapter_training.checkpoints import (
    load_projection,
    untrained_projection,
)  # noqa: E402
from adapter_training.dataset import (  # noqa: E402
    DEFAULT_DATASET,
    load_group_records,
    load_records,
    load_topic_records,
    load_topics,
    load_vector_store,
    pooled_position_means,
)
from adapter_training.extract_grouped_vectors import drop_semicolon_topics  # noqa: E402
from adapter_training.grouped_examples import group_record_from_topic  # noqa: E402
from adapter_training.retrieval_eval import (  # noqa: E402
    DEFAULT_INDEX_STRATEGY,
    GenerationConfig,
    _ProjectionAdapter,
    build_index,
    check_sentence_transformers_available,
    evaluate_grouped_positions,
    evaluate_positions,
)


def load_query_records(
    vectors_dir: Path,
    *,
    split: str,
    restrict_to: Path | None,
    limit_topics: int | None,
    seed: int,
):
    """The topics to query: `--vectors`' own `split`, optionally intersected
    with another directory's topic set and/or subsampled.
    """
    records = load_records(vectors_dir, restrict_to)
    records = [record for record in records if record.split == split]
    if limit_topics is not None and limit_topics < len(records):
        records = random.Random(seed).sample(records, limit_topics)
    return records


def load_grouped_query_records(
    vectors_dir: Path,
    *,
    split: str,
    limit_topics: int | None,
    seed: int,
):
    """The `--grouped` counterpart to `load_query_records`: groups instead of
    topics, no `restrict_to` (a group's true topic set is plural, so
    intersecting by title does not carry the same meaning it does for a
    single-topic directory).
    """
    records = load_group_records(vectors_dir)
    records = [record for record in records if record.split == split]
    if limit_topics is not None and limit_topics < len(records):
        records = random.Random(seed).sample(records, limit_topics)
    return records


def load_bridged_records(directory: Path):
    """A directory's records as `GroupRecord`s, bridging a `topics.json`
    directory (k=1) exactly as the training mixture does
    (`grouped_examples.group_record_from_topic`,
    `extract_grouped_vectors.drop_semicolon_topics`) so a `--vectors-k` run's
    k=1 slice is scored over the same population and shape as its k=2/k=3
    slices -- both splits, not filtered to the query split.
    """
    if (directory / "groups.json").exists():
        return load_group_records(directory)
    topic_records, _dropped = drop_semicolon_topics(load_topic_records(directory))
    return [group_record_from_topic(record) for record in topic_records]


def run_multi_k(
    args, *, index, model, tokenizer, device, generation_config, k_values
) -> dict:
    """The `--vectors-k` path: query every given k, all centred against one
    pooled reference (D13/D14) built from `--pool-vectors-k` (defaulting to
    `--vectors-k`'s own directories), reported per k (bg_think_many D12) in
    one invocation so the index and the pooled reference are each built once.
    """
    records_cache: dict[Path, list] = {}

    def bridged(directory: Path):
        if directory not in records_cache:
            records_cache[directory] = load_bridged_records(directory)
        return records_cache[directory]

    pool_sources = [
        (directory, bridged(directory))
        for _k, directory in sorted(args.pool_vectors_k.items())
    ]
    pooled = pooled_position_means(pool_sources) if args.center else None

    store_and_records: dict[int, tuple] = {}
    for k, directory in sorted(args.vectors_k.items()):
        all_records = bridged(directory)
        query_records = [r for r in all_records if r.split == args.split]
        if args.limit_topics is not None and args.limit_topics < len(query_records):
            query_records = random.Random(args.seed).sample(
                query_records, args.limit_topics
            )
        store = load_vector_store(
            directory, center=args.center, records=all_records, means=pooled
        )
        store_and_records[k] = (store, query_records)
        print(f"k={k}: querying {len(query_records)} groups from {directory}")

    hidden_size = next(iter(store_and_records.values()))[0].hidden_size
    if args.checkpoint == "untrained":
        projection = untrained_projection(hidden_size, device=device)
        checkpoint_metadata = {"checkpoint": "untrained"}
    else:
        projection, checkpoint_metadata = load_projection(
            args.checkpoint, device=device, dim=hidden_size
        )
    adapter = _ProjectionAdapter(projection)

    per_k = {}
    for k, (store, query_records) in sorted(store_and_records.items()):
        result = evaluate_grouped_positions(
            index,
            model,
            tokenizer,
            adapter,
            store.vectors,
            query_records,
            args.positions,
            generation_config,
            k_values,
            device,
        )
        directory = args.vectors_k[k]
        with open(directory / "positions.json") as handle:
            run_positions = json.load(handle)
        per_k[str(k)] = {
            **result,
            "vectors_dir": str(directory),
            "layer": run_positions.get("layer"),
            "prompt_style": run_positions.get("prompt_style"),
        }

    return {
        "mode": "multi_k",
        "checkpoint": args.checkpoint,
        "checkpoint_metadata": checkpoint_metadata,
        "center": args.center,
        "vectors_k": {str(k): str(d) for k, d in sorted(args.vectors_k.items())},
        "pool_vectors_k": {
            str(k): str(d) for k, d in sorted(args.pool_vectors_k.items())
        },
        "positions_spec": args.positions,
        "split": args.split,
        "model": args.model,
        "index_size": len(index.titles),
        "index_strategy": DEFAULT_INDEX_STRATEGY.value,
        "embedding_model": args.embedding_model,
        "generation_config": vars(generation_config),
        "per_k": per_k,
    }


def main(args) -> dict:
    from model_loading import load_base_model, load_tokenizer, resolve_device

    check_sentence_transformers_available()

    print(f"Centring mode: {'centred' if args.center else 'raw'}")

    if args.vectors_k is not None:
        topics = load_topics(DEFAULT_DATASET, args.dataset_file)
        print(f"Index corpus: {len(topics)} topics")
        tokenizer = load_tokenizer(args.model)
        model = load_base_model(args.model, device=args.device, dtype=args.dtype)
        device = resolve_device(model)
        index = build_index(
            topics,
            strategy=DEFAULT_INDEX_STRATEGY,
            embedding_model=args.embedding_model,
            device=args.device,
        )
        if args.index_cache is not None:
            index.build_or_load_index(cache_path=args.index_cache)
        k_values = [int(k) for k in args.k_values.split(",")]
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            n_samples=1,
            seed=args.gen_seed,
        )
        report = run_multi_k(
            args,
            index=index,
            model=model,
            tokenizer=tokenizer,
            device=device,
            generation_config=generation_config,
            k_values=k_values,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "per_k"}, indent=2))
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with open(args.report, "w") as handle:
                json.dump(report, handle, indent=2)
        return report

    if args.grouped:
        records = load_grouped_query_records(
            args.vectors,
            split=args.split,
            limit_topics=args.limit_topics,
            seed=args.seed,
        )
    else:
        records = load_query_records(
            args.vectors,
            split=args.split,
            restrict_to=args.restrict_topics_to,
            limit_topics=args.limit_topics,
            seed=args.seed,
        )
    print(
        f"Querying {len(records)} {'groups' if args.grouped else 'topics'} "
        f"from {args.vectors}"
    )

    store = (
        load_vector_store(
            args.vectors,
            center=args.center,
            records=load_group_records(args.vectors),
        )
        if args.grouped
        else load_vector_store(args.vectors, center=args.center)
    )

    topics = load_topics(DEFAULT_DATASET, args.dataset_file)
    print(f"Index corpus: {len(topics)} topics")

    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, device=args.device, dtype=args.dtype)
    device = resolve_device(model)

    index = build_index(
        topics,
        strategy=DEFAULT_INDEX_STRATEGY,
        embedding_model=args.embedding_model,
        # GTE-large is a separate, much smaller model -- args.device (a
        # single literal device) is fine for it even when the base model
        # above needed sharding across several.
        device=args.device,
    )
    if args.index_cache is not None:
        index.build_or_load_index(cache_path=args.index_cache)

    if args.checkpoint == "untrained":
        projection = untrained_projection(store.hidden_size, device=device)
        checkpoint_metadata = {"checkpoint": "untrained"}
    else:
        projection, checkpoint_metadata = load_projection(
            args.checkpoint, device=device, dim=store.hidden_size
        )
    adapter = _ProjectionAdapter(projection)

    k_values = [int(k) for k in args.k_values.split(",")]
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        n_samples=1,
        seed=args.gen_seed,
    )

    evaluate_fn = evaluate_grouped_positions if args.grouped else evaluate_positions
    result = evaluate_fn(
        index,
        model,
        tokenizer,
        adapter,
        store.vectors,
        records,
        args.positions,
        generation_config,
        k_values,
        device,
    )

    with open(args.vectors / "positions.json") as handle:
        run_positions = json.load(handle)

    report = {
        **result,
        "checkpoint": args.checkpoint,
        "checkpoint_metadata": checkpoint_metadata,
        "vectors_dir": str(args.vectors),
        "center": args.center,
        "grouped": args.grouped,
        "positions_spec": args.positions,
        "restrict_topics_to": (
            str(args.restrict_topics_to) if args.restrict_topics_to else None
        ),
        "split": args.split,
        "model": args.model,
        "layer": run_positions.get("layer"),
        "prompt_style": run_positions.get("prompt_style"),
        "index_size": len(index.titles),
        "index_strategy": DEFAULT_INDEX_STRATEGY.value,
        "embedding_model": args.embedding_model,
        "generation_config": vars(generation_config),
    }
    print(
        json.dumps({k: v for k, v in report.items() if k != "per_position"}, indent=2)
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    main(args)
