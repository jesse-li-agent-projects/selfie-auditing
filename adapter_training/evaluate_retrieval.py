"""Generate a description per held-out vector and score it by GTE-large
embedding retrieval against a topic index -- the experiment's headline
number.

    python -m adapter_training.evaluate_retrieval \\
        --vectors vectors/pangram_l19 --split val \\
        --checkpoint runs/phase0_armB/best.pt \\
        --dataset-file <jsonl> --center \\
        --positions all --report eval/armB_retrieval.json

`--checkpoint untrained` scores the floor comparator instead of a file.
"""

import argparse
import json
import random
from pathlib import Path

# Light import: config.py pulls in no heavy dependencies, so --help stays fast.
from config import BASE_MODEL_8B


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a projection checkpoint by embedding retrieval "
        "over the generated descriptions of its held-out vectors."
    )
    parser.add_argument(
        "--vectors",
        type=lambda value: Path("outputs") / value,
        required=True,
        help="extraction output dir, written under outputs/ (implicitly prepended)",
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
        help="subtract per-position means -- the paper-comparable condition (default)",
    )
    parser.add_argument(
        "--no-center",
        dest="center",
        action="store_false",
        help="raw vectors -- the downstream deployment condition",
    )
    parser.set_defaults(center=True)
    parser.add_argument(
        "--positions",
        default="all",
        help="'all' (mean over every position, arm B's primary number), "
        "'last' (each topic's own last vector), or a comma-separated list "
        "of offsets; ignored for a one-vector-per-topic directory",
    )
    parser.add_argument(
        "--restrict-topics-to",
        type=lambda value: Path("outputs") / value,
        default=None,
        help="intersect --vectors' topics with this directory's own topic set "
        "before querying, so a recall difference cannot be a topic-population "
        "difference (e.g. restrict arm A's baseline topics to the pangram "
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
    return parser.parse_args()


# Parsed before the heavy imports below, so `--help` costs no torch import.
args = parse_args() if __name__ == "__main__" else None

from adapter_training.checkpoints import (
    load_projection,
    untrained_projection,
)  # noqa: E402
from adapter_training.dataset import (  # noqa: E402
    DEFAULT_DATASET,
    load_records,
    load_topics,
    load_vector_store,
)
from adapter_training.retrieval_eval import (  # noqa: E402
    DEFAULT_INDEX_STRATEGY,
    GenerationConfig,
    _ProjectionAdapter,
    build_index,
    check_sentence_transformers_available,
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


def main(args) -> dict:
    from model_loading import load_base_model, load_tokenizer, resolve_device

    check_sentence_transformers_available()

    print(
        f"Centring mode: {'centred' if args.center else 'raw'} "
        f"({'paper-comparable' if args.center else 'downstream deployment condition'})"
    )

    records = load_query_records(
        args.vectors,
        split=args.split,
        restrict_to=args.restrict_topics_to,
        limit_topics=args.limit_topics,
        seed=args.seed,
    )
    print(f"Querying {len(records)} topics from {args.vectors}")

    store = load_vector_store(args.vectors, center=args.center)

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

    result = evaluate_positions(
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
