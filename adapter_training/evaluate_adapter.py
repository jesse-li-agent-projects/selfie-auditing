"""Scores any projection checkpoint on a val (or train) split. This is the
cheapest safety check for a newly-trained adapter: score the *published*
upstream adapter through this loss path and compare the measured loss to its
recorded `best_val_loss` of 1.3662.

    python -m adapter_training.evaluate_adapter \\
        --vectors vectors/baseline_l19 \\
        --checkpoint keenanpepper/selfie-adapters-llama-3.1-8b-instruct:wikipedia-scalar-affine.safetensors \\
        --split val --batch-size 256 --center --report eval/upstream_baseline.json

`--checkpoint untrained` scores the untrained floor instead of a file.
"""

import argparse
import json
from pathlib import Path

# Light import: config.py pulls in no heavy dependencies, so --help stays fast.
from config import BASE_MODEL_8B


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a projection checkpoint's loss on an extraction run's examples."
    )
    parser.add_argument(
        "--vectors",
        type=lambda value: Path("outputs") / value,
        required=True,
        help="extraction output dir, written under outputs/ (implicitly prepended)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="'untrained', a local .pt/.safetensors path, or a 'repo_id:filename' Hub pair",
    )
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--limit-examples",
        type=int,
        default=None,
        help="score a fixed random subsample of this size instead of the whole split",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for --limit-examples"
    )
    parser.add_argument(
        "--center",
        dest="center",
        action="store_true",
        help="subtract per-position means -- required to reproduce upstream's 1.3662",
    )
    parser.add_argument(
        "--no-center",
        dest="center",
        action="store_false",
        help="raw vectors, matching downstream interpretation-time use (default)",
    )
    parser.set_defaults(center=False)
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="arm C: pool each topic's vectors into one before scoring",
    )
    parser.add_argument(
        "--restrict-topics-to",
        type=lambda value: Path("outputs") / value,
        default=None,
        help="intersect --vectors' topics with this directory's own topic set "
        "(e.g. the baseline style's, which filters nothing) before scoring",
    )
    parser.add_argument("--model", default=BASE_MODEL_8B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-loss", type=float, default=100.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--no-strip-labels", dest="strip_labels", action="store_false", default=True
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="write the JSON report here"
    )
    return parser.parse_args()


# Parsed before the heavy imports below, so `--help` costs no torch import.
args = parse_args() if __name__ == "__main__" else None

from adapter_training.checkpoints import load_projection, untrained_projection
from adapter_training.dataset import (
    examples_from_records,
    load_records,
    load_vector_store,
    pooled_vector_store,
)
from adapter_training.loss import LossConfig, SoftPromptLoss, evaluate, subsample


def load_eval_set(
    vectors_dir: Path,
    *,
    split: str,
    center: bool,
    pooled: bool,
    restrict_to: Path | None,
):
    """Build the `(VectorStore, examples)` pair `evaluate` scores.

    :param vectors_dir: an extraction output directory
    :param split: which split's topics to keep
    :param center: passed to `load_vector_store` (ignored for `pooled`, which
        always centres -- pooling raw vectors would average across positions
        before the per-position mean is subtracted, a different operation)
    :param pooled: arm C -- pool each topic's vectors into one
    :param restrict_to: intersect topics with this directory's own topic set
    """
    records = load_records(vectors_dir, restrict_to)
    if pooled:
        return pooled_vector_store(vectors_dir, records=records, split=split)
    store = load_vector_store(vectors_dir, center=center)
    return store, examples_from_records(records, split)


def main(args) -> dict:
    from model_loading import load_base_model, load_tokenizer, resolve_device

    print(
        f"Centring mode: {'centred' if args.center else 'raw'} "
        f"({'matches upstream validate() -- needed to reproduce 1.3662' if args.center else 'matches downstream interpretation-time use (interpret.py)'})"
    )

    store, examples = load_eval_set(
        args.vectors,
        split=args.split,
        center=args.center,
        pooled=args.pooled,
        restrict_to=args.restrict_topics_to,
    )
    if args.limit_examples is not None:
        examples = subsample(examples, args.limit_examples, args.seed)
    print(f"Scoring {len(examples)} examples from {args.vectors}")

    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, device=args.device, dtype=args.dtype)
    device = resolve_device(model)

    if args.checkpoint == "untrained":
        projection = untrained_projection(store.hidden_size, device=device)
        checkpoint_metadata = {"checkpoint": "untrained"}
    else:
        projection, checkpoint_metadata = load_projection(
            args.checkpoint, device=device, dim=store.hidden_size
        )

    loss_config = LossConfig(
        max_loss=args.max_loss,
        label_smoothing=args.label_smoothing,
        strip_labels=args.strip_labels,
    )
    scorer = SoftPromptLoss(model, tokenizer, projection, loss_config)

    result = evaluate(store, examples, scorer, args.batch_size)

    with open(args.vectors / "positions.json") as handle:
        positions = json.load(handle)

    report = {
        **result,
        "checkpoint": args.checkpoint,
        "checkpoint_metadata": checkpoint_metadata,
        "vectors_dir": str(args.vectors),
        "center": args.center,
        "pooled": args.pooled,
        "restrict_topics_to": (
            str(args.restrict_topics_to) if args.restrict_topics_to else None
        ),
        "split": args.split,
        "model": args.model,
        "layer": positions.get("layer"),
        "prompt_style": positions.get("prompt_style"),
    }
    print(json.dumps(report, indent=2))
    if "best_val_loss" in checkpoint_metadata:
        print(
            f"Recorded best_val_loss: {checkpoint_metadata['best_val_loss']} "
            f"vs measured: {result['measured_loss']}"
        )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    main(args)
