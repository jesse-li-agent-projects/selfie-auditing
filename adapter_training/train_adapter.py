"""Trains one projection to a budget expressed in examples seen (plan step 2b),
validates on a fixed subsample, and writes checkpoints
`selfie_adapters.load_adapter` can read -- so `interpret.py` and every other
downstream consumer keep working unchanged regardless of who trained the
file.

    python -m adapter_training.train_adapter \\
        --vectors vectors/pangram_l19 \\
        --run-dir runs/armB_scalar_affine \\
        --budget-examples 755391 \\
        --batch-size 256 --micro-batch-size 64 \\
        --projection-type scalar_affine \\
        --lr 0.01 --init-scale 5.0 --warmup-steps 10 --grad-clip 0.5 --seed 42 \\
        --val-subsample 5000 --validate-every 100 \\
        --pool-positions            # arm C: mean the 10 positions before the adapter

**Budget is examples seen, never epochs** (parent plan S4.1, D4): the cosine
schedule is laid out over `ceil(budget_examples / batch_size)` steps, and
`--max-steps` only stops the loop early -- it never changes that horizon, so
a debug run exercises the same schedule code the real run does.

**This trainer always uses centred vectors** (parent plan S3.2, S5.3): that
is what upstream's own `validate()` scored, and what the 1.3662 reproduction
check (`evaluate_adapter.py --center`) needs to be comparable to. Raw
vectors are a downstream-interpretation-time concern (`interpret.py`),
never a training one.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Light imports: config.py pulls in no heavy dependencies, so --help stays fast.
from config import BASE_MODEL_8B


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a SelfIE adapter projection to an examples-seen budget."
    )
    parser.add_argument(
        "--vectors",
        type=lambda value: Path("outputs") / value,
        required=True,
        help="extraction output dir, written under outputs/ (implicitly prepended)",
    )
    parser.add_argument(
        "--run-dir",
        type=lambda value: Path("outputs") / value,
        required=True,
        help="checkpoints and metrics written under outputs/ (implicitly prepended)",
    )
    parser.add_argument(
        "--budget-examples",
        type=int,
        required=True,
        help="examples seen, not epochs (parent plan S4.1, D4)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=256,
        help="gradient-accumulation chunk; defaults to --batch-size (no accumulation)",
    )
    parser.add_argument("--projection-type", default="scalar_affine")
    parser.add_argument(
        "--projection-rank",
        type=int,
        default=None,
        help="rank for scalar_affine_plus_low_rank / low_rank_only -- a config "
        "field, never a literal in code (parent plan S5.5)",
    )
    parser.add_argument("--low-rank-init-factor", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--init-scale", type=float, default=5.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-subsample", type=int, default=5000)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop the loop early; does NOT change the cosine schedule's horizon",
    )
    parser.add_argument(
        "--pool-positions",
        action="store_true",
        help="arm C: mean the 10 positions before the adapter",
    )
    parser.add_argument(
        "--restrict-topics-to",
        type=lambda value: Path("outputs") / value,
        default=None,
        help="intersect --vectors' topics with this directory's own topic set",
    )
    parser.add_argument(
        "--normalize-input", dest="normalize_input", action="store_true", default=True
    )
    parser.add_argument(
        "--no-normalize-input", dest="normalize_input", action="store_false"
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-loss", type=float, default=100.0)
    parser.add_argument(
        "--no-strip-labels", dest="strip_labels", action="store_false", default=True
    )
    parser.add_argument(
        "--buffer-batches",
        type=int,
        default=50,
        help="length-bucketing shuffle buffer, in batches (parent plan S4.2.1)",
    )
    parser.add_argument("--model", default=BASE_MODEL_8B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="off by default (parent plan D8): a ~1.5x tax that also nulls "
        "past_key_values, which blocks the prefix cache of step 2c",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        default=False,
        help="not implemented -- the default multi-GPU path is one run per GPU "
        "(parent plan S4.3); use --device to pick which GPU this run uses",
    )
    return parser.parse_args()


# Parsed before the heavy imports below, so `--help` costs no torch import.
args = parse_args() if __name__ == "__main__" else None

import itertools
import json
import random
import subprocess
from dataclasses import dataclass

import numpy as np
import torch

from adapter_training.checkpoints import save_checkpoint
from adapter_training.dataset import (
    Example,
    examples_from_records,
    load_topic_records,
    load_vector_store,
    pooled_vector_store,
    restrict_to_titles,
)
from adapter_training.evaluate_adapter import evaluate, subsample
from adapter_training.loss import LossConfig, SoftPromptLoss, target_text
from selfie_adapters.projection import create_projection_module


@dataclass(frozen=True)
class TrainConfig:
    """Everything the training loop needs, decoupled from argparse so tests
    can construct one directly."""

    budget_examples: int
    batch_size: int = 256
    micro_batch_size: int = 256
    projection_type: str = "scalar_affine"
    projection_rank: int | None = None
    low_rank_init_factor: float = 0.01
    lr: float = 0.01
    init_scale: float = 5.0
    warmup_steps: int = 10
    grad_clip: float = 0.5
    weight_decay: float = 0.01
    seed: int = 42
    val_subsample: int = 5000
    validate_every: int = 100
    max_steps: int | None = None
    normalize_input: bool = True
    label_smoothing: float = 0.0
    max_loss: float = 100.0
    strip_labels: bool = True
    buffer_batches: int = 50

    @classmethod
    def from_args(cls, args) -> "TrainConfig":
        return cls(
            budget_examples=args.budget_examples,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            projection_type=args.projection_type,
            projection_rank=args.projection_rank,
            low_rank_init_factor=args.low_rank_init_factor,
            lr=args.lr,
            init_scale=args.init_scale,
            warmup_steps=args.warmup_steps,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay,
            seed=args.seed,
            val_subsample=args.val_subsample,
            validate_every=args.validate_every,
            max_steps=args.max_steps,
            normalize_input=args.normalize_input,
            label_smoothing=args.label_smoothing,
            max_loss=args.max_loss,
            strip_labels=args.strip_labels,
            buffer_batches=args.buffer_batches,
        )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (parent plan S6 step 2, "seed everything").

    :param seed: the seed to use everywhere
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_total_steps(budget_examples: int, batch_size: int) -> int:
    """`ceil(budget_examples / batch_size)` -- the cosine schedule's horizon.

    755,391 examples at batch 256 gives 2,951, the published checkpoint's
    recorded `global_step` (parent plan S3.1).
    """
    return math.ceil(budget_examples / batch_size)


def lr_at_step(
    step: int, *, base_lr: float, warmup_steps: int, total_steps: int
) -> float:
    """Closed-form reproduction of upstream's chained `LinearLR` warmup ->
    `CosineAnnealingLR` (`_setup_scheduler` / `_get_current_scheduler`,
    `resources/selfie-adapters/training/trainer.py`).

    Upstream steps the scheduler *after* each optimizer step and increments
    `global_step` after that, so the learning rate an optimizer step at index
    `step` (0-indexed) actually uses is the scheduler's value at
    `last_epoch == step`, computed here in closed form instead of by
    stepping two chained `torch` schedulers:

    - warmup (`LinearLR(start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)`):
      linear from ~0 to `base_lr` over `warmup_steps` steps.
    - cosine (`CosineAnnealingLR(T_max=total_steps - warmup_steps, eta_min=0)`),
      starting fresh (its own `last_epoch=0`, i.e. full `base_lr`) the moment
      warmup ends -- so step `warmup_steps` lands exactly on the peak.

    :param step: 0-indexed optimizer step
    :param base_lr: the configured peak learning rate
    :param warmup_steps: number of linear warmup steps
    :param total_steps: the schedule's total horizon (never `--max-steps`)
    :return: the learning rate to use for this step
    """
    if step < warmup_steps:
        warmup_factor = 1e-6 + (1 - 1e-6) * step / warmup_steps
        return base_lr * warmup_factor
    t_max = max(total_steps - warmup_steps, 1)
    c = step - warmup_steps
    return base_lr * (1 + math.cos(math.pi * c / t_max)) / 2


def compute_target_lengths(
    examples: list[Example], tokenizer, loss_config: LossConfig
) -> dict[str, int]:
    """Tokenize every distinct label's target text once and cache it (parent
    plan S4.2.1: "Label lengths are tokenized once at startup"), for the
    length-bucketed batcher to sort by.

    :param examples: the pool `bucketed_batches` will draw from
    :param tokenizer: the model's tokenizer
    :param loss_config: supplies `target_text`'s `strip_labels`/`eos_token`
    :return: target token length, keyed by label text
    """
    lengths: dict[str, int] = {}
    for example in examples:
        if example.label not in lengths:
            text = target_text(example.label, loss_config)
            lengths[example.label] = len(
                tokenizer(text, add_special_tokens=False).input_ids
            )
    return lengths


def example_stream(examples: list[Example], seed: int):
    """An infinite generator over `examples`: a seeded shuffle, reshuffled
    into a fresh permutation whenever exhausted (parent plan S4.4).

    One pass is one full, non-repeating draw of every example in a fixed
    random order -- never a resample. Arms A and C spend exactly one budget
    on one pass; arm B's larger pool means its budget only ever completes
    ~0.1 of one.

    :param examples: the train-only pool to draw from (val is never passed here)
    :param seed: seed for the per-epoch shuffle
    :return: a generator yielding `Example`s forever
    """
    if not examples:
        raise ValueError("example_stream: empty example pool")
    epoch = 0
    while True:
        order = list(examples)
        random.Random(f"{seed}-epoch-{epoch}").shuffle(order)
        yield from order
        epoch += 1


def bucketed_batches(
    stream, batch_size: int, buffer_batches: int, length_of: dict, seed: int
):
    """Length-bucketed batching (parent plan S4.2.1, D8): fill a shuffle
    buffer of `buffer_batches` batches from `stream`, sort the buffer by
    target length, cut it into batches, shuffle the batch order, and emit.

    This is exact: it only changes which examples share a batch, never which
    examples are drawn or how many times -- `stream`'s own multiset is
    untouched.

    :param stream: an infinite example generator, e.g. `example_stream`
    :param batch_size: examples per batch
    :param buffer_batches: shuffle-buffer size, in batches
    :param length_of: target token length per label, from `compute_target_lengths`
    :param seed: seed for the batch-order shuffle
    :return: a generator yielding `list[Example]` batches forever
    """
    buffer_size = buffer_batches * batch_size
    buffer_index = 0
    while True:
        buffer = list(itertools.islice(stream, buffer_size))
        if not buffer:
            return
        buffer.sort(key=lambda example: length_of[example.label])
        chunks = [buffer[i : i + batch_size] for i in range(0, len(buffer), batch_size)]
        order = list(range(len(chunks)))
        random.Random(f"{seed}-bucket-order-{buffer_index}").shuffle(order)
        buffer_index += 1
        for i in order:
            yield chunks[i]


def check_validation_compute_ratio(
    *,
    val_subsample_size: int,
    batch_size: int,
    micro_batch_size: int,
    validate_every: int,
    steps_to_check: int,
) -> float:
    """Port of upstream's `_check_validation_compute_ratio` (parent plan S4.5,
    `resources/selfie-adapters/training/trainer.py`): refuse to start if
    validation would cost more than half of training's forward passes.

    Both sides are counted in forward-pass units (micro-batches), since that
    is what a validation run and a gradient-accumulated training step
    actually spend compute on.

    :param val_subsample_size: size of the fixed validation subsample
    :param batch_size: training's global batch size
    :param micro_batch_size: training's gradient-accumulation chunk size,
        also used for validation batches
    :param validate_every: optimizer steps between validations
    :param steps_to_check: optimizer steps the run will actually take
        (`--max-steps`-limited if set, else the full schedule)
    :raises ValueError: if the ratio exceeds 50%, naming `--val-subsample`
    :return: the computed ratio, for logging
    """
    grad_accum_steps = math.ceil(batch_size / micro_batch_size)
    val_batches_per_run = math.ceil(val_subsample_size / micro_batch_size)
    val_runs = steps_to_check / validate_every
    total_val_batches = val_batches_per_run * val_runs
    train_batches = steps_to_check * grad_accum_steps
    ratio = total_val_batches / train_batches if train_batches else float("inf")
    threshold = 0.5
    if ratio > threshold:
        raise ValueError(
            f"Validation would cost {ratio:.1%} of training's forward passes "
            f"(threshold {threshold:.0%}): {val_batches_per_run} val micro-batches "
            f"x {val_runs:.1f} runs = {total_val_batches:.0f}, against "
            f"{train_batches} train micro-batches. Lower --val-subsample or raise "
            f"--validate-every."
        )
    return ratio


def build_optimizer(projection, *, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW over the projection's parameters, splitting the scalar scale
    parameter into its own weight-decay-free group -- matching upstream's
    `_setup_optimizer`, which never decays `log_scale`/`base_log_scale`.
    Every group shares one learning rate: this trainer exposes a single
    `--lr`, matching upstream's default (`scale_lr`/`bias_lr` fall back to
    the base rate unless overridden, which upstream's own published run never did).

    :param projection: the projection module to optimize
    :param lr: learning rate for every parameter group
    :param weight_decay: weight decay for every group except the scale
    """
    scale_params, other_params = [], []
    for name, param in projection.named_parameters():
        if "log_scale" in name:
            scale_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if scale_params:
        groups.append({"params": scale_params, "lr": lr, "weight_decay": 0.0})
    if other_params:
        groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})
    return torch.optim.AdamW(groups)


def optimizer_step(
    batch: list[Example],
    store,
    scorer: SoftPromptLoss,
    optimizer: torch.optim.Optimizer,
    micro_batch_size: int,
    grad_clip: float,
) -> tuple[float, float]:
    """One global-batch optimizer step via gradient accumulation.

    Each micro-batch's loss is scaled by `micro_len / batch_len` rather than
    averaged, so the accumulated gradient stays equivalent to a single batch
    of `len(batch)` regardless of how it was chunked (parent plan S4.2.1,
    item 6) -- micro-batches from a bucketed batch happen to have equal
    length here, but this does not rely on that.

    :param batch: one global batch, in the order to micro-batch it
    :param store: the `VectorStore` `batch`'s vector indices address
    :param scorer: bound to the model, tokenizer and trainable projection
    :param optimizer: stepped once, over the whole accumulated gradient
    :param micro_batch_size: examples per micro-batch
    :param grad_clip: max gradient norm, applied to the projection's own parameters
    :return: `(batch loss, gradient norm)`, both as floats
    """
    batch_len = len(batch)
    optimizer.zero_grad()
    total_loss = 0.0
    for start in range(0, batch_len, micro_batch_size):
        micro = batch[start : start + micro_batch_size]
        micro_len = len(micro)
        vectors = store.vectors[[example.vector_index for example in micro]]
        labels = [example.label for example in micro]
        loss, _ = scorer(vectors, labels)
        (loss * (micro_len / batch_len)).backward()
        total_loss += loss.item() * micro_len / batch_len
    grad_norm = torch.nn.utils.clip_grad_norm_(
        scorer.projection.parameters(), grad_clip
    )
    optimizer.step()
    return total_loss, float(grad_norm)


def checkpoint_config(config: TrainConfig, *, total_steps: int) -> dict:
    """The `config` dict `checkpoints.save_checkpoint` writes into the
    checkpoint -- everything `selfie_adapters.load_adapter` needs to
    reconstruct the projection, plus the run's own training settings.
    """
    return {
        "projection": {
            "type": config.projection_type,
            "normalize_input": config.normalize_input,
            "init_scale": config.init_scale,
            "low_rank_rank": config.projection_rank,
        },
        "training": {
            "optimizer_type": "adamw",
            "learning_rate": config.lr,
            "weight_decay": config.weight_decay,
            "scheduler_type": "cosine",
            "warmup_steps": config.warmup_steps,
            "gradient_clip_norm": config.grad_clip,
            "batch_size": config.batch_size,
            "micro_batch_size": config.micro_batch_size,
            "budget_examples": config.budget_examples,
            "total_steps": total_steps,
            "seed": config.seed,
        },
    }


def _metric(projection, name: str):
    getter = getattr(projection, name, None)
    return getter() if getter is not None else None


def train(
    *,
    model,
    tokenizer,
    train_store,
    train_examples: list[Example],
    val_store,
    val_examples: list[Example],
    config: TrainConfig,
    run_dir: Path,
    device,
) -> dict:
    """The training loop: seeding, schedule, sampling, micro-batching,
    validation and checkpointing (parent plan S6 step 2). Callable directly
    (as tests do, with a fake model and a hand-built store) or via `main`.

    :param model: a frozen HF-style CausalLM (`.model`, `.lm_head`,
        `.get_input_embeddings()`) -- the caller freezes it
        (`model.requires_grad_(False)`)
    :param tokenizer: the model's tokenizer
    :param train_store: `VectorStore` `train_examples`' vector indices address
    :param train_examples: the train-split pool (never includes val examples)
    :param val_store: `VectorStore` `val_examples`' vector indices address
    :param val_examples: the full val split, for the final full-val pass
    :param config: training configuration
    :param run_dir: directory for checkpoints, metrics and reports
    :param device: device to create the projection on
    :return: the final full-val report (also written to `final_eval.json`)
    """
    seed_everything(config.seed)

    hidden_size = train_store.hidden_size
    projection = create_projection_module(
        projection_type=config.projection_type,
        dim=hidden_size,
        normalize_input=config.normalize_input,
        device=device,
        init_scale=config.init_scale,
        low_rank_rank=config.projection_rank,
        low_rank_init_factor=config.low_rank_init_factor,
    )
    loss_config = LossConfig(
        max_loss=config.max_loss,
        label_smoothing=config.label_smoothing,
        strip_labels=config.strip_labels,
    )
    scorer = SoftPromptLoss(model, tokenizer, projection, loss_config)

    total_steps = compute_total_steps(config.budget_examples, config.batch_size)
    steps_to_run = (
        min(total_steps, config.max_steps)
        if config.max_steps is not None
        else total_steps
    )

    val_subsample = subsample(val_examples, config.val_subsample, config.seed)
    check_validation_compute_ratio(
        val_subsample_size=len(val_subsample),
        batch_size=config.batch_size,
        micro_batch_size=config.micro_batch_size,
        validate_every=config.validate_every,
        steps_to_check=steps_to_run,
    )

    target_lengths = compute_target_lengths(train_examples, tokenizer, loss_config)
    stream = example_stream(train_examples, config.seed)
    batches = bucketed_batches(
        stream, config.batch_size, config.buffer_batches, target_lengths, config.seed
    )
    optimizer = build_optimizer(
        projection, lr=config.lr, weight_decay=config.weight_decay
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_config = checkpoint_config(config, total_steps=total_steps)
    best_val_loss = float("inf")

    with open(run_dir / "metrics.jsonl", "w") as metrics_handle:
        for step in range(steps_to_run):
            lr = lr_at_step(
                step,
                base_lr=config.lr,
                warmup_steps=config.warmup_steps,
                total_steps=total_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            batch = next(batches)
            train_loss, grad_norm = optimizer_step(
                batch,
                train_store,
                scorer,
                optimizer,
                config.micro_batch_size,
                config.grad_clip,
            )

            global_step = step + 1
            examples_seen = global_step * config.batch_size
            is_last_step = global_step == steps_to_run
            if global_step % config.validate_every == 0 or is_last_step:
                val_result = evaluate(
                    val_store, val_subsample, scorer, config.batch_size
                )
                val_loss = val_result["measured_loss"]
                record = {
                    "examples_seen": examples_seen,
                    "step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "scale": _metric(projection, "get_scale"),
                    "bias_norm": _metric(projection, "get_bias_norm"),
                }
                metrics_handle.write(json.dumps(record) + "\n")
                metrics_handle.flush()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        run_dir / "best.pt",
                        projection,
                        ckpt_config,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                    )
                save_checkpoint(
                    run_dir / "last.pt",
                    projection,
                    ckpt_config,
                    global_step=global_step,
                    best_val_loss=(
                        best_val_loss if best_val_loss < float("inf") else None
                    ),
                )

    final_result = evaluate(val_store, val_examples, scorer, config.batch_size)
    final_report = {
        **final_result,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
        "global_step": steps_to_run,
        "total_steps": total_steps,
    }
    with open(run_dir / "final_eval.json", "w") as handle:
        json.dump(final_report, handle, indent=2)

    return final_report


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def write_run_config(run_dir: Path, args, *, total_steps: int) -> None:
    """`run_config.json`: every CLI arg, the resolved step count, and enough
    provenance (the vectors dir, its `position_means.pt`, the git commit) to
    trace a checkpoint back to the centring it was trained under (parent plan
    S6 step 2, item 9).
    """
    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config["resolved_total_steps"] = total_steps
    config["position_means_path"] = str(args.vectors / "position_means.pt")
    config["git_commit"] = _git_commit()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_config.json", "w") as handle:
        json.dump(config, handle, indent=2, default=str)


def load_train_and_val(
    vectors_dir: Path, *, pool_positions: bool, restrict_to: Path | None
):
    """Build `(train_store, train_examples, val_store, val_examples)`,
    loading `vectors.pt` exactly once regardless of `pool_positions`
    (`load_vector_store`/`pooled_vector_store` are otherwise each called once
    per split, which would load the -- potentially multi-GB, parent plan
    S4.4 -- raw vector table twice for no reason).

    :param vectors_dir: an extraction output directory
    :param pool_positions: arm C -- pool each topic's vectors into one
    :param restrict_to: intersect topics with this directory's own topic set
    """
    records = load_topic_records(vectors_dir)
    if restrict_to is not None:
        other_titles = {record.title for record in load_topic_records(restrict_to)}
        records = restrict_to_titles(records, other_titles)

    if pool_positions:
        store, examples = pooled_vector_store(vectors_dir, records=records, split=None)
        train_indices = {
            i for i, record in enumerate(records) if record.split == "train"
        }
        val_indices = {i for i, record in enumerate(records) if record.split == "val"}
        train_examples = [e for e in examples if e.vector_index in train_indices]
        val_examples = [e for e in examples if e.vector_index in val_indices]
        return store, train_examples, store, val_examples

    store = load_vector_store(vectors_dir, center=True)
    train_examples = examples_from_records(records, "train")
    val_examples = examples_from_records(records, "val")
    return store, train_examples, store, val_examples


def main(args) -> dict:
    if args.ddp:
        raise NotImplementedError(
            "DDP is not implemented. The default multi-GPU path is one run per "
            "GPU (parent plan S4.3): launch one process per card with --device."
        )

    from model_loading import load_base_model, load_tokenizer

    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, device=args.device, dtype=args.dtype)
    model.requires_grad_(False)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_store, train_examples, val_store, val_examples = load_train_and_val(
        args.vectors,
        pool_positions=args.pool_positions,
        restrict_to=args.restrict_topics_to,
    )
    print(
        f"{len(train_examples)} train examples, {len(val_examples)} val examples "
        f"({'pooled' if args.pool_positions else 'per-position'})"
    )

    config = TrainConfig.from_args(args)
    total_steps = compute_total_steps(config.budget_examples, config.batch_size)
    write_run_config(args.run_dir, args, total_steps=total_steps)

    result = train(
        model=model,
        tokenizer=tokenizer,
        train_store=train_store,
        train_examples=train_examples,
        val_store=val_store,
        val_examples=val_examples,
        config=config,
        run_dir=args.run_dir,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main(args)
