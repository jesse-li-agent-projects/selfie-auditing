"""Trains one projection to a budget expressed in examples seen, validates on
a fixed subsample, and writes checkpoints `adapter_training.load_adapter` can
read -- so `interpret.py` and every other downstream consumer keep working
unchanged regardless of who trained the file.

    python -m adapter_training.train_adapter \\
        --vectors vectors/bg_think_l19 \\
        --run-dir runs/bg_think_scalar_affine \\
        --budget-examples 755391 \\
        --batch-size 256 --micro-batch-size 64 \\
        --projection-type scalar_affine \\
        --lr 0.01 --init-scale 5.0 --warmup-steps 10 --grad-clip 0.5 --seed 42 \\
        --val-subsample 5000 --validate-every 100 \\
        --pool-positions            # one pooled vector per topic instead of per position

**Budget is examples seen, never epochs** (`compute_total_steps`).

**A stopped run is restarted with the same command plus `--resume`**, which
carries on from the last validation point; without the flag the run starts
over from step 0.

**This trainer always uses centred vectors**: that is what upstream's own
`validate()` scored. Raw vectors are a downstream-interpretation-time concern
(`interpret.py`), never a training one.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Light imports: config.py pulls in no heavy dependencies, so --help stays fast.
from config import BASE_MODEL_8B


def parse_vectors_k(values: list[str]) -> dict[str, Path]:
    """Parse repeated `--vectors-k K=DIR` values into `{"kK": outputs/DIR}`.

    Sources are keyed by name everywhere below, so each k becomes the source
    name `"k1"`/`"k2"`/`"k3"`, ordered smallest k first -- which is the order
    `--mixture-ratio` was always read in, so an archived `--vectors-k` command
    keeps its meaning.

    :param values: raw `"K=DIR"` strings, one per `--vectors-k` occurrence
    :raises ValueError: on a malformed entry or a repeated k
    """
    ks: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--vectors-k expects K=DIR, got {value!r}")
        key, _, raw_dir = value.partition("=")
        k = int(key)
        if k in ks:
            raise ValueError(f"--vectors-k given twice for k={k}")
        ks[k] = Path("outputs") / raw_dir
    return {f"k{k}": ks[k] for k in sorted(ks)}


def parse_mixture_ratio(text: str, names: list[str]) -> dict[str, int]:
    """Parse `"1:2:3"` into `{source: weight}`, in `names`' order.

    :param text: colon-separated weights, in the order the sources were given
    :param names: the source names, in the order they were given
    :raises ValueError: if the weight count does not match `len(names)`
    """
    parts = text.split(":")
    if len(parts) != len(names):
        raise ValueError(
            f"--mixture-ratio has {len(parts)} entries but {len(names)} sources "
            f"were given ({names})"
        )
    return {name: int(weight) for name, weight in zip(names, parts)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a SelfIE adapter projection to an examples-seen budget."
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
        "--vectors-k 2=bg_think_many_l19_k2 --vectors-k 3=bg_think_many_l19_k3' "
        "(DIR written under outputs/, implicitly prepended). Repeatable. "
        "Mutually exclusive with --vectors",
    )
    parser.add_argument(
        "--mixture-ratio",
        default="1:2:3",
        help="colon-separated example-count ratio, in the same k order as "
        "--vectors-k's smallest-to-largest k (bg_think_many's D6 default 1:2:3)",
    )
    parser.add_argument(
        "--val-total-examples",
        type=int,
        default=None,
        help="--vectors-k only: size of the full val pool sampled for "
        "final_eval.json (periodic validation still subsamples --val-subsample "
        "from it). Required with --vectors-k",
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
        help="examples seen, not epochs",
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
        "field, never a literal in code",
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
        "--log-every",
        type=int,
        default=50,
        help="steps between metrics.jsonl records; must divide --validate-every",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop the loop early; does NOT change the cosine schedule's "
        "horizon, so a debug run exercises the real run's schedule",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="pick the run back up from --run-dir's resume.pt if one is there, "
        "and start fresh if not -- so a crashed run is relaunched with the "
        "same command it was started with",
    )
    parser.add_argument(
        "--pool-positions",
        action="store_true",
        help="mean each topic's positions into one vector before the adapter",
    )
    parser.add_argument(
        "--restrict-topics-to",
        type=lambda value: Path("outputs") / value,
        default=None,
        help="intersect --vectors' topics with this directory's own topic set",
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
        help="length-bucketing shuffle buffer, in batches",
    )
    parser.add_argument("--model", default=BASE_MODEL_8B)
    parser.add_argument(
        "--device",
        default="cuda",
        help="'cuda:N' to pick which GPU this run uses (no DDP: run one "
        "process per GPU for multiple concurrent runs), or 'auto' to shard "
        "one model across every visible GPU",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="off by default: a ~1.5x tax that also nulls past_key_values, "
        "which blocks the prefix cache",
    )
    parsed = parser.parse_args()

    if (parsed.vectors is None) == (parsed.vectors_k is None):
        parser.error("exactly one of --vectors or --vectors-k is required")
    if parsed.vectors_k is not None:
        try:
            parsed.vectors_k = parse_vectors_k(parsed.vectors_k)
            parsed.mixture_ratio = parse_mixture_ratio(
                parsed.mixture_ratio, list(parsed.vectors_k)
            )
        except ValueError as exc:
            parser.error(str(exc))
        if parsed.val_total_examples is None:
            parser.error("--val-total-examples is required with --vectors-k")
    return parsed


# Parsed before the heavy imports below, so `--help` costs no torch import.
args = parse_args() if __name__ == "__main__" else None

import itertools
import json
import random
import subprocess
from dataclasses import asdict, dataclass

import numpy as np
import torch

from adapter_training.checkpoints import (
    load_resume_state,
    save_checkpoint,
    save_resume_state,
)
from adapter_training.dataset import (
    Example,
    examples_from_records,
    load_records,
    load_vector_store,
    pooled_vector_store,
)
from adapter_training.grouped_examples import Mixture, build_mixture
from adapter_training.loss import (
    LossConfig,
    SoftPromptLoss,
    evaluate,
    subsample,
    target_text,
)
from adapter_training.projection import create_projection_module


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
    log_every: int = 50
    max_steps: int | None = None
    label_smoothing: float = 0.0
    max_loss: float = 100.0
    strip_labels: bool = True
    buffer_batches: int = 50

    def __post_init__(self):
        # A validation step that is not also a log step would write a record
        # without `train_loss_mean` and leave the accumulator spanning it.
        assert self.validate_every % self.log_every == 0, (
            f"validate_every ({self.validate_every}) must be a multiple of "
            f"log_every ({self.log_every})"
        )

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
            log_every=args.log_every,
            max_steps=args.max_steps,
            label_smoothing=args.label_smoothing,
            max_loss=args.max_loss,
            strip_labels=args.strip_labels,
            buffer_batches=args.buffer_batches,
        )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch.

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
    recorded `global_step`.
    """
    return math.ceil(budget_examples / batch_size)


def lr_at_step(
    step: int, *, base_lr: float, warmup_steps: int, total_steps: int
) -> float:
    """Closed-form reproduction of upstream's chained `LinearLR` warmup ->
    `CosineAnnealingLR` (`resources/selfie-adapters/training/trainer.py`).

    The one subtlety: upstream steps its schedulers *after* the optimizer, so
    step `warmup_steps` lands exactly on the peak learning rate rather than
    one cosine step past it.

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
    """Tokenize every distinct label's target text once and cache it, for the
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
    into a fresh permutation whenever exhausted.

    One pass is one full, non-repeating draw of every example in a fixed
    random order -- never a resample. Whether a run's budget covers a whole
    pass, several, or a fraction of one is the caller's business.

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
    """Length-bucketed batching: fill a shuffle buffer of `buffer_batches`
    batches from `stream`, sort the buffer by target length, cut it into
    batches, shuffle the batch order, and emit.

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
    """Port of upstream's `_check_validation_compute_ratio`
    (`resources/selfie-adapters/training/trainer.py`): refuse to start if
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


def micro_batches(
    batch: list[Example],
    micro_batch_size: int,
    target_lengths: dict[str, int] | None,
    template_len: int,
    max_target_len: int,
) -> list[list[Example]]:
    """Chunk a global batch into micro-batches of bounded peak memory.

    A micro-batch's activation cost scales with `examples x (template_len +
    its longest target)`, and `bucketed_batches` groups by length, so a
    fixed example count makes the longest bucket -- whose targets can be
    several times the median -- the only one that has to fit. Budgeting on
    that product instead lets short-target batches take more examples at the
    same peak, and shrinks the long ones that would otherwise OOM.

    `micro_batch_size` therefore sets the count at the *worst* target length
    in the pool; shorter batches get proportionally more. The count is capped
    at twice it because attention memory grows faster than linearly in
    length, which makes a linear budget optimistic at the short end.

    :param batch: one global batch
    :param micro_batch_size: examples per micro-batch at `max_target_len`
    :param target_lengths: target token length per label, from
        `compute_target_lengths`; `None` falls back to fixed-size chunks
    :param template_len: the interpretation template's token length
    :param max_target_len: the longest target in the pool
    :return: the micro-batches, in order, together covering `batch` exactly
    """
    if target_lengths is None:
        return [
            batch[start : start + micro_batch_size]
            for start in range(0, len(batch), micro_batch_size)
        ]
    budget = micro_batch_size * (template_len + max_target_len)
    cap = 2 * micro_batch_size
    chunks: list[list[Example]] = []
    current: list[Example] = []
    current_max = 0
    for example in batch:
        length = target_lengths[example.label]
        width = template_len + max(current_max, length)
        if current and (len(current) + 1 > cap or (len(current) + 1) * width > budget):
            chunks.append(current)
            current, current_max = [], 0
            width = template_len + length
        current.append(example)
        current_max = max(current_max, length)
    if current:
        chunks.append(current)
    return chunks


def optimizer_step(
    batch: list[Example],
    store,
    scorer: SoftPromptLoss,
    optimizer: torch.optim.Optimizer,
    micro_batch_size: int,
    grad_clip: float,
    target_lengths: dict[str, int] | None = None,
    template_len: int = 0,
    max_target_len: int = 0,
) -> tuple[float, float]:
    """One global-batch optimizer step via gradient accumulation.

    Each micro-batch's loss is scaled by `micro_len / batch_len` rather than
    averaged, so the accumulated gradient stays equivalent to a single batch
    of `len(batch)` regardless of how it was chunked -- micro-batches from a
    bucketed batch happen to have equal length here, but this does not rely
    on that.

    :param batch: one global batch, in the order to micro-batch it
    :param store: the `VectorStore` `batch`'s vector indices address
    :param scorer: bound to the model, tokenizer and trainable projection
    :param optimizer: stepped once, over the whole accumulated gradient
    :param micro_batch_size: examples per micro-batch, at `max_target_len`
    :param grad_clip: max gradient norm, applied to the projection's own parameters
    :param target_lengths: enables length-aware micro-batching -- see
        `micro_batches`; `None` keeps fixed-size chunks
    :param template_len: the interpretation template's token length
    :param max_target_len: the longest target in the pool
    :return: `(batch loss, gradient norm)`, both as floats
    """
    batch_len = len(batch)
    optimizer.zero_grad()
    total_loss = 0.0
    for micro in micro_batches(
        batch, micro_batch_size, target_lengths, template_len, max_target_len
    ):
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
    checkpoint -- everything `adapter_training.load_adapter` needs to
    reconstruct the projection, plus the run's own training settings.
    """
    return {
        "projection": {
            "type": config.projection_type,
            "normalize_input": True,
            "init_scale": config.init_scale,
            "low_rank_rank": config.projection_rank,
            "low_rank_init_factor": config.low_rank_init_factor,
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


def restore_resume_state(
    path: Path, projection, optimizer, config: TrainConfig
) -> tuple[int, float]:
    """Load a resume state into `projection` and `optimizer` in place.

    :param path: a `checkpoints.save_resume_state` file
    :param projection: the freshly created projection to overwrite
    :param optimizer: its freshly built optimizer, whose moments to restore
    :param config: the config this process was started with
    :return: `(step to carry on from, best validation loss so far)`
    :raises ValueError: if the state was written under a different config --
        carrying on under changed hyperparameters gives a trajectory neither
        config describes, and no later reader could tell
    """
    state = load_resume_state(path)
    saved, current = dict(state["train_config"]), asdict(config)
    # `max_steps` bounds how much of the schedule one process runs, never the
    # schedule or the batches, so it is free to change across a resume.
    saved.pop("max_steps", None)
    current.pop("max_steps", None)
    differing = sorted(
        key
        for key in saved.keys() | current.keys()
        if saved.get(key) != current.get(key)
    )
    if differing:
        changes = ", ".join(
            f"{key} ({saved.get(key)!r} -> {current.get(key)!r})" for key in differing
        )
        raise ValueError(f"{path} was written under a different config: {changes}")
    projection.load_state_dict(state["projection_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return state["global_step"], state["best_val_loss"]


def truncate_metrics_after(path: Path, global_step: int) -> None:
    """Drop `metrics.jsonl` records past `global_step`.

    Records are written more often than resume state is saved, so a resume
    replays steps that were already logged; without this they appear twice.

    :param path: the `metrics.jsonl` to rewrite in place
    :param global_step: the last step to keep
    """
    if not path.exists():
        return
    with open(path) as handle:
        kept = [line for line in handle if json.loads(line)["step"] <= global_step]
    with open(path, "w") as handle:
        handle.writelines(kept)


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
    resume: bool = False,
    val_source_ranges: dict[str, tuple[int, int]] | None = None,
) -> dict:
    """The training loop: seeding, schedule, sampling, micro-batching,
    validation and checkpointing. Callable directly (as tests do, with a
    fake model and a hand-built store) or via `main`.

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
    :param resume: carry on from `run_dir`'s resume state if there is one,
        which costs up to `validate_every` steps of redone work but needs
        nothing saved per step; a run with no resume state starts fresh
    :param val_source_ranges: mixture runs only -- each source's
        `(start, end)` slice of `val_examples`, so the final report also
        scores each source's own val slice beside the whole-mixture number
        `measured_loss` keeps. Never pooled into a headline figure
    :return: the final full-val report (also written to `final_eval.json`)
    """
    seed_everything(config.seed)

    hidden_size = train_store.hidden_size
    projection = create_projection_module(
        projection_type=config.projection_type,
        dim=hidden_size,
        normalize_input=True,
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
    max_target_len = max(target_lengths.values())
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
    resume_path = run_dir / "resume.pt"

    start_step = 0
    if resume and resume_path.exists():
        start_step, best_val_loss = restore_resume_state(
            resume_path, projection, optimizer, config
        )
        # The batch stream is a pure function of the seed, so replaying it --
        # cheap, no model involved -- is all it takes to land on the batch
        # this step would have drawn had the run never stopped.
        for _ in range(start_step):
            next(batches)
        truncate_metrics_after(run_dir / "metrics.jsonl", start_step)
        print(f"resuming at step {start_step}, best val loss {best_val_loss:.4f}")

    train_loss_accum = 0.0
    train_loss_count = 0
    with open(run_dir / "metrics.jsonl", "a" if start_step else "w") as metrics_handle:
        for step in range(start_step, steps_to_run):
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
                target_lengths=target_lengths,
                template_len=scorer.template_len,
                max_target_len=max_target_len,
            )
            train_loss_accum += train_loss
            train_loss_count += 1

            global_step = step + 1
            examples_seen = global_step * config.batch_size
            is_last_step = global_step == steps_to_run
            should_log = global_step % config.log_every == 0 or is_last_step
            should_validate = global_step % config.validate_every == 0 or is_last_step

            if should_log or should_validate:
                record = {
                    "examples_seen": examples_seen,
                    "step": global_step,
                    "train_loss": train_loss,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "scale": _metric(projection, "get_scale"),
                    "bias_norm": _metric(projection, "get_bias_norm"),
                    "low_rank_norm": _metric(projection, "get_low_rank_norm"),
                    "low_rank_to_diagonal_ratio": _metric(
                        projection, "get_low_rank_to_diagonal_ratio"
                    ),
                }
                if should_log:
                    record["train_loss_mean"] = train_loss_accum / train_loss_count
                    train_loss_accum = 0.0
                    train_loss_count = 0

                if should_validate:
                    val_result = evaluate(
                        val_store, val_subsample, scorer, config.micro_batch_size
                    )
                    val_loss = val_result["measured_loss"]
                    record["val_loss"] = val_loss

                metrics_handle.write(json.dumps(record) + "\n")
                metrics_handle.flush()

                if should_validate:
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
                    save_resume_state(
                        resume_path,
                        projection,
                        optimizer,
                        train_config=asdict(config),
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                    )

    final_result = evaluate(val_store, val_examples, scorer, config.micro_batch_size)
    final_report = {
        **final_result,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
        "global_step": steps_to_run,
        "total_steps": total_steps,
    }
    if val_source_ranges is not None:
        final_report["val_loss_by_source"] = {
            name: evaluate(
                val_store, val_examples[start:end], scorer, config.micro_batch_size
            )
            for name, (start, end) in val_source_ranges.items()
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


def write_run_config(
    run_dir: Path,
    args,
    *,
    total_steps: int,
    mixture_source_ranges: dict[str, tuple[int, int]] | None = None,
    semicolon_drops: dict[str, int] | None = None,
) -> None:
    """`run_config.json`: every CLI arg, the resolved step count, and enough
    provenance (the vectors dir(s), their `position_means.pt`, the git
    commit) to trace a checkpoint back to the centring it was trained under.

    :param mixture_source_ranges: mixture runs only -- each source's
        `(start, end)` slice of the train example list, so a later step does
        not have to recompute the mixture to find them
    :param semicolon_drops: mixture runs only -- how many topics the
        `;`-label filter dropped per source, which a pre-run check reports
    """
    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config["resolved_total_steps"] = total_steps
    if args.vectors_k is not None:
        config["vectors_k"] = {name: str(d) for name, d in args.vectors_k.items()}
        config["position_means_paths"] = {
            name: str(d / "position_means.pt") for name, d in args.vectors_k.items()
        }
        if mixture_source_ranges is not None:
            config["mixture_source_ranges"] = {
                name: list(v) for name, v in mixture_source_ranges.items()
            }
        if semicolon_drops is not None:
            config["semicolon_drops"] = dict(semicolon_drops)
    else:
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
    per split, which would load the -- potentially multi-GB -- raw vector
    table twice for no reason).

    :param vectors_dir: an extraction output directory
    :param pool_positions: pool each topic's vectors into one
    :param restrict_to: intersect topics with this directory's own topic set
    """
    records = load_records(vectors_dir, restrict_to)

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


def load_grouped_train_and_val(
    directories: dict[str, Path],
    *,
    ratio: dict[str, int],
    budget_examples: int,
    val_total_examples: int,
    seed: int,
) -> tuple[Mixture, Mixture]:
    """The mixture counterpart to `load_train_and_val`: several named sources
    in one ratio, sampled once per split (`build_mixture`).

    Train and val each get their own `VectorStore` -- the same directories
    loaded and centred twice, rather than one store shared across splits
    complicating the indexing for no memory saving worth the complexity at
    this scale.

    :param directories: source name -> extraction output directory
    :param ratio: relative example count per source
    :param budget_examples: total train examples across every source
    :param val_total_examples: total val examples across every source -- the
        "full" val pool `train()` scores at the end (`final_eval.json`);
        periodic validation subsamples `--val-subsample` from it
    :param seed: seeds train and val sampling independently
    :return: the train and val `Mixture`s; the val one's `source_ranges` is
        what lets `train()` score each source's own val slice
    """
    train = build_mixture(directories, "train", budget_examples, ratio=ratio, seed=seed)
    val = build_mixture(
        directories, "val", val_total_examples, ratio=ratio, seed=f"{seed}-val"
    )
    return train, val


def main(args) -> dict:
    from model_loading import load_base_model, load_tokenizer, resolve_device

    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, device=args.device, dtype=args.dtype)
    device = resolve_device(model)
    model.requires_grad_(False)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    mixture_source_ranges = None
    val_source_ranges = None
    semicolon_drops = None
    if args.vectors_k is not None:
        if args.pool_positions or args.restrict_topics_to is not None:
            raise ValueError(
                "--pool-positions and --restrict-topics-to are not supported "
                "with a mixture"
            )
        train_mixture, val_mixture = load_grouped_train_and_val(
            args.vectors_k,
            ratio=args.mixture_ratio,
            budget_examples=args.budget_examples,
            val_total_examples=args.val_total_examples,
            seed=args.seed,
        )
        train_store, train_examples = train_mixture.store, train_mixture.examples
        val_store, val_examples = val_mixture.store, val_mixture.examples
        mixture_source_ranges = train_mixture.source_ranges
        val_source_ranges = val_mixture.source_ranges
        semicolon_drops = train_mixture.semicolon_drops
    else:
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
    write_run_config(
        args.run_dir,
        args,
        total_steps=total_steps,
        mixture_source_ranges=mixture_source_ranges,
        semicolon_drops=semicolon_drops,
    )

    result = train(
        model=model,
        tokenizer=tokenizer,
        train_store=train_store,
        train_examples=train_examples,
        val_store=val_store,
        val_examples=val_examples,
        config=config,
        run_dir=args.run_dir,
        device=device,
        resume=args.resume,
        val_source_ranges=val_source_ranges,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main(args)
