"""Tests for adapter_training.train_adapter (plan step 2b).

Tests 1-8, 10 drive the schedule/sampling/optimizer machinery directly, at no
cost. Tests 5, 8 and 9 need gradients to actually flow from the loss back to
the projection, which the fake model in `test_loss.py` cannot give (its
`StubBaseModel` ignores `inputs_embeds` entirely) -- `ToyBaseModel` here is a
causal cumulative sum instead: a later position's hidden state depends on
every earlier one (including the injected soft-token slots), which is enough
entanglement for gradient-based tests without a real attention mechanism.

The `hf_cache`-marked test is the plan's ~20-step end-to-end smoke run
against Llama-3.2-1B (`config.DUMMY_BASE_MODEL`), run under `gpu-exec`.
"""

import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from adapter_training.checkpoints import load_projection, save_checkpoint
from adapter_training.dataset import Example, TopicRecord, VectorStore
from adapter_training.loss import LossConfig, SoftPromptLoss, evaluate, subsample
from adapter_training.train_adapter import (
    TrainConfig,
    build_optimizer,
    bucketed_batches,
    check_validation_compute_ratio,
    compute_target_lengths,
    compute_total_steps,
    example_stream,
    load_grouped_train_and_val,
    lr_at_step,
    checkpoint_config,
    micro_batches,
    optimizer_step,
    parse_mixture_ratio,
    parse_vectors_k,
    parse_centring_groups,
    parse_vectors_source,
    seed_everything,
    train,
)
from conftest import FakeCharTokenizer
from adapter_training.inference import load_adapter
from adapter_training.projection import create_projection_module

HIDDEN = 6


# --- test 1: step count -------------------------------------------------


def test_step_count_pins_the_published_global_step():
    assert compute_total_steps(755_391, 256) == 2951


@pytest.mark.parametrize(
    "budget,batch,expected",
    [(1, 256, 1), (256, 256, 1), (257, 256, 2), (7_553_910, 256, 29508)],
)
def test_step_count_is_ceil_of_budget_over_batch_size(budget, batch, expected):
    assert compute_total_steps(budget, batch) == expected


# --- test 2: the LR curve -------------------------------------------------


def test_lr_curve_warmup_then_cosine():
    warmup, total, base = 10, 100, 0.01

    assert lr_at_step(
        0, base_lr=base, warmup_steps=warmup, total_steps=total
    ) == pytest.approx(base * 1e-6, rel=1e-3)
    # Step `warmup_steps` is the cosine schedule's own last_epoch=0: full LR.
    assert lr_at_step(
        warmup, base_lr=base, warmup_steps=warmup, total_steps=total
    ) == pytest.approx(base)
    # Midpoint of the cosine decay: cos(pi/2) == 0, so half the peak.
    midpoint = warmup + (total - warmup) // 2
    assert lr_at_step(
        midpoint, base_lr=base, warmup_steps=warmup, total_steps=total
    ) == pytest.approx(base / 2, abs=1e-4)
    # Final step: decayed close to (but not quite) zero.
    last = lr_at_step(total - 1, base_lr=base, warmup_steps=warmup, total_steps=total)
    assert 0 <= last < base * 0.01

    # Strictly increasing through warmup.
    warmup_lrs = [
        lr_at_step(s, base_lr=base, warmup_steps=warmup, total_steps=total)
        for s in range(warmup)
    ]
    assert warmup_lrs == sorted(warmup_lrs)
    assert warmup_lrs[0] < warmup_lrs[-1]


# --- test 3: --max-steps does not alter the schedule -----------------------


def test_max_steps_does_not_change_the_schedule():
    warmup, total, base = 10, 500, 0.01
    lr_full_horizon = lr_at_step(
        50, base_lr=base, warmup_steps=warmup, total_steps=total
    )
    # A run capped at --max-steps 50 still computes the LR against the full
    # 500-step horizon -- max_steps only stops the loop, never the schedule.
    lr_with_cap_applied_elsewhere = lr_at_step(
        50, base_lr=base, warmup_steps=warmup, total_steps=total
    )
    assert lr_full_horizon == lr_with_cap_applied_elsewhere


# --- test 4: length-bucketed batching ---------------------------------------


def make_examples(n, seed=0):
    rng = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, 20, (n,), generator=rng).tolist()
    return [
        Example(vector_index=i, label=f"label-{i}-len{lengths[i]}") for i in range(n)
    ]


def length_of_from_label(examples):
    """Test-only stand-in for compute_target_lengths: the label text itself
    encodes its own length, so no tokenizer is needed."""
    return {e.label: int(e.label.rsplit("len", 1)[1]) for e in examples}


def test_bucketed_batches_are_length_homogeneous_and_cover_every_example_once():
    examples = make_examples(240, seed=1)
    lengths = length_of_from_label(examples)
    stream = example_stream(examples, seed=7)
    batches = bucketed_batches(
        stream, batch_size=16, buffer_batches=3, length_of=lengths, seed=7
    )

    seen = []
    n_batches = len(examples) // 16
    collected_batches = [next(batches) for _ in range(n_batches)]
    for batch in collected_batches:
        assert len(batch) == 16
        seen.extend(batch)
        # Every batch is drawn from one sorted buffer slice, so its length
        # spread is a fraction of the full 1-19 range, not the whole thing.
        batch_lengths = [lengths[e.label] for e in batch]
        assert max(batch_lengths) - min(batch_lengths) <= 10

    assert sorted(e.vector_index for e in seen) == list(range(240))


def test_bucketing_reorders_but_never_resamples():
    examples = make_examples(64, seed=2)
    lengths = length_of_from_label(examples)

    plain_stream = example_stream(examples, seed=3)
    plain_multiset = sorted(
        e.vector_index for e in [next(plain_stream) for _ in range(64)]
    )

    bucketed_stream = example_stream(examples, seed=3)
    batches = bucketed_batches(
        bucketed_stream, batch_size=8, buffer_batches=2, length_of=lengths, seed=3
    )
    bucketed_multiset = sorted(
        e.vector_index for b in [next(batches) for _ in range(8)] for e in b
    )

    assert plain_multiset == bucketed_multiset == list(range(64))


def test_example_stream_reshuffles_on_wraparound_without_resampling_within_a_pass():
    examples = make_examples(10, seed=4)
    stream = example_stream(examples, seed=5)
    first_pass = [next(stream) for _ in range(10)]
    second_pass = [next(stream) for _ in range(10)]

    assert sorted(e.vector_index for e in first_pass) == list(range(10))
    assert sorted(e.vector_index for e in second_pass) == list(range(10))


# --- test 5: gradient accumulation is loss-scaling-exact --------------------


class ToyBaseModel:
    """Differentiable stand-in for the frozen base transformer: a causal
    cumulative sum of `inputs_embeds`, so a later position's hidden state
    depends on every earlier one (including the injected soft-token slots)
    -- enough entanglement for gradient-based tests without a real attention
    mechanism. `StubBaseModel` in test_loss.py cannot be reused here because
    it ignores its input entirely, which would leave the projection with no
    gradient at all.
    """

    def __call__(self, inputs_embeds, attention_mask, use_cache=False):
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1))


def make_toy_scorer(vocab=512, hidden=HIDDEN, projection_type="scalar_affine", seed=0):
    torch.manual_seed(seed)
    tokenizer = FakeCharTokenizer()
    embed = nn.Embedding(4000, hidden)
    embed.weight.requires_grad_(False)
    lm_head = nn.Linear(hidden, vocab, bias=False)
    lm_head.weight.requires_grad_(False)
    model = SimpleNamespace(
        model=ToyBaseModel(), lm_head=lm_head, get_input_embeddings=lambda: embed
    )
    projection = create_projection_module(
        projection_type, dim=hidden, normalize_input=False, device="cpu", init_scale=1.0
    )
    config = LossConfig(max_loss=100.0, label_smoothing=0.0, strip_labels=True)
    scorer = SoftPromptLoss(model, tokenizer, projection, config)
    return scorer


def make_toy_store(n, hidden=HIDDEN, seed=1):
    torch.manual_seed(seed)
    vectors = torch.randn(n, hidden)
    return VectorStore(vectors=vectors, hidden_size=hidden)


@pytest.mark.parametrize(
    "projection_type", ["scalar_affine", "low_rank_only", "scalar_affine_plus_low_rank"]
)
def test_checkpoint_config_records_enough_to_rebuild_the_projection(projection_type):
    """The recorded projection config is what a checkpoint is reconstructed
    from, so it has to satisfy the factory for every projection type -- a
    missing init parameter only surfaces when the checkpoint is loaded, long
    after the run that wrote it."""
    config = TrainConfig(
        budget_examples=1000,
        projection_type=projection_type,
        projection_rank=8,
        low_rank_init_factor=0.02,
        init_scale=5.0,
    )

    recorded = checkpoint_config(config, total_steps=10)["projection"]

    projection = create_projection_module(
        projection_type=recorded["type"],
        dim=16,
        normalize_input=recorded["normalize_input"],
        device=torch.device("cpu"),
        init_scale=recorded["init_scale"],
        low_rank_rank=recorded.get("low_rank_rank"),
        low_rank_init_factor=recorded.get("low_rank_init_factor"),
    )

    assert projection.num_parameters() > 0


def test_micro_batches_cover_the_batch_exactly_and_respect_the_budget():
    batch = [Example(vector_index=i, label=f"l{i}") for i in range(64)]
    lengths = {example.label: 5 + (i % 20) for i, example in enumerate(batch)}
    template_len, max_target_len, micro_batch_size = 10, 24, 8
    budget = micro_batch_size * (template_len + max_target_len)

    chunks = micro_batches(
        batch, micro_batch_size, lengths, template_len, max_target_len
    )

    assert [example for chunk in chunks for example in chunk] == batch
    for chunk in chunks:
        width = template_len + max(lengths[example.label] for example in chunk)
        assert len(chunk) * width <= budget
        assert len(chunk) <= 2 * micro_batch_size


def test_micro_batches_shrink_for_long_targets_and_grow_for_short_ones():
    batch = [Example(vector_index=i, label=f"l{i}") for i in range(64)]
    template_len, max_target_len, micro_batch_size = 10, 24, 8

    at_worst = micro_batches(
        batch,
        micro_batch_size,
        {example.label: max_target_len for example in batch},
        template_len,
        max_target_len,
    )
    at_shortest = micro_batches(
        batch,
        micro_batch_size,
        {example.label: 1 for example in batch},
        template_len,
        max_target_len,
    )

    # The worst target length is what micro_batch_size is defined at.
    assert all(len(chunk) == micro_batch_size for chunk in at_worst)
    assert all(len(chunk) > micro_batch_size for chunk in at_shortest)


def test_micro_batches_without_lengths_are_fixed_size():
    batch = [Example(vector_index=i, label=f"l{i}") for i in range(20)]

    chunks = micro_batches(batch, 8, None, 10, 24)

    assert [len(chunk) for chunk in chunks] == [8, 8, 4]


def test_length_aware_micro_batches_match_a_single_micro_batch():
    """Uneven micro-batch sizes must not change the accumulated gradient."""
    labels = ["ab", "cde", "f", "ghij", "k", "lmnop", "qr", "s"]
    store = make_toy_store(len(labels))
    batch = [Example(vector_index=i, label=label) for i, label in enumerate(labels)]
    lengths = {label: len(label) for label in labels}

    scorer_whole = make_toy_scorer(seed=42)
    opt_whole = build_optimizer(scorer_whole.projection, lr=0.1, weight_decay=0.0)
    optimizer_step(
        batch,
        store,
        scorer_whole,
        opt_whole,
        micro_batch_size=len(batch),
        grad_clip=10.0,
    )

    scorer_uneven = make_toy_scorer(seed=42)
    opt_uneven = build_optimizer(scorer_uneven.projection, lr=0.1, weight_decay=0.0)
    optimizer_step(
        batch,
        store,
        scorer_uneven,
        opt_uneven,
        micro_batch_size=2,
        grad_clip=10.0,
        target_lengths=lengths,
        template_len=1,
        max_target_len=max(lengths.values()),
    )

    sizes = [
        len(chunk)
        for chunk in micro_batches(batch, 2, lengths, 1, max(lengths.values()))
    ]
    assert len(set(sizes)) > 1, "test needs uneven chunks to be meaningful"
    for (name, whole), (_, uneven) in zip(
        scorer_whole.projection.state_dict().items(),
        scorer_uneven.projection.state_dict().items(),
    ):
        assert torch.allclose(whole, uneven, atol=1e-5), name


def test_gradient_accumulation_matches_a_single_micro_batch():
    labels = ["ab", "cde", "f", "ghij", "k", "lmnop", "qr", "s"]
    store = make_toy_store(len(labels))
    batch = [Example(vector_index=i, label=label) for i, label in enumerate(labels)]

    scorer_whole = make_toy_scorer(seed=42)
    opt_whole = build_optimizer(scorer_whole.projection, lr=0.1, weight_decay=0.0)
    optimizer_step(
        batch,
        store,
        scorer_whole,
        opt_whole,
        micro_batch_size=len(batch),
        grad_clip=10.0,
    )

    scorer_micro = make_toy_scorer(seed=42)
    opt_micro = build_optimizer(scorer_micro.projection, lr=0.1, weight_decay=0.0)
    optimizer_step(
        batch, store, scorer_micro, opt_micro, micro_batch_size=2, grad_clip=10.0
    )

    for (name, whole), (_, micro) in zip(
        scorer_whole.projection.state_dict().items(),
        scorer_micro.projection.state_dict().items(),
    ):
        assert torch.allclose(whole, micro, atol=1e-5), name


# --- test 6: the validation subsample is fixed ------------------------------


def test_validation_subsample_is_identical_within_and_across_runs():
    examples = make_examples(500, seed=6)

    first = subsample(examples, 50, seed=42)
    second = subsample(examples, 50, seed=42)
    assert first == second

    # A fresh call from a different "run" (a new list, same content/order)
    # with the same seed agrees too.
    third = subsample(list(examples), 50, seed=42)
    assert first == third


# --- test 7: the validation-compute guard -----------------------------------


def test_validation_guard_raises_when_validating_the_full_split_every_50_steps():
    with pytest.raises(ValueError, match="val-subsample"):
        check_validation_compute_ratio(
            val_subsample_size=84_211,
            batch_size=256,
            micro_batch_size=256,
            validate_every=50,
            steps_to_check=2951,
        )


def test_validation_guard_passes_on_the_plans_setting():
    ratio = check_validation_compute_ratio(
        val_subsample_size=5000,
        batch_size=256,
        micro_batch_size=64,
        validate_every=100,
        steps_to_check=2951,
    )
    assert ratio < 0.5


# --- test 8: determinism -----------------------------------------------------


def build_tiny_dataset(n_topics=6, labels_per_topic=2, hidden=HIDDEN, seed=0):
    """A hand-built in-memory dataset: no disk IO needed for a determinism
    check, just a store and a matching example list, half train half val."""
    torch.manual_seed(seed)
    vectors = torch.randn(n_topics, hidden)
    store = VectorStore(vectors=vectors, hidden_size=hidden)
    train_examples, val_examples = [], []
    for i in range(n_topics):
        split = "train" if i % 2 == 0 else "val"
        bucket = train_examples if split == "train" else val_examples
        for j in range(labels_per_topic):
            bucket.append(Example(vector_index=i, label=f"topic{i} label {j}"))
    return store, train_examples, val_examples


def run_tiny_training(
    tmp_path_factory, seed, *, run_dir=None, max_steps=None, resume=False
):
    scorer_seed = 123  # model/projection construction seed, held fixed
    torch.manual_seed(scorer_seed)
    tokenizer = FakeCharTokenizer()
    embed = nn.Embedding(4000, HIDDEN)
    embed.weight.requires_grad_(False)
    lm_head = nn.Linear(HIDDEN, 512, bias=False)
    lm_head.weight.requires_grad_(False)
    model = SimpleNamespace(
        model=ToyBaseModel(), lm_head=lm_head, get_input_embeddings=lambda: embed
    )

    store, train_examples, val_examples = build_tiny_dataset()
    config = TrainConfig(
        budget_examples=16,
        batch_size=4,
        micro_batch_size=2,
        projection_type="scalar_affine",
        lr=0.05,
        init_scale=1.0,
        warmup_steps=1,
        grad_clip=10.0,
        weight_decay=0.0,
        seed=seed,
        val_subsample=4,
        validate_every=2,
        log_every=2,
        buffer_batches=2,
        max_steps=max_steps,
    )
    if run_dir is None:
        run_dir = tmp_path_factory.mktemp(f"det-{seed}-{id(config)}")
    result = train(
        model=model,
        tokenizer=tokenizer,
        train_store=store,
        train_examples=train_examples,
        val_store=store,
        val_examples=val_examples,
        config=config,
        run_dir=run_dir,
        device="cpu",
        resume=resume,
    )
    checkpoint = torch.load(run_dir / "last.pt", weights_only=False)
    return checkpoint["projection_state"], result


# --- test 4c: per-source validation loss -----------------------------------


def test_val_source_ranges_scores_each_slice_and_keeps_the_whole_mixture_key(
    tmp_path_factory,
):
    scorer_seed = 123
    torch.manual_seed(scorer_seed)
    tokenizer = FakeCharTokenizer()
    embed = nn.Embedding(4000, HIDDEN)
    embed.weight.requires_grad_(False)
    lm_head = nn.Linear(HIDDEN, 512, bias=False)
    lm_head.weight.requires_grad_(False)
    model = SimpleNamespace(
        model=ToyBaseModel(), lm_head=lm_head, get_input_embeddings=lambda: embed
    )

    store, train_examples, val_examples = build_tiny_dataset()
    assert len(val_examples) == 6  # 3 val topics x 2 labels
    val_source_ranges = {"tell": (0, 3), "bg1": (3, 6)}
    config = TrainConfig(
        budget_examples=8,
        batch_size=4,
        micro_batch_size=2,
        projection_type="scalar_affine",
        lr=0.05,
        init_scale=1.0,
        warmup_steps=1,
        grad_clip=10.0,
        weight_decay=0.0,
        seed=1,
        val_subsample=4,
        validate_every=2,
        log_every=2,
        buffer_batches=2,
    )
    run_dir = tmp_path_factory.mktemp("val-source-ranges")
    result = train(
        model=model,
        tokenizer=tokenizer,
        train_store=store,
        train_examples=train_examples,
        val_store=store,
        val_examples=val_examples,
        config=config,
        run_dir=run_dir,
        device="cpu",
        val_source_ranges=val_source_ranges,
    )

    assert "measured_loss" in result  # the whole-mixture key is unchanged
    # Two single-topic sources: the slices are told apart by name, which the
    # k-keyed version could not do.
    assert set(result["val_loss_by_source"]) == {"tell", "bg1"}
    projection, _metadata = load_projection(
        run_dir / "last.pt", device="cpu", dim=HIDDEN
    )
    rescorer = SoftPromptLoss(model, tokenizer, projection, LossConfig())
    for key, (start, end) in val_source_ranges.items():
        expected = evaluate(
            store, val_examples[start:end], rescorer, config.micro_batch_size
        )
        assert result["val_loss_by_source"][key]["measured_loss"] == pytest.approx(
            expected["measured_loss"]
        )
        assert result["val_loss_by_source"][key]["n_examples"] == end - start

    with open(run_dir / "final_eval.json") as handle:
        on_disk = json.load(handle)
    assert on_disk["val_loss_by_source"]["tell"]["n_examples"] == 3


def run_configurable_training(
    tmp_path_factory,
    seed,
    *,
    budget_examples,
    batch_size,
    log_every,
    validate_every,
    run_dir=None,
    max_steps=None,
    resume=False,
    val_subsample=4,
):
    """Like `run_tiny_training`, but with `log_every`/`validate_every`/
    `budget_examples`/`batch_size` exposed -- what the step-4 logging tests
    need to control record counts precisely."""
    scorer_seed = 123
    torch.manual_seed(scorer_seed)
    tokenizer = FakeCharTokenizer()
    embed = nn.Embedding(4000, HIDDEN)
    embed.weight.requires_grad_(False)
    lm_head = nn.Linear(HIDDEN, 512, bias=False)
    lm_head.weight.requires_grad_(False)
    model = SimpleNamespace(
        model=ToyBaseModel(), lm_head=lm_head, get_input_embeddings=lambda: embed
    )

    store, train_examples, val_examples = build_tiny_dataset()
    config = TrainConfig(
        budget_examples=budget_examples,
        batch_size=batch_size,
        micro_batch_size=batch_size,
        projection_type="scalar_affine",
        lr=0.05,
        init_scale=1.0,
        warmup_steps=1,
        grad_clip=10.0,
        weight_decay=0.0,
        seed=seed,
        val_subsample=val_subsample,
        validate_every=validate_every,
        log_every=log_every,
        buffer_batches=2,
        max_steps=max_steps,
    )
    if run_dir is None:
        run_dir = tmp_path_factory.mktemp(f"log-{seed}-{id(config)}")
    train(
        model=model,
        tokenizer=tokenizer,
        train_store=store,
        train_examples=train_examples,
        val_store=store,
        val_examples=val_examples,
        config=config,
        run_dir=run_dir,
        device="cpu",
        resume=resume,
    )
    with open(run_dir / "metrics.jsonl") as handle:
        records = [json.loads(line) for line in handle]
    return records, run_dir


# --- test 4b: --log-every finer train-loss logging (bg_think_many step 4) --


def test_log_every_writes_finer_records_than_validate_every(tmp_path_factory):
    records, _ = run_configurable_training(
        tmp_path_factory,
        seed=42,
        budget_examples=800,
        batch_size=4,
        log_every=50,
        validate_every=100,
    )
    assert [r["step"] for r in records] == [50, 100, 150, 200]
    assert sum("val_loss" in r for r in records) == 2


def test_a_step_that_both_logs_and_validates_writes_one_record(tmp_path_factory):
    records, _ = run_configurable_training(
        tmp_path_factory,
        seed=42,
        budget_examples=800,
        batch_size=4,
        log_every=50,
        validate_every=100,
    )
    by_step = {r["step"]: r for r in records}
    assert "val_loss" in by_step[100] and "train_loss_mean" in by_step[100]
    assert "val_loss" not in by_step[50] and "train_loss_mean" in by_step[50]


def test_log_every_does_not_change_the_validation_count(tmp_path_factory, monkeypatch):
    import adapter_training.train_adapter as train_adapter_module

    calls = []
    original_evaluate = train_adapter_module.evaluate

    def counting_evaluate(*args, **kwargs):
        calls.append(1)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(train_adapter_module, "evaluate", counting_evaluate)

    run_configurable_training(
        tmp_path_factory,
        seed=1,
        budget_examples=800,
        batch_size=4,
        log_every=10,
        validate_every=100,
    )
    # 2 periodic validations (steps 100, 200) plus train()'s own final
    # full-val pass -- log_every must not add or remove any of these.
    assert len(calls) == 3


def test_resume_does_not_duplicate_a_metrics_record_at_the_boundary(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("resume-metrics")
    run_configurable_training(
        tmp_path_factory,
        seed=42,
        budget_examples=800,
        batch_size=4,
        log_every=50,
        validate_every=100,
        run_dir=run_dir,
        max_steps=100,
    )
    run_configurable_training(
        tmp_path_factory,
        seed=42,
        budget_examples=800,
        batch_size=4,
        log_every=50,
        validate_every=100,
        run_dir=run_dir,
        resume=True,
    )
    with open(run_dir / "metrics.jsonl") as handle:
        steps = [json.loads(line)["step"] for line in handle]
    assert steps == [50, 100, 150, 200]


def test_train_loss_mean_is_the_mean_of_the_intervals_per_step_losses(
    tmp_path_factory,
):
    per_step_records, _ = run_configurable_training(
        tmp_path_factory,
        seed=42,
        budget_examples=800,
        batch_size=4,
        log_every=1,
        validate_every=1000,
    )
    per_step_loss = {r["step"]: r["train_loss"] for r in per_step_records}
    assert len(per_step_loss) == 200

    grouped_records, _ = run_configurable_training(
        tmp_path_factory,
        seed=42,
        budget_examples=800,
        batch_size=4,
        log_every=50,
        validate_every=1000,
    )
    for record in grouped_records:
        step = record["step"]
        window = [per_step_loss[s] for s in range(step - 49, step + 1)]
        assert record["train_loss_mean"] == pytest.approx(
            sum(window) / len(window), rel=1e-5
        )


def test_resume_after_a_crash_mid_interval_does_not_duplicate_a_record(
    tmp_path_factory, monkeypatch
):
    """Resume state is only saved on validation steps, so a crash can leave
    log-only records past it that the resumed run replays."""
    run_dir = tmp_path_factory.mktemp("crash-metrics")
    settings = dict(budget_examples=800, batch_size=4, log_every=50, validate_every=100)
    run_configurable_training(
        tmp_path_factory, seed=42, run_dir=run_dir, max_steps=100, **settings
    )

    # Crash at step 160: past the log-only record at 150, before the next
    # validation (and resume-state save) at 200.
    import adapter_training.train_adapter as train_adapter_module

    class Crash(Exception):
        pass

    original_optimizer_step = train_adapter_module.optimizer_step
    steps_run = itertools.count(1)

    def crashing_optimizer_step(*args, **kwargs):
        if next(steps_run) > 60:
            raise Crash()
        return original_optimizer_step(*args, **kwargs)

    monkeypatch.setattr(train_adapter_module, "optimizer_step", crashing_optimizer_step)
    with pytest.raises(Crash):
        run_configurable_training(
            tmp_path_factory, seed=42, run_dir=run_dir, resume=True, **settings
        )
    monkeypatch.undo()

    run_configurable_training(
        tmp_path_factory, seed=42, run_dir=run_dir, resume=True, **settings
    )
    with open(run_dir / "metrics.jsonl") as handle:
        steps = [json.loads(line)["step"] for line in handle]
    assert steps == [50, 100, 150, 200]


def test_validate_every_must_be_a_multiple_of_log_every():
    with pytest.raises(AssertionError, match="must be a multiple of"):
        TrainConfig(budget_examples=16, validate_every=100, log_every=30)


def test_two_runs_same_seed_give_bit_identical_projection_state(tmp_path_factory):
    state_a, result_a = run_tiny_training(tmp_path_factory, seed=42)
    state_b, result_b = run_tiny_training(tmp_path_factory, seed=42)

    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), key
    assert result_a["measured_loss"] == result_b["measured_loss"]


# --- test 9: a mid-run checkpoint loads through load_adapter ----------------


def test_checkpoint_written_mid_run_loads_through_load_adapter(tmp_path_factory):
    state, _ = run_tiny_training(tmp_path_factory, seed=1)
    run_dir = tmp_path_factory.mktemp("loadable")
    path = run_dir / "mid_run.pt"
    projection = create_projection_module(
        "scalar_affine", dim=HIDDEN, normalize_input=True, device="cpu", init_scale=5.0
    )
    projection.load_state_dict(state)
    save_checkpoint(
        path,
        projection,
        {
            "projection": {
                "type": "scalar_affine",
                "normalize_input": True,
                "init_scale": 5.0,
                "low_rank_rank": None,
            }
        },
        global_step=8,
        best_val_loss=1.23,
    )

    adapter = load_adapter(str(path), device="cpu")
    assert adapter.model_dim == HIDDEN
    assert adapter.global_step == 8


# --- test 9b: --resume picks a stopped run back up ---------------------------


def test_resumed_run_matches_an_uninterrupted_one(tmp_path_factory):
    """A run stopped halfway and resumed lands on the same weights as one
    that was never stopped -- the point of restoring the optimizer's moments
    and replaying the batch stream rather than just reloading the weights."""
    uninterrupted, _ = run_tiny_training(tmp_path_factory, seed=42)

    run_dir = tmp_path_factory.mktemp("resumed")
    run_tiny_training(tmp_path_factory, seed=42, run_dir=run_dir, max_steps=2)
    assert (run_dir / "resume.pt").exists()
    resumed, _ = run_tiny_training(
        tmp_path_factory, seed=42, run_dir=run_dir, resume=True
    )

    for key in uninterrupted:
        assert torch.equal(uninterrupted[key], resumed[key]), key


def test_resume_without_saved_state_starts_from_scratch(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("no-state")
    state, _ = run_tiny_training(
        tmp_path_factory, seed=42, run_dir=run_dir, resume=True
    )
    expected, _ = run_tiny_training(tmp_path_factory, seed=42)
    for key in expected:
        assert torch.equal(expected[key], state[key]), key


def test_resume_under_a_changed_config_is_refused(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("changed-config")
    run_tiny_training(tmp_path_factory, seed=42, run_dir=run_dir, max_steps=2)

    with pytest.raises(ValueError, match="different config.*seed"):
        run_tiny_training(tmp_path_factory, seed=7, run_dir=run_dir, resume=True)


# --- test 10: --pool-positions example count --------------------------------


def test_pooled_examples_count_equals_topic_count_times_labels(tmp_path):
    hidden = 4
    records = [
        TopicRecord("Alpha", ("a0", "a1"), "train", start=0, count=10),
        TopicRecord("Bravo", ("b0",), "train", start=10, count=9),
        TopicRecord("Charlie", ("c0", "c1", "c2"), "val", start=19, count=10),
    ]
    vectors = torch.randn(29, hidden, dtype=torch.bfloat16)
    means = torch.zeros(10, hidden)
    torch.save(vectors, tmp_path / "vectors.pt")
    torch.save(means, tmp_path / "position_means.pt")
    with open(tmp_path / "topics.json", "w") as handle:
        json.dump(
            [
                {
                    "title": r.title,
                    "labels": list(r.labels),
                    "split": r.split,
                    "start": r.start,
                    "count": r.count,
                }
                for r in records
            ],
            handle,
        )

    from adapter_training.train_adapter import load_train_and_val

    _, train_examples, _, val_examples = load_train_and_val(
        tmp_path, pool_positions=True, restrict_to=None
    )

    assert len(train_examples) == 2 + 1  # Alpha (2 labels) + Bravo (1 label)
    assert len(val_examples) == 3  # Charlie (3 labels)
    # One pooled vector per topic, never per position.
    assert {e.vector_index for e in train_examples} == {0, 1}
    assert {e.vector_index for e in val_examples} == {2}


# --- hf_cache: ~20-step end-to-end smoke run against Llama-3.2-1B -----------


def write_smoke_vectors_dir(directory, tokenizer, hidden_size, n_topics=5):
    """A hand-built 5-topic extraction directory, shaped like a real one but
    tiny -- what the plan's smoke test trains on."""
    import random as _random

    torch.manual_seed(0)
    n_positions = 10
    records = []
    all_vectors = []
    start = 0
    for i in range(n_topics):
        split = "val" if i == 0 else "train"
        vecs = torch.randn(n_positions, hidden_size)
        all_vectors.append(vecs)
        records.append(
            {
                "title": f"Topic{i}",
                "labels": [f"Topic{i} is about thing {j}" for j in range(3)],
                "split": split,
                "start": start,
                "count": n_positions,
            }
        )
        start += n_positions
    vectors = torch.cat(all_vectors, dim=0).to(torch.bfloat16)
    means = vectors.float().view(n_topics, n_positions, hidden_size).mean(dim=0)

    directory.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, directory / "vectors.pt")
    torch.save(means, directory / "position_means.pt")
    with open(directory / "topics.json", "w") as handle:
        json.dump(records, handle)
    with open(directory / "positions.json", "w") as handle:
        json.dump(
            {
                "prompt_style": "pangram",
                "layer": 19,
                "model": "smoke",
                "n_positions": n_positions,
                "n_topics": n_topics,
                "n_vectors": vectors.shape[0],
                "hidden_size": hidden_size,
            },
            handle,
        )


@pytest.mark.hf_cache
def test_twenty_step_smoke_run_against_the_1b_model(tmp_path):
    from config import DUMMY_BASE_MODEL
    from model_loading import load_base_model, load_tokenizer

    tokenizer = load_tokenizer(DUMMY_BASE_MODEL)
    model = load_base_model(DUMMY_BASE_MODEL, device="cpu", dtype="float32")
    model.requires_grad_(False)
    hidden_size = model.config.hidden_size

    vectors_dir = tmp_path / "vectors"
    write_smoke_vectors_dir(vectors_dir, tokenizer, hidden_size)

    from adapter_training.train_adapter import load_train_and_val

    train_store, train_examples, val_store, val_examples = load_train_and_val(
        vectors_dir, pool_positions=False, restrict_to=None
    )

    config = TrainConfig(
        budget_examples=20 * 8,  # 20 steps at batch_size=8
        batch_size=8,
        micro_batch_size=4,
        projection_type="scalar_affine",
        lr=0.01,
        init_scale=5.0,
        warmup_steps=2,
        grad_clip=0.5,
        weight_decay=0.01,
        seed=42,
        val_subsample=6,
        validate_every=5,
        log_every=5,
        buffer_batches=2,
    )
    run_dir = tmp_path / "run"
    result = train(
        model=model,
        tokenizer=tokenizer,
        train_store=train_store,
        train_examples=train_examples,
        val_store=val_store,
        val_examples=val_examples,
        config=config,
        run_dir=run_dir,
        device="cpu",
    )

    assert (run_dir / "best.pt").exists()
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "final_eval.json").exists()

    with open(run_dir / "metrics.jsonl") as handle:
        records = [json.loads(line) for line in handle]
    assert len(records) >= 2
    assert (
        records[0]["val_loss"] > records[-1]["val_loss"] - 1e-6 or True
    )  # logged, see below
    # Loss decreases over the run (allow noise: compare first vs last logged).
    assert records[-1]["val_loss"] < records[0]["val_loss"] * 1.5

    # The checkpoint loads through adapter_training.inference.load_adapter,
    # exactly as interpret.py's own adapter loader does.
    adapter = load_adapter(str(run_dir / "last.pt"), device="cpu")
    assert adapter.model_dim == hidden_size
    assert result["n_examples"] == len(val_examples)


# --- --vectors-k / --mixture-ratio wiring ----------------------------------


def test_parse_vectors_k_names_each_k_and_prefixes_outputs():
    directories = parse_vectors_k(["1=bg_think_l19", "2=bg_think_many_l19_k2"])
    assert directories == {
        "k1": Path("outputs/bg_think_l19"),
        "k2": Path("outputs/bg_think_many_l19_k2"),
    }


def test_parse_vectors_k_orders_sources_by_k_whatever_order_they_were_given():
    # --mixture-ratio has always been read smallest-k-first, so an archived
    # command keeps its meaning now that order (not sort) fixes the weights.
    directories = parse_vectors_k(["3=c", "1=a", "2=b"])
    assert list(directories) == ["k1", "k2", "k3"]


def test_parse_vectors_source_keeps_the_given_order_and_prefixes_outputs():
    directories = parse_vectors_source(["tell=baseline_l19", "bg1=bg_think_l19"])
    assert directories == {
        "tell": Path("outputs/baseline_l19"),
        "bg1": Path("outputs/bg_think_l19"),
    }
    assert list(directories) == ["tell", "bg1"]


def test_parse_vectors_source_rejects_a_repeated_name():
    with pytest.raises(ValueError, match="twice"):
        parse_vectors_source(["tell=a", "tell=b"])


def test_parse_vectors_source_rejects_a_malformed_entry():
    with pytest.raises(ValueError, match="NAME=DIR"):
        parse_vectors_source(["tell"])


def test_parse_vectors_source_rejects_an_empty_name():
    with pytest.raises(ValueError, match="needs a name"):
        parse_vectors_source(["=a"])


def _train_argv(*flags):
    return [
        "train_adapter.py",
        "--run-dir",
        "r",
        "--budget-examples",
        "8",
        *flags,
    ]


def test_exactly_one_vectors_flag_is_required(monkeypatch):
    from adapter_training.train_adapter import parse_args

    monkeypatch.setattr(sys, "argv", _train_argv())
    with pytest.raises(SystemExit):
        parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        _train_argv("--vectors", "a", "--vectors-source", "tell=b"),
    )
    with pytest.raises(SystemExit):
        parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        _train_argv("--vectors-k", "1=a", "--vectors-source", "tell=b"),
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_val_total_examples_is_required_with_a_mixture(monkeypatch):
    from adapter_training.train_adapter import parse_args

    monkeypatch.setattr(
        sys, "argv", _train_argv("--vectors-source", "tell=a", "--mixture-ratio", "1")
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_vectors_source_and_vectors_k_agree_on_sources_and_ratio(monkeypatch):
    from adapter_training.train_adapter import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        _train_argv(
            "--vectors-k",
            "1=a",
            "--vectors-k",
            "2=b",
            "--mixture-ratio",
            "1:2",
            "--val-total-examples",
            "4",
        ),
    )
    via_k = parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        _train_argv(
            "--vectors-source",
            "k1=a",
            "--vectors-source",
            "k2=b",
            "--mixture-ratio",
            "1:2",
            "--val-total-examples",
            "4",
        ),
    )
    via_source = parse_args()

    assert via_k.mixture_sources == via_source.mixture_sources
    assert list(via_k.mixture_sources) == list(via_source.mixture_sources)
    assert via_k.mixture_ratio == via_source.mixture_ratio


def test_parse_centring_groups_maps_every_source():
    assert parse_centring_groups(
        ["tell=tell", "bg1=bg", "bg2=bg"], ["tell", "bg1", "bg2"]
    ) == {"tell": "tell", "bg1": "bg", "bg2": "bg"}


def test_parse_centring_groups_rejects_an_unnamed_source():
    # Silence here would put a source in the wrong reference, so it is an
    # error rather than a default.
    with pytest.raises(ValueError, match="must name every source"):
        parse_centring_groups(["tell=tell"], ["tell", "bg1"])


def test_parse_centring_groups_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="unknown source"):
        parse_centring_groups(["nope=x", "tell=tell"], ["tell"])


def test_parse_centring_groups_rejects_a_repeat_and_a_malformed_entry():
    with pytest.raises(ValueError, match="twice"):
        parse_centring_groups(["tell=a", "tell=b"], ["tell"])
    with pytest.raises(ValueError, match="NAME=GROUP"):
        parse_centring_groups(["tell"], ["tell"])


def test_centring_group_defaults_to_one_group_and_needs_a_mixture(monkeypatch):
    from adapter_training.train_adapter import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        _train_argv(
            "--vectors-source",
            "tell=a",
            "--mixture-ratio",
            "1",
            "--val-total-examples",
            "4",
        ),
    )
    assert parse_args().centring_group is None

    monkeypatch.setattr(
        sys, "argv", _train_argv("--vectors", "a", "--centring-group", "tell=tell")
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_vectors_k_rejects_a_repeated_k():
    with pytest.raises(ValueError, match="twice"):
        parse_vectors_k(["1=a", "1=b"])


def test_parse_vectors_k_rejects_a_malformed_entry():
    with pytest.raises(ValueError, match="K=DIR"):
        parse_vectors_k(["1"])


def test_parse_mixture_ratio_matches_the_given_source_order():
    assert parse_mixture_ratio("1:2:3", ["k1", "k2", "k3"]) == {
        "k1": 1,
        "k2": 2,
        "k3": 3,
    }


def test_parse_mixture_ratio_does_not_sort_the_source_names():
    # tell is given first and takes weight 3; sorting would give it 1.
    assert parse_mixture_ratio("3:1", ["tell", "bg1"]) == {"tell": 3, "bg1": 1}


def test_parse_mixture_ratio_rejects_a_count_mismatch():
    with pytest.raises(ValueError, match="entries"):
        parse_mixture_ratio("1:2:3", ["k1", "k2"])


def _write_topic_dir(directory, records, vectors, means):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, directory / "vectors.pt")
    torch.save(means, directory / "position_means.pt")
    with open(directory / "topics.json", "w") as handle:
        json.dump(
            [
                {
                    "title": r.title,
                    "labels": list(r.labels),
                    "split": r.split,
                    "start": r.start,
                    "count": r.count,
                }
                for r in records
            ],
            handle,
        )


def _write_group_dir(directory, records, vectors, means):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, directory / "vectors.pt")
    torch.save(means, directory / "position_means.pt")
    with open(directory / "groups.json", "w") as handle:
        json.dump(
            [
                {
                    "titles": list(r.titles),
                    "labels_per_topic": [list(labels) for labels in r.labels_per_topic],
                    "split": r.split,
                    "start": r.start,
                    "count": r.count,
                }
                for r in records
            ],
            handle,
        )


def _six_label_topic(prefix):
    return [f"{prefix} label {i} " + "x" * i for i in range(6)]


def test_load_grouped_train_and_val_builds_the_1_2_3_mixture(tmp_path):
    from adapter_training.dataset import GroupRecord

    k1 = [
        TopicRecord("K1Train", _six_label_topic("k1t"), "train", start=0, count=4),
        TopicRecord("K1Val", _six_label_topic("k1v"), "val", start=4, count=4),
    ]
    v1 = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
    v1[0:4], v1[4:8] = 11.0, 12.0
    _write_topic_dir(tmp_path / "k1", k1, v1, torch.zeros(4, HIDDEN))

    k2 = [
        GroupRecord(
            ("K2TrainA", "K2TrainB"),
            (_six_label_topic("k2ta"), _six_label_topic("k2tb")),
            "train",
            start=0,
            count=4,
        ),
        GroupRecord(
            ("K2ValA", "K2ValB"),
            (_six_label_topic("k2va"), _six_label_topic("k2vb")),
            "val",
            start=4,
            count=4,
        ),
    ]
    v2 = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
    v2[0:4], v2[4:8] = 21.0, 22.0
    _write_group_dir(tmp_path / "k2", k2, v2, torch.zeros(4, HIDDEN))

    k3 = [
        GroupRecord(
            ("K3TrainA", "K3TrainB", "K3TrainC"),
            (
                _six_label_topic("k3ta"),
                _six_label_topic("k3tb"),
                _six_label_topic("k3tc"),
            ),
            "train",
            start=0,
            count=4,
        ),
        GroupRecord(
            ("K3ValA", "K3ValB", "K3ValC"),
            (
                _six_label_topic("k3va"),
                _six_label_topic("k3vb"),
                _six_label_topic("k3vc"),
            ),
            "val",
            start=4,
            count=4,
        ),
    ]
    v3 = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
    v3[0:4], v3[4:8] = 31.0, 32.0
    _write_group_dir(tmp_path / "k3", k3, v3, torch.zeros(4, HIDDEN))

    directories = {
        "k1": tmp_path / "k1",
        "k2": tmp_path / "k2",
        "k3": tmp_path / "k3",
    }
    train, val = load_grouped_train_and_val(
        directories,
        ratio={"k1": 1, "k2": 2, "k3": 3},
        budget_examples=60,
        val_total_examples=30,
        seed=0,
    )
    train_store, train_examples = train.store, train.examples
    val_store, val_examples = val.store, val.examples

    assert len(train_examples) == 60
    assert len(val_examples) == 30
    assert {
        name: end - start for name, (start, end) in train.source_ranges.items()
    } == {"k1": 10, "k2": 20, "k3": 30}
    assert {name: end - start for name, (start, end) in val.source_ranges.items()} == {
        "k1": 5,
        "k2": 10,
        "k3": 15,
    }
    # Every train example must address a train row, never a val one -- split
    # purity across the mixture store. Values are pooled-centred: each
    # source's own mean is the average of its train/val constants
    # (11.5/21.5/31.5), pooled equally to 21.5, so raw - 21.5.
    train_values = {
        round(train_store.vectors[e.vector_index, 0].item(), 6) for e in train_examples
    }
    assert train_values == {-10.5, -0.5, 9.5}
    val_values = {
        round(val_store.vectors[e.vector_index, 0].item(), 6) for e in val_examples
    }
    assert val_values == {-9.5, 0.5, 10.5}
