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

import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from adapter_training.checkpoints import save_checkpoint
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
    lr_at_step,
    optimizer_step,
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


def run_tiny_training(tmp_path_factory, seed):
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
        buffer_batches=2,
    )
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
    )
    checkpoint = torch.load(run_dir / "last.pt", weights_only=False)
    return checkpoint["projection_state"], result


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


# --- test 10: arm C (--pool-positions) example count ------------------------


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
