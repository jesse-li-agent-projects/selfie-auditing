"""Tests for adapter_training.grouped_examples (bg_think_many step 3)."""

import json
from collections import Counter
from itertools import permutations

import pytest
import torch

from adapter_training.dataset import GroupRecord, TopicRecord
from adapter_training.grouped_examples import (
    _split_by_ratio,
    build_mixture,
    compose_label,
    group_record_from_topic,
    sample_examples,
)
from adapter_training.label_complexity import label_buckets

HIDDEN = 3


# --- compose_label -----------------------------------------------------


def test_compose_label_joins_with_semicolon_space():
    assert compose_label(["Alpha", "Bravo", "Charlie"]) == "Alpha; Bravo; Charlie"


def test_compose_label_k1_is_the_bare_label():
    # What keeps the k=1 slice's targets byte-identical to bg_think's.
    assert compose_label(["Alpha"]) == "Alpha"


# --- group_record_from_topic --------------------------------------------


def test_group_record_from_topic_preserves_fields():
    topic = TopicRecord(
        title="Alpha",
        labels=("a0", "a1"),
        split="train",
        start=5,
        count=3,
        variant="v",
    )
    group = group_record_from_topic(topic)
    assert group == GroupRecord(
        titles=("Alpha",),
        labels_per_topic=(("a0", "a1"),),
        split="train",
        start=5,
        count=3,
        variant="v",
    )


def test_group_record_from_topic_composed_labels_equal_original():
    topic = TopicRecord("Alpha", ("a0", "a1"), "train", start=0, count=1)
    group = group_record_from_topic(topic)
    for labels in group.labels_per_topic:
        for label in labels:
            assert compose_label([label]) == label


# --- sample_examples -----------------------------------------------------


def make_group(titles, labels_per_topic, split="train", start=0, count=4):
    return GroupRecord(
        titles=tuple(titles),
        labels_per_topic=tuple(tuple(labels) for labels in labels_per_topic),
        split=split,
        start=start,
        count=count,
    )


def six_label_topic(prefix):
    return [f"{prefix} label {i} " + "x" * i for i in range(6)]


def test_sample_examples_returns_exact_count():
    record = make_group(
        ["Alpha", "Bravo"], [six_label_topic("a"), six_label_topic("b")]
    )
    examples = sample_examples([record], 37, seed=0)
    assert len(examples) == 37


def test_sample_examples_is_deterministic():
    record = make_group(
        ["Alpha", "Bravo"], [six_label_topic("a"), six_label_topic("b")]
    )
    first = sample_examples([record], 20, seed="s")
    second = sample_examples([record], 20, seed="s")
    assert first == second


def test_sample_examples_different_seed_differs():
    record = make_group(
        ["Alpha", "Bravo"], [six_label_topic("a"), six_label_topic("b")]
    )
    first = sample_examples([record], 20, seed="s1")
    second = sample_examples([record], 20, seed="s2")
    assert first != second


def test_sample_examples_never_duplicates_vector_label_pairs():
    record = make_group(
        ["Alpha", "Bravo"], [six_label_topic("a"), six_label_topic("b")]
    )
    examples = sample_examples([record], 50, seed=1)
    keys = [(e.vector_index, e.label) for e in examples]
    assert len(keys) == len(set(keys))


def test_sample_examples_raises_when_pool_is_exhausted():
    # One position, two single-label topics: exactly two composed labels
    # exist ("a; b" and "b; a"), so a third draw cannot avoid repeating one.
    record = make_group(["Alpha", "Bravo"], [["a"], ["b"]], count=1)
    with pytest.raises(ValueError, match="could not draw"):
        sample_examples([record], 3, seed=0)


def test_sample_examples_empty_bucket_falls_back_and_is_counted():
    # Rule A never produces an empty bucket on its own (n>=3 labels always
    # split into 3 non-empty groups), so this drives the fallback with a
    # deliberately bucket-hostile `buckets_of`: bucket 1 is always empty.
    def buckets_of(labels):
        return [list(labels[:1]), [], list(labels[1:])]

    record = make_group(["Alpha"], [["a0", "a1"]], count=5)
    stats = {}
    examples = sample_examples([record], 10, buckets_of=buckets_of, seed=0, stats=stats)
    assert len(examples) == 10
    assert stats["empty_bucket_fallbacks"] > 0
    assert all(e.label in ("a0", "a1") for e in examples)


def test_sample_examples_vector_index_uses_start_plus_position():
    record = make_group(["Alpha"], [six_label_topic("a")], start=100, count=3)
    examples = sample_examples([record], 15, seed=0)
    assert all(100 <= e.vector_index < 103 for e in examples)


def test_sample_examples_permutation_covers_all_orders_roughly_uniformly():
    # Single-label topics per position, so every draw's *order* is the only
    # thing varying; the "duplicate" guard would otherwise starve this after
    # 6 draws, so give it 200 positions' worth of room.
    labels = ["a", "b", "c"]
    record = make_group(["Alpha", "Bravo", "Charlie"], [[l] for l in labels], count=200)
    examples = sample_examples([record], 600, seed=7)
    counts = Counter(e.label for e in examples)
    expected_orders = {"; ".join(p) for p in permutations(labels)}
    assert set(counts) == expected_orders
    for count in counts.values():
        assert 60 < count < 140  # loose: ~100 expected over 600 draws / 6 orders


def test_sample_examples_composed_label_parts_share_a_bucket_index():
    topic_a = six_label_topic("a")
    topic_b = six_label_topic("b")
    record = make_group(["Alpha", "Bravo"], [topic_a, topic_b], count=10)
    buckets_a = label_buckets(topic_a)
    buckets_b = label_buckets(topic_b)

    def bucket_index_of(label, buckets):
        for i, bucket in enumerate(buckets):
            if label in bucket:
                return i
        raise AssertionError(label)

    examples = sample_examples([record], 60, seed=3)
    for example in examples:
        parts = example.label.split("; ")
        # One part is topic_a's label, the other topic_b's; whichever order
        # they came out in, their bucket indices must match.
        a_part, b_part = (
            (parts[0], parts[1]) if parts[0] in topic_a else (parts[1], parts[0])
        )
        assert bucket_index_of(a_part, buckets_a) == bucket_index_of(b_part, buckets_b)


# --- _split_by_ratio -------------------------------------------------------


def test_split_by_ratio_matches_the_dataset_budget():
    assert _split_by_ratio(1_510_782, {1: 1, 2: 2, 3: 3}) == {
        1: 251_797,
        2: 503_594,
        3: 755_391,
    }


def test_split_by_ratio_sums_exactly_with_a_remainder():
    counts = _split_by_ratio(10, {1: 1, 2: 2, 3: 3})
    assert sum(counts.values()) == 10


def test_split_by_ratio_is_never_negative():
    counts = _split_by_ratio(1, {1: 1, 2: 2, 3: 3})
    assert sum(counts.values()) == 1
    assert all(v >= 0 for v in counts.values())


# --- build_mixture ---------------------------------------------------------


def write_topic_dir(directory, records, vectors, means):
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


def write_group_dir(directory, records, vectors, means):
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


def build_fixture(tmp_path):
    """k=1, k=2, k=3 directories, each one train group and one val group,
    with position means all zero (so raw == centred) and a distinguishing
    constant value per k, per split -- lets a test read off which
    directory/split a vector_index came from."""
    directories = {}

    k1_train = TopicRecord("K1Train", six_label_topic("k1t"), "train", start=0, count=4)
    k1_val = TopicRecord("K1Val", six_label_topic("k1v"), "val", start=4, count=4)
    vectors1 = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
    vectors1[0:4] = 11.0
    vectors1[4:8] = 12.0
    write_topic_dir(
        tmp_path / "k1", [k1_train, k1_val], vectors1, torch.zeros(4, HIDDEN)
    )
    directories[1] = tmp_path / "k1"

    k2_train = GroupRecord(
        ("K2TrainA", "K2TrainB"),
        (six_label_topic("k2ta"), six_label_topic("k2tb")),
        "train",
        start=0,
        count=4,
    )
    k2_val = GroupRecord(
        ("K2ValA", "K2ValB"),
        (six_label_topic("k2va"), six_label_topic("k2vb")),
        "val",
        start=4,
        count=4,
    )
    vectors2 = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
    vectors2[0:4] = 21.0
    vectors2[4:8] = 22.0
    write_group_dir(
        tmp_path / "k2", [k2_train, k2_val], vectors2, torch.zeros(4, HIDDEN)
    )
    directories[2] = tmp_path / "k2"

    k3_train = GroupRecord(
        ("K3TrainA", "K3TrainB", "K3TrainC"),
        (six_label_topic("k3ta"), six_label_topic("k3tb"), six_label_topic("k3tc")),
        "train",
        start=0,
        count=4,
    )
    k3_val = GroupRecord(
        ("K3ValA", "K3ValB", "K3ValC"),
        (six_label_topic("k3va"), six_label_topic("k3vb"), six_label_topic("k3vc")),
        "val",
        start=4,
        count=4,
    )
    vectors3 = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
    vectors3[0:4] = 31.0
    vectors3[4:8] = 32.0
    write_group_dir(
        tmp_path / "k3", [k3_train, k3_val], vectors3, torch.zeros(4, HIDDEN)
    )
    directories[3] = tmp_path / "k3"

    return directories


def test_build_mixture_splits_by_ratio(tmp_path):
    directories = build_fixture(tmp_path)
    _, examples, k_ranges = build_mixture(
        directories, "train", 12, ratio={1: 1, 2: 1, 3: 1}, seed=0
    )
    assert len(examples) == 12
    assert {k: end - start for k, (start, end) in k_ranges.items()} == {
        1: 4,
        2: 4,
        3: 4,
    }
    for k, (start, end) in k_ranges.items():
        assert examples[start:end] != []


def test_build_mixture_offsets_index_into_the_concatenated_store(tmp_path):
    directories = build_fixture(tmp_path)
    store, examples, k_ranges = build_mixture(
        directories, "train", 12, ratio={1: 1, 2: 1, 3: 1}, seed=0
    )
    # k=1's train rows are 11.0, k=2's are 21.0, k=3's are 31.0 (means are
    # zero, so centring is a no-op); every example's vector_index must land
    # on that k's rows in the concatenated store.
    expected_value = {1: 11.0, 2: 21.0, 3: 31.0}
    for k, (start, end) in k_ranges.items():
        for example in examples[start:end]:
            assert store.vectors[example.vector_index, 0].item() == expected_value[k]


def test_build_mixture_never_addresses_a_val_group_from_a_train_call(tmp_path):
    directories = build_fixture(tmp_path)
    store, examples, _ = build_mixture(
        directories, "train", 12, ratio={1: 1, 2: 1, 3: 1}, seed=0
    )
    # Val rows are 12.0/22.0/32.0; none should appear among train's examples.
    values = {store.vectors[e.vector_index, 0].item() for e in examples}
    assert values == {11.0, 21.0, 31.0}


def test_build_mixture_val_call_only_addresses_val_groups(tmp_path):
    directories = build_fixture(tmp_path)
    store, examples, _ = build_mixture(
        directories, "val", 12, ratio={1: 1, 2: 1, 3: 1}, seed=0
    )
    values = {store.vectors[e.vector_index, 0].item() for e in examples}
    assert values == {12.0, 22.0, 32.0}
