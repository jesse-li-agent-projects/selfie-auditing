"""Tests for adapter_training.grouped_examples."""

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
    assert _split_by_ratio(1_510_782, {"k1": 1, "k2": 2, "k3": 3}) == {
        "k1": 251_797,
        "k2": 503_594,
        "k3": 755_391,
    }


def test_split_by_ratio_sums_exactly_with_a_remainder():
    counts = _split_by_ratio(10, {"k1": 1, "k2": 2, "k3": 3})
    assert sum(counts.values()) == 10


def test_split_by_ratio_is_never_negative():
    counts = _split_by_ratio(1, {"k1": 1, "k2": 2, "k3": 3})
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
    """k=1, k=2, k=3 directories, each one train group and one val group, a
    distinguishing constant raw value per k, per split -- lets a test read
    off which directory/split a vector_index came from.

    The stored `position_means.pt` files are all zero, but that no longer
    makes centring a no-op (D13, amended 2026-09-07): the pooled reference is
    recomputed from these vectors and weighted equally per k, not read from
    disk. Each k's own mean is the average of its train and val constants
    (11.5, 21.5, 31.5), pooled equally to 21.5, so a `raw` constant centres to
    `raw - 21.5` -- see `EXPECTED_CENTRED` below.
    """
    directories = {}

    k1_train = TopicRecord("K1Train", six_label_topic("k1t"), "train", start=0, count=4)
    k1_val = TopicRecord("K1Val", six_label_topic("k1v"), "val", start=4, count=4)
    vectors1 = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
    vectors1[0:4] = 11.0
    vectors1[4:8] = 12.0
    write_topic_dir(
        tmp_path / "k1", [k1_train, k1_val], vectors1, torch.zeros(4, HIDDEN)
    )
    directories["k1"] = tmp_path / "k1"

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
    directories["k2"] = tmp_path / "k2"

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
    directories["k3"] = tmp_path / "k3"

    return directories


EVEN = {"k1": 1, "k2": 1, "k3": 1}


def test_build_mixture_splits_by_ratio(tmp_path):
    directories = build_fixture(tmp_path)
    mixture = build_mixture(directories, "train", 12, ratio=EVEN, seed=0)
    assert len(mixture.examples) == 12
    assert {
        name: end - start for name, (start, end) in mixture.source_ranges.items()
    } == {"k1": 4, "k2": 4, "k3": 4}
    for start, end in mixture.source_ranges.values():
        assert mixture.examples[start:end] != []


def test_build_mixture_rejects_a_ratio_that_names_other_sources(tmp_path):
    directories = build_fixture(tmp_path)
    with pytest.raises(ValueError, match="ratio names"):
        build_mixture(directories, "train", 12, ratio={"k1": 1, "nope": 1}, seed=0)


# raw -> pooled-centred (raw - 21.5), per build_fixture's docstring.
TRAIN_CENTRED = {"k1": -10.5, "k2": -0.5, "k3": 9.5}
VAL_CENTRED = {"k1": -9.5, "k2": 0.5, "k3": 10.5}


def test_build_mixture_offsets_index_into_the_mixture_store(tmp_path):
    directories = build_fixture(tmp_path)
    mixture = build_mixture(directories, "train", 12, ratio=EVEN, seed=0)
    # Every example's vector_index must land on that source's own
    # (pooled-centred) rows, not another source's or the val split's.
    for name, (start, end) in mixture.source_ranges.items():
        for example in mixture.examples[start:end]:
            assert mixture.store.vectors[
                example.vector_index, 0
            ].item() == pytest.approx(TRAIN_CENTRED[name])


def test_build_mixture_examples_stay_inside_their_own_source_rows(tmp_path):
    # The check that replaces counting "; " separators: that one cannot tell
    # two single-topic sources apart, since both compose to zero separators.
    directories = build_fixture(tmp_path)
    mixture = build_mixture(directories, "train", 12, ratio=EVEN, seed=0)
    for name, (start, end) in mixture.source_ranges.items():
        row_start, row_end = mixture.source_rows[name]
        for example in mixture.examples[start:end]:
            assert row_start <= example.vector_index < row_end


def test_build_mixture_never_addresses_a_val_group_from_a_train_call(tmp_path):
    directories = build_fixture(tmp_path)
    mixture = build_mixture(directories, "train", 12, ratio=EVEN, seed=0)
    values = {
        round(mixture.store.vectors[e.vector_index, 0].item(), 6)
        for e in mixture.examples
    }
    assert values == set(TRAIN_CENTRED.values())


def test_build_mixture_val_call_only_addresses_val_groups(tmp_path):
    directories = build_fixture(tmp_path)
    mixture = build_mixture(directories, "val", 12, ratio=EVEN, seed=0)
    values = {
        round(mixture.store.vectors[e.vector_index, 0].item(), 6)
        for e in mixture.examples
    }
    assert values == set(VAL_CENTRED.values())


# --- two sources at the same k --------------------------------------------


def build_two_single_topic_sources(tmp_path):
    """Two single-topic (k=1) sources, the case the k-keyed code could not
    express at all. Raw constants 11/12 and 41/42; each source's own mean is
    11.5 and 41.5, pooled equally to 26.5.
    """
    directories = {}
    for name, base in (("tell", 11.0), ("bg1", 41.0)):
        train = TopicRecord(
            f"{name}Train", six_label_topic(f"{name}t"), "train", start=0, count=4
        )
        val = TopicRecord(
            f"{name}Val", six_label_topic(f"{name}v"), "val", start=4, count=4
        )
        vectors = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
        vectors[0:4] = base
        vectors[4:8] = base + 1.0
        write_topic_dir(tmp_path / name, [train, val], vectors, torch.zeros(4, HIDDEN))
        directories[name] = tmp_path / name
    return directories


def test_two_sources_at_the_same_k_land_in_disjoint_offset_ranges(tmp_path):
    directories = build_two_single_topic_sources(tmp_path)
    mixture = build_mixture(
        directories, "train", 8, ratio={"tell": 1, "bg1": 1}, seed=0
    )
    assert mixture.source_rows == {"tell": (0, 8), "bg1": (8, 16)}
    for name, (start, end) in mixture.source_ranges.items():
        row_start, row_end = mixture.source_rows[name]
        for example in mixture.examples[start:end]:
            assert row_start <= example.vector_index < row_end


def test_two_sources_at_the_same_k_are_centred_on_the_pooled_mean(tmp_path):
    directories = build_two_single_topic_sources(tmp_path)
    mixture = build_mixture(
        directories, "train", 8, ratio={"tell": 1, "bg1": 1}, seed=0
    )
    centred = {
        name: mixture.store.vectors[mixture.examples[start].vector_index, 0].item()
        for name, (start, _end) in mixture.source_ranges.items()
    }
    assert centred["tell"] == pytest.approx(11.0 - 26.5)
    assert centred["bg1"] == pytest.approx(41.0 - 26.5)


def test_mixture_ratio_follows_the_given_order_not_a_sorted_one(tmp_path):
    # bg1 is given second and takes weight 3, so a sorted-by-name reading
    # (bg1 before tell) would hand it 1 instead.
    directories = build_two_single_topic_sources(tmp_path)
    mixture = build_mixture(
        directories, "train", 8, ratio={"tell": 1, "bg1": 3}, seed=0
    )
    assert {
        name: end - start for name, (start, end) in mixture.source_ranges.items()
    } == {"tell": 2, "bg1": 6}


def test_single_topic_source_reports_its_semicolon_drops(tmp_path):
    directories = build_two_single_topic_sources(tmp_path)
    dropped = TopicRecord("Dropped", ("a; b", "c; d"), "train", start=8, count=4)
    kept = TopicRecord("Kept", six_label_topic("kept"), "train", start=0, count=4)
    vectors = torch.zeros(12, HIDDEN, dtype=torch.bfloat16)
    vectors[0:4] = 51.0
    vectors[8:12] = 52.0
    write_topic_dir(
        tmp_path / "filtered", [kept, dropped], vectors, torch.zeros(4, HIDDEN)
    )
    directories = {"filtered": tmp_path / "filtered", "tell": directories["tell"]}
    mixture = build_mixture(
        directories, "train", 4, ratio={"filtered": 1, "tell": 1}, seed=0
    )
    assert mixture.semicolon_drops == {"filtered": 1, "tell": 0}
    # And the dropped topic's rows are never addressed.
    start, end = mixture.source_ranges["filtered"]
    for example in mixture.examples[start:end]:
        assert example.vector_index < 4
