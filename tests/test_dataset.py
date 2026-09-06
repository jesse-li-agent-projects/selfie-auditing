"""Tests for adapter_training.dataset."""

import json

import pytest
import torch

from adapter_training.dataset import (
    Example,
    TopicRecord,
    examples_from_records,
    load_examples,
    load_records,
    load_topic_records,
    load_vector_store,
    pooled_position_means,
    pooled_vector_store,
    position_counts,
    restrict_to_titles,
)

HIDDEN = 3


def write_extraction_dir(tmp_path, records, vectors, means):
    """A hand-built extraction output directory: `topics.json`, `vectors.pt`,
    `position_means.pt` -- exactly the subset of files `dataset.py` reads."""
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
    return tmp_path


def two_topic_dir(tmp_path):
    """topic0: 10 vectors (train); topic1: 9 vectors (val) -- mirrors the
    pangram style's two variant lengths."""
    records = [
        TopicRecord("Alpha", ("a label",), "train", start=0, count=10),
        TopicRecord("Bravo", ("b label",), "val", start=10, count=9),
    ]
    # Raw vector at position p of a topic is a constant `base` plus `p`, so a
    # correct per-position centering collapses every vector back to `base`.
    means = torch.stack([torch.full((HIDDEN,), float(p)) for p in range(10)])
    vectors = torch.zeros(19, HIDDEN, dtype=torch.bfloat16)
    for p in range(10):
        vectors[p] = 100.0 + p
    for p in range(9):
        vectors[10 + p] = 200.0 + p
    write_extraction_dir(tmp_path, records, vectors, means)
    return records


def test_centering_subtracts_each_vectors_own_position_mean(tmp_path):
    two_topic_dir(tmp_path)

    store = load_vector_store(tmp_path, center=True)

    assert torch.allclose(store.vectors[:10], torch.full((10, HIDDEN), 100.0))
    # The 9-vector topic's last vector is position 8, not 9: it must be
    # centred by means[8], not means[9] -- position 9 has no data for this
    # topic (count is genuinely per-topic).
    assert torch.allclose(store.vectors[10:19], torch.full((9, HIDDEN), 200.0))


def test_flat_means_file_is_rejected_rather_than_broadcast(tmp_path):
    """`[hidden]` means would slice to a scalar and silently no-op (PR #51)."""
    records = [TopicRecord("Alpha", ("a label",), "train", start=0, count=1)]
    write_extraction_dir(tmp_path, records, torch.zeros(1, HIDDEN), torch.zeros(HIDDEN))

    with pytest.raises(ValueError, match="position means"):
        load_vector_store(tmp_path, center=True)


def test_pooled_means_equal_the_mean_over_every_directorys_vectors(tmp_path):
    # Two directories whose populations differ by a constant offset -- the
    # thing per-directory centring removes and pooled centring keeps.
    dirs, sources = [], []
    for index, base in enumerate((100.0, 300.0)):
        directory = tmp_path / f"d{index}"
        directory.mkdir()
        records = [TopicRecord("Alpha", ("a",), "train", start=0, count=2)]
        # Offsets chosen to be exactly representable in bf16.
        means = torch.tensor([[base] * HIDDEN, [base + 4] * HIDDEN])
        write_extraction_dir(directory, records, means.to(torch.bfloat16), means)
        dirs.append(directory)
        sources.append((directory, records))

    pooled = pooled_position_means(sources)

    assert torch.allclose(pooled, torch.tensor([[200.0] * HIDDEN, [204.0] * HIDDEN]))
    # Centred against the pooled mean, each directory keeps its own offset.
    first = load_vector_store(dirs[0], records=sources[0][1], means=pooled)
    second = load_vector_store(dirs[1], records=sources[1][1], means=pooled)
    assert torch.allclose(first.vectors, torch.full((2, HIDDEN), -100.0))
    assert torch.allclose(second.vectors, torch.full((2, HIDDEN), 100.0))


def test_pooled_means_weight_positions_by_how_many_records_reached_them(tmp_path):
    # A 9-vector topic contributes nothing to position 9, so that position's
    # pooled mean must come from the 10-vector topic alone -- the same rule
    # the extractor used when it wrote each file.
    records = two_topic_dir(tmp_path)

    pooled = pooled_position_means([(tmp_path, records)])
    counts = position_counts(records, 10)

    assert counts.tolist() == [2.0] * 9 + [1.0]
    # One directory pooled with itself is that directory's own means.
    stored = torch.load(tmp_path / "position_means.pt", weights_only=True)
    assert torch.allclose(pooled, stored)


def test_no_center_returns_raw_vectors(tmp_path):
    two_topic_dir(tmp_path)

    store = load_vector_store(tmp_path, center=False)

    assert store.vectors[0, 0].item() == 100.0
    assert store.vectors[10 + 8, 0].item() == 208.0


def test_load_examples_flattens_every_vector_against_every_label(tmp_path):
    records = [
        TopicRecord("Alpha", ("l0", "l1"), "train", start=0, count=2),
        TopicRecord("Bravo", ("m0",), "val", start=2, count=3),
    ]
    write_extraction_dir(
        tmp_path, records, torch.zeros(5, HIDDEN), torch.zeros(3, HIDDEN)
    )

    train_examples = load_examples(tmp_path, "train")
    val_examples = load_examples(tmp_path, "val")

    assert train_examples == [
        Example(0, "l0"),
        Example(1, "l0"),
        Example(0, "l1"),
        Example(1, "l1"),
    ]
    assert val_examples == [Example(2, "m0"), Example(3, "m0"), Example(4, "m0")]


def test_pooled_vector_store_means_a_topics_own_vectors(tmp_path):
    two_topic_dir(tmp_path)
    records = [
        TopicRecord("Alpha", ("a0", "a1"), "train", start=0, count=10),
        TopicRecord("Bravo", ("b0",), "val", start=10, count=9),
    ]
    write_extraction_dir(
        tmp_path,
        records,
        torch.load(tmp_path / "vectors.pt", weights_only=True),
        torch.load(tmp_path / "position_means.pt", weights_only=True),
    )

    store, examples = pooled_vector_store(tmp_path)

    assert store.vectors.shape == (2, HIDDEN)
    assert torch.allclose(store.vectors[0], torch.full((HIDDEN,), 100.0))
    assert torch.allclose(store.vectors[1], torch.full((HIDDEN,), 200.0))
    assert examples == [
        Example(0, "a0"),
        Example(0, "a1"),
        Example(1, "b0"),
    ]


def test_pooled_vector_store_can_restrict_to_a_split(tmp_path):
    two_topic_dir(tmp_path)

    store, examples = pooled_vector_store(tmp_path, split="train")

    assert store.vectors.shape == (1, HIDDEN)
    assert examples == [Example(0, "a label")]


def test_restrict_to_titles_keeps_the_intersection_and_index_integrity(tmp_path):
    records = [
        TopicRecord("Alpha", ("a",), "train", start=0, count=2),
        TopicRecord("Bravo", ("b",), "train", start=2, count=1),
        TopicRecord("Charlie", ("c",), "train", start=3, count=1),
    ]

    restricted = restrict_to_titles(records, {"Alpha", "Charlie"})

    assert [r.title for r in restricted] == ["Alpha", "Charlie"]
    examples = examples_from_records(restricted, "train")
    assert examples == [Example(0, "a"), Example(1, "a"), Example(3, "c")]


def test_load_topic_records_round_trips_topics_json(tmp_path):
    records = [TopicRecord("Alpha", ("a",), "train", start=0, count=1)]
    write_extraction_dir(
        tmp_path, records, torch.zeros(1, HIDDEN), torch.zeros(1, HIDDEN)
    )

    assert load_topic_records(tmp_path) == records


def test_load_records_without_restrict_to_is_unfiltered(tmp_path):
    records = two_topic_dir(tmp_path)

    assert load_records(tmp_path) == records


def test_load_records_restrict_to_intersects_titles(tmp_path):
    two_topic_dir(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    write_extraction_dir(
        other_dir,
        [TopicRecord("Alpha", ("x",), "train", start=0, count=1)],
        torch.zeros(1, HIDDEN),
        torch.zeros(1, HIDDEN),
    )

    restricted = load_records(tmp_path, restrict_to=other_dir)

    assert [r.title for r in restricted] == ["Alpha"]
