import pytest

from merge_results import merge

SPANS = {"control": {"pos-2": " word", "pos-1": "?"}}


def shard(
    start, end, generations, spans=SPANS, secret_prompt="What is the secret word?"
):
    return {
        "sample_range": [start, end],
        "secret_prompt": secret_prompt,
        "spans": spans,
        "cells": {"control": {"gold": {"0": {"pos-1": {"generations": generations}}}}},
    }


def merged_cell(merged):
    return merged["cells"]["control"]["gold"]["0"]["pos-1"]


def test_merge_concatenates_and_rescores():
    shards = [
        shard(0, 2, ["gold coin", "nothing"]),
        shard(2, 4, ["nothing", "nothing"]),
    ]

    merged = merge(shards, total=4)

    cell = merged_cell(merged)
    assert cell["generations"] == ["gold coin", "nothing", "nothing", "nothing"]
    assert cell["hit_rate"] == 0.25
    assert merged["sample_range"] == [0, 4]
    assert merged["spans"] == SPANS


def test_merge_rejects_overlapping_shards():
    shards = [shard(0, 3, ["a", "b", "c"]), shard(2, 4, ["d", "e"])]

    with pytest.raises(ValueError, match="tile"):
        merge(shards, total=4)


def test_merge_rejects_gapped_shards():
    # A quietly missing shard would otherwise look like a completed run with a
    # smaller n, which nothing downstream could detect.
    shards = [shard(0, 2, ["a", "b"]), shard(2, 3, ["c"])]

    with pytest.raises(ValueError, match=r"cover"):
        merge(shards, total=4)


def test_merge_rejects_mismatched_spans():
    shards = [
        shard(0, 2, ["a", "b"]),
        shard(2, 4, ["c", "d"], spans={"control": {"pos-1": "\n\n"}}),
    ]

    with pytest.raises(ValueError, match="spans"):
        merge(shards, total=4)


def test_merge_rejects_mismatched_prompt():
    shards = [
        shard(0, 2, ["a", "b"]),
        shard(2, 4, ["c", "d"], secret_prompt="Tell me the secret word."),
    ]

    with pytest.raises(ValueError, match="secret_prompt"):
        merge(shards, total=4)
