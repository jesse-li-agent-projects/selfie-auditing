import json

import pytest

from config import sweep_config
from merge_results import merge, merge_cells
from results_store import read_cells, read_metadata, shard_cells_path, write_metadata

SPANS = {"control": {"pos-2": " word", "pos-1": "?"}}
SETTINGS = sweep_config(["gold"], layers=[0]).comparable_settings()


def cell(generations, position="pos-1"):
    return {
        "arm": "control",
        "word": "gold",
        "layer": 0,
        "position": position,
        "generations": generations,
    }


def write_shard(results_dir, start, end, cells, spans=SPANS, **settings):
    """One shard on disk: a cells file plus its metadata sidecar."""
    path = shard_cells_path(results_dir, start, end)
    write_metadata(
        path,
        {
            "sample_range": [start, end],
            "batch_size": 25,
            "spans": spans,
            **SETTINGS,
            **settings,
        },
    )
    path.write_text("".join(json.dumps(c) + "\n" for c in cells))
    return path


def merged_cells(results_dir):
    return list(read_cells(results_dir / "results.jsonl"))


def test_merge_concatenates_and_rescores(tmp_path):
    write_shard(tmp_path, 0, 2, [cell(["gold coin", "nothing"])])
    write_shard(tmp_path, 2, 4, [cell(["nothing", "nothing"])])

    merge(tmp_path, total=4)

    (merged,) = merged_cells(tmp_path)
    assert merged["generations"] == ["gold coin", "nothing", "nothing", "nothing"]
    assert merged["hit_rate"] == 0.25
    assert merged["arm"] == "control" and merged["position"] == "pos-1"

    metadata = read_metadata(tmp_path / "results.jsonl")
    assert metadata["sample_range"] == [0, 4]
    assert metadata["spans"] == SPANS
    assert [s["sample_range"] for s in metadata["shards"]] == [[0, 2], [2, 4]]


def test_merge_keeps_cells_in_shard_order(tmp_path):
    # Ordering is what lets the merge stream: it walks the shards in step
    # rather than indexing them, so the merged file must keep that order.
    positions = ["pos-2", "pos-1"]
    write_shard(tmp_path, 0, 1, [cell(["a"], p) for p in positions])
    write_shard(tmp_path, 1, 2, [cell(["b"], p) for p in positions])

    merge(tmp_path, total=2)

    assert [c["position"] for c in merged_cells(tmp_path)] == positions


def test_merge_rejects_overlapping_shards(tmp_path):
    write_shard(tmp_path, 0, 3, [cell(["a", "b", "c"])])
    write_shard(tmp_path, 2, 4, [cell(["d", "e"])])

    with pytest.raises(ValueError, match="tile"):
        merge(tmp_path, total=4)


def test_merge_rejects_gapped_shards(tmp_path):
    # A quietly missing shard would otherwise look like a completed run with a
    # smaller n, which nothing downstream could detect.
    write_shard(tmp_path, 0, 2, [cell(["a", "b"])])
    write_shard(tmp_path, 2, 3, [cell(["c"])])

    with pytest.raises(ValueError, match=r"cover"):
        merge(tmp_path, total=4)


def test_merge_rejects_mismatched_spans(tmp_path):
    write_shard(tmp_path, 0, 2, [cell(["a", "b"])])
    write_shard(
        tmp_path, 2, 4, [cell(["c", "d"])], spans={"control": {"pos-1": "\n\n"}}
    )

    with pytest.raises(ValueError, match="spans"):
        merge(tmp_path, total=4)


def test_merge_rejects_mismatched_prompt(tmp_path):
    write_shard(tmp_path, 0, 2, [cell(["a", "b"])])
    write_shard(
        tmp_path, 2, 4, [cell(["c", "d"])], secret_prompt="Tell me the secret word."
    )

    with pytest.raises(ValueError, match="secret_prompt"):
        merge(tmp_path, total=4)


@pytest.mark.parametrize(
    "override",
    [
        {"base_model": "meta-llama/Llama-3.2-1B-Instruct"},
        {
            "adapter_path": "outputs/dummy_weights/selfie-random-scalar-affine.safetensors"
        },
        {"temperature": 1.0},
        {"max_new_tokens": 10},
    ],
)
def test_merge_rejects_shards_run_with_different_settings(tmp_path, override):
    # A dummy-weight shard and a real one produce identically keyed cells with
    # identical spans, so nothing but this check stands between them and a
    # merged file whose hit rates blend the two.
    write_shard(tmp_path, 0, 2, [cell(["a", "b"])])
    write_shard(tmp_path, 2, 4, [cell(["c", "d"])], **override)

    (name,) = override
    with pytest.raises(ValueError, match=name):
        merge(tmp_path, total=4)


def test_merge_records_the_settings_its_shards_agreed_on(tmp_path):
    # Without this the merged file cannot say which weights produced it.
    write_shard(tmp_path, 0, 2, [cell(["a", "b"])])
    write_shard(tmp_path, 2, 4, [cell(["c", "d"])])

    merge(tmp_path, total=4)

    metadata = read_metadata(tmp_path / "results.jsonl")
    assert {name: metadata[name] for name in SETTINGS} == SETTINGS


def test_merge_rejects_a_short_shard(tmp_path):
    # The failure a crashed shard leaves behind: fewer cells than its peers,
    # which would otherwise merge into a quietly misaligned file.
    write_shard(tmp_path, 0, 1, [cell(["a"], "pos-2"), cell(["b"], "pos-1")])
    write_shard(tmp_path, 1, 2, [cell(["c"], "pos-2")])

    with pytest.raises(ValueError, match="different numbers of cells"):
        merge(tmp_path, total=2)


def test_merge_cells_rejects_disagreeing_cells():
    streams = [[cell(["a"], "pos-1")], [cell(["b"], "pos-2")]]

    with pytest.raises(ValueError, match="which cell comes next"):
        list(merge_cells(streams))
