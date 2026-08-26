import json

from results_store import (
    append_cell,
    metadata_path,
    read_cells,
    read_metadata,
    shard_cells_path,
    shard_cells_paths,
    write_cells,
    write_metadata,
)

CELLS = [
    {"arm": "control", "word": "gold", "layer": 0, "position": "pos-1", "hits": 1},
    {"arm": "control", "word": "gold", "layer": 1, "position": "pos-1", "hits": 0},
]


def test_shard_paths_are_named_by_sample_range(tmp_path):
    path = shard_cells_path(tmp_path, 0, 100)

    assert path.name == "results_000000_000100.jsonl"
    assert metadata_path(path).name == "results_000000_000100.json"


def test_cells_round_trip(tmp_path):
    path = shard_cells_path(tmp_path, 0, 100)

    write_cells(path, CELLS)

    assert list(read_cells(path)) == CELLS


def test_a_partly_written_shard_still_reads(tmp_path):
    # The point of appending per cell: a run killed mid-sweep keeps every cell
    # it had already paid for.
    path = shard_cells_path(tmp_path, 0, 100)
    with open(path, "w") as handle:
        append_cell(handle, CELLS[0])

    assert list(read_cells(path)) == CELLS[:1]


def test_metadata_round_trips_beside_its_cells(tmp_path):
    path = shard_cells_path(tmp_path, 0, 100)
    metadata = {"sample_range": [0, 100], "batch_size": 25}

    write_metadata(path, metadata)

    assert read_metadata(path) == metadata
    assert json.loads(metadata_path(path).read_text()) == metadata


def test_shard_glob_ignores_a_merged_file(tmp_path):
    write_cells(shard_cells_path(tmp_path, 0, 100), CELLS)
    write_cells(tmp_path / "results.jsonl", CELLS)

    assert [p.name for p in shard_cells_paths(tmp_path)] == [
        "results_000000_000100.jsonl"
    ]
