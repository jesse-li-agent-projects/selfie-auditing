"""On-disk shape of a sweep's results: a JSON metadata sidecar, one JSONL line per cell.

A full sweep is hundreds of MB of raw generations (plan S4.6 keeps every one),
which a single nested JSON document forces the writer, the merge and every
reader to hold in memory whole. One self-describing line per cell keeps all
three streaming, and lets a shard append each cell as it finishes -- so a run
that dies keeps everything it had already produced.

Cells are written in the sweep's own deterministic order, which is what lets
the merge walk several shards in step instead of indexing them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Iterable, Iterator

# The fields that identify a cell; the rest of a line is its payload.
KEY_FIELDS = ("arm", "word", "layer", "position")


def metadata_path(cells_path: Path) -> Path:
    """The metadata sidecar belonging to a cells file."""
    return cells_path.with_suffix(".json")


def shard_cells_path(output_dir: Path, sample_start: int, sample_end: int) -> Path:
    """This shard's cells file, named by its sample range.

    Shards of one sweep share an output directory, so the range in the name is
    what keeps them from colliding.
    """
    return output_dir / f"results_{sample_start:06d}_{sample_end:06d}.jsonl"


def shard_cells_paths(results_dir: Path) -> list[Path]:
    """Every shard's cells file in `results_dir`, unsorted.

    Matches only the range-suffixed shard files, never a merged `results.jsonl`
    written back into the same directory.
    """
    return list(results_dir.glob("results_*_*.jsonl"))


def cell_key(cell: dict) -> tuple:
    """The (arm, word, layer, position) identity of a cell record."""
    return tuple(cell[field] for field in KEY_FIELDS)


def write_metadata(cells_path: Path, metadata: dict) -> None:
    """Write the sidecar for `cells_path`.

    Written before the first cell, so an interrupted shard still says what it
    was doing.
    """
    metadata_path(cells_path).write_text(json.dumps(metadata, indent=2))


def read_metadata(cells_path: Path) -> dict:
    """Read the sidecar belonging to `cells_path`."""
    return json.loads(metadata_path(cells_path).read_text())


def append_cell(handle: IO[str], cell: dict) -> None:
    """Append one cell record and flush it.

    Flushing per cell is what makes a killed run's output usable: a cell costs
    many seconds of generation, so the write is free by comparison.
    """
    handle.write(json.dumps(cell) + "\n")
    handle.flush()


def read_cells(cells_path: Path) -> Iterator[dict]:
    """Stream a cells file, one record at a time."""
    with open(cells_path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_cells(cells_path: Path, cells: Iterable[dict]) -> int:
    """Write a whole cells file from an iterable, streaming.

    :param cells_path: destination cells file
    :param cells: cell records, already in the sweep's key order
    :return: how many cells were written
    """
    written = 0
    with open(cells_path, "w") as handle:
        for cell in cells:
            append_cell(handle, cell)
            written += 1
    return written
