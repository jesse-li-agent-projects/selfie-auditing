"""Combine the per-shard results of one sharded sweep into results.jsonl.

    python merge_results.py --results-dir results/sweep/ --total 200

Sharding is by sample: each shard ran every cell for its own slice of the
sample range, so merging means concatenating each cell's generations in sample
order and rescoring. Every shard writes its cells in the same deterministic
order, so the merge walks them in step and holds one cell in memory at a time
rather than the whole sweep. Pure dict and file logic, no heavy imports.

The checks here exist to stop a broken merge from looking healthy. Missing or
overlapping shards would silently produce a "200-sample" cell holding some
other number; shards whose prompt or span metadata disagree are not measuring
the same token at all, so concatenating them would be meaningless rather than
merely short. Shards launched from different code are the case preflight.py
cannot see, which is why the comparability check stays here as well.
"""

import argparse
from contextlib import ExitStack, closing
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Iterator

from results_store import (
    cell_key,
    read_cells,
    read_metadata,
    shard_cells_paths,
    write_cells,
    write_metadata,
)

MERGED_NAME = "results.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir", required=True, type=str, help="Directory of results_*.jsonl"
    )
    parser.add_argument(
        "--total",
        required=True,
        type=int,
        help="Expected total samples per cell; the shards must cover [0, total)",
    )
    return parser.parse_args()


def load_shards(results_dir: Path) -> list[tuple[dict, Path]]:
    """Every shard in `results_dir` as (metadata, cells path), in sample order.

    :param results_dir: directory holding one sharded sweep's output
    :return: (metadata, cells path) pairs, sorted by sample_range
    :raises ValueError: if `results_dir` holds no shards
    """
    shards = [(read_metadata(path), path) for path in shard_cells_paths(results_dir)]
    if not shards:
        raise ValueError(f"no results_*.jsonl shards in {results_dir}")
    return sorted(shards, key=lambda shard: shard[0]["sample_range"])


def check_coverage(metadata: list[dict], total: int) -> None:
    """Assert the shards tile [0, total) exactly -- no gaps, no overlaps.

    :param metadata: shard metadata, sorted by sample_range
    :param total: expected total samples per cell
    :raises ValueError: if the shards' sample ranges leave a gap, overlap, or
        don't cover [0, total)
    """
    covered = 0
    for shard in metadata:
        start, end = shard["sample_range"]
        if start != covered:
            raise ValueError(
                f"shard sample ranges do not tile [0, {total}): expected the next "
                f"shard to start at {covered}, got {start}"
            )
        covered = end
    if covered != total:
        raise ValueError(
            f"shard sample ranges cover [0, {covered}), not [0, {total}) -- a shard "
            "is missing or --total is wrong"
        )


def check_comparable(metadata: list[dict]) -> None:
    """Assert every shard read the same prompt at the same tokens.

    :param metadata: shard metadata to compare
    :raises ValueError: if any shard's secret_prompt or spans differ from the first
    """
    first = metadata[0]
    for shard in metadata[1:]:
        for field in ("secret_prompt", "spans"):
            if shard[field] != first[field]:
                raise ValueError(
                    f"shards disagree on {field!r}: {first[field]!r} vs "
                    f"{shard[field]!r} -- they are not measuring the same thing"
                )


def merge_cells(streams: list[Iterable[dict]]) -> Iterator[dict]:
    """Concatenate each cell's generations across shards and rescore, streaming.

    Rescoring rather than concatenating the stored `hits`: `hit_rate` is a
    ratio, so it cannot be merged arithmetically without also trusting each
    shard's `n`, and rescoring costs nothing.

    :param streams: one iterable of cell records per shard, each in key order
    :yield: the merged cell records, in the same order
    :raises ValueError: if the shards' cells disagree in identity or number
    """
    from scoring import score_cell

    for group in zip_longest(*streams):
        if any(cell is None for cell in group):
            raise ValueError(
                "shards hold different numbers of cells -- one of them did not "
                "finish, or they were run with different configs"
            )
        keys = {cell_key(cell) for cell in group}
        if len(keys) != 1:
            raise ValueError(
                f"shards disagree on which cell comes next: {sorted(keys)} -- they "
                "were run with different configs"
            )
        generations = [g for cell in group for g in cell["generations"]]
        scored = score_cell(generations, group[0]["word"])
        yield dict(
            group[0],
            generations=scored.generations,
            hits=scored.hits,
            hit_rate=scored.hit_rate,
        )


def merge(results_dir: Path, total: int) -> Path:
    """Merge one directory of shards into results.jsonl and its metadata sidecar.

    :param results_dir: directory holding one sharded sweep's output
    :param total: expected total samples per cell; the shards must cover [0, total)
    :return: path to the merged cells file
    :raises ValueError: if the shards don't tile [0, total), disagree on prompt
        or span metadata, or hold different cells
    """
    shards = load_shards(results_dir)
    metadata = [shard[0] for shard in shards]
    check_coverage(metadata, total)
    check_comparable(metadata)

    merged_path = results_dir / MERGED_NAME
    write_metadata(
        merged_path,
        {
            "sample_range": [0, total],
            "secret_prompt": metadata[0]["secret_prompt"],
            "spans": metadata[0]["spans"],
            # Per shard, not merged into one value: batch size is a property of
            # how a shard was produced, and shards may differ in it.
            "shards": [
                {"sample_range": m["sample_range"], "batch_size": m.get("batch_size")}
                for m in metadata
            ],
        },
    )
    with ExitStack() as stack:
        streams = [stack.enter_context(closing(read_cells(path))) for _, path in shards]
        count = write_cells(merged_path, merge_cells(streams))
    print(f"Merged {len(shards)} shards, {count} cells -> {merged_path}")
    return merged_path


if __name__ == "__main__":
    args = parse_args()

    merge(Path(args.results_dir), args.total)
