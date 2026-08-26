"""Combine the per-shard results files of one sharded sweep into results.json.

    python merge_results.py --results-dir results/sweep/ --total 200

Sharding is by sample: each shard ran every cell for its own slice of the
sample range, so merging means concatenating each cell's generations in
sample order and rescoring. Pure dict logic, no heavy imports.

Both checks here exist to stop a broken merge from looking healthy. Missing or
overlapping shards would silently produce a "200-sample" cell holding some
other number, and shards whose prompt or span metadata disagree are not
measuring the same token at all, so concatenating them would be meaningless
rather than merely short.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir", required=True, type=str, help="Directory of results_*.json"
    )
    parser.add_argument(
        "--total",
        required=True,
        type=int,
        help="Expected total samples per cell; the shards must cover [0, total)",
    )
    return parser.parse_args()


def load_shards(results_dir: Path) -> list[dict]:
    """Every results_*.json in `results_dir`, in sample order.

    :param results_dir: directory holding one sharded sweep's results_*.json files
    :return: parsed shard documents, sorted by sample_range
    :raises ValueError: if `results_dir` has no results_*.json files
    """
    shards = [
        json.loads(path.read_text()) for path in results_dir.glob("results_*.json")
    ]
    if not shards:
        raise ValueError(f"no results_*.json files in {results_dir}")
    return sorted(shards, key=lambda shard: shard["sample_range"])


def check_coverage(shards: list[dict], total: int) -> None:
    """Assert the shards tile [0, total) exactly -- no gaps, no overlaps.

    :param shards: shard documents, sorted by sample_range
    :param total: expected total samples per cell
    :raises ValueError: if the shards' sample ranges leave a gap, overlap, or
        don't cover [0, total)
    """
    covered = 0
    for shard in shards:
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


def check_comparable(shards: list[dict]) -> None:
    """Assert every shard read the same prompt at the same tokens.

    :param shards: shard documents to compare
    :raises ValueError: if any shard's secret_prompt or spans differ from the first
    """
    first = shards[0]
    for shard in shards[1:]:
        for field in ("secret_prompt", "spans"):
            if shard[field] != first[field]:
                raise ValueError(
                    f"shards disagree on {field!r}: {first[field]!r} vs "
                    f"{shard[field]!r} -- they are not measuring the same thing"
                )


def merge_cells(shards: list[dict]) -> dict:
    """Concatenate every cell's generations across shards and rescore.

    Rescoring rather than concatenating the stored `hits`: `hit_rate` is a
    ratio, so it cannot be merged arithmetically without also trusting each
    shard's `n`, and rescoring costs nothing.

    :param shards: shard documents to merge; assumed already checked comparable
    :return: the merged arm -> word -> layer -> position cells
    """
    from scoring import score_cell

    merged: dict = {}
    for shard in shards:
        for arm, words in shard["cells"].items():
            for word, layers in words.items():
                for layer, positions in layers.items():
                    for position, cell in positions.items():
                        target = (
                            merged.setdefault(arm, {})
                            .setdefault(word, {})
                            .setdefault(layer, {})
                            .setdefault(position, [])
                        )
                        target.extend(cell["generations"])

    for arm, words in merged.items():
        for word, layers in words.items():
            for layer, positions in layers.items():
                for position, generations in positions.items():
                    scored = score_cell(generations, word)
                    positions[position] = {
                        "generations": scored.generations,
                        "hits": scored.hits,
                        "hit_rate": scored.hit_rate,
                    }
    return merged


def merge(shards: list[dict], total: int) -> dict:
    """One results document from many shards' worth of the same cells.

    :param shards: shard documents to merge
    :param total: expected total samples per cell; the shards must cover [0, total)
    :return: a merged results document with the same shape as one shard
    :raises ValueError: if the shards don't tile [0, total) or disagree on
        prompt/span metadata
    """
    check_coverage(shards, total)
    check_comparable(shards)
    return {
        "sample_range": [0, total],
        "secret_prompt": shards[0]["secret_prompt"],
        "spans": shards[0]["spans"],
        "cells": merge_cells(shards),
    }


if __name__ == "__main__":
    args = parse_args()

    results_dir = Path(args.results_dir)
    merged = merge(load_shards(results_dir), args.total)
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(
        f"Merged {len(list(results_dir.glob('results_*.json')))} shards -> {results_path}"
    )
