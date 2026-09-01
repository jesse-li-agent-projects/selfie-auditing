"""Score and plot one or more `run_bridge_entity.py` sweeps (paper S3.6).

    python report_bridge_entity.py --run baseline=bridge_entity/baseline \
        --run armB=bridge_entity/armB

The paper's headline number is per *question*, not per generation: a bridge
entity counts as detected if any generation, at any layer and any token,
names it. This writes that rate with a 95% interval, alongside the much
smaller fraction of individual generations that hit, and a layer x token
heat map per run on one shared colour scale.

Heat maps are aligned as the paper's Figure 3 is: a question's token
positions are shifted so that position 0 is the first token where any run
detects the entity at all. Bridge entities only become inferable after a
particular word ("Plato" is not recoverable before "Republic"), and that word
sits at a different offset in every question, so aggregating on raw offsets
would smear the signal across positions. Questions no run ever detects have
no such token and are left out of the maps -- they are still counted in the
detection rate, which is the number the maps illustrate rather than replace.
"""

import argparse
import json
from pathlib import Path

# Light import: config.py pulls in no heavy dependencies, so --help stays fast.
from config import outputs_relative


def parse_run(spec: str) -> tuple[str, Path]:
    """Parse `--run`: a `name=output_dir` pair, the directory under outputs/."""
    name, separator, directory = spec.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(
            f"{spec!r} is not a 'name=output_dir' pair, e.g. 'armB=bridge_entity/armB'"
        )
    return name, outputs_relative(directory)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run",
        type=parse_run,
        action="append",
        required=True,
        dest="runs",
        help="'name=output_dir' of a run_bridge_entity.py sweep; repeatable",
    )
    parser.add_argument(
        "--report",
        type=outputs_relative,
        default="bridge_entity/report.json",
        help="where the summary goes, under outputs/ (implicitly prepended)",
    )
    parser.add_argument("--plot-dir", type=Path, default=Path("plot"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from results_store import CELLS_FILE, read_cells  # noqa: E402


def load_cells(run_dir: Path) -> pd.DataFrame:
    """Load one sweep's cells into a flat frame.

    :param run_dir: a `run_bridge_entity.py` output directory
    :return: one row per (question, layer, position) cell, with the position's
        end-relative offset as an integer
    """
    frame = pd.DataFrame(
        {key: value for key, value in cell.items() if key != "generations"}
        for cell in read_cells(run_dir / CELLS_FILE)
    )
    if frame.empty:
        raise ValueError(f"{run_dir / CELLS_FILE} holds no cells")
    frame["offset"] = frame["position"].str.removeprefix("pos").astype(int)
    return frame


def summarize(frame: pd.DataFrame) -> dict:
    """Detection rate over questions, and hit rate over generations.

    The interval is the normal approximation the paper quotes its own
    91.0%+-1.3 with.

    :param frame: one run's cells
    :return: the run's headline numbers
    """
    detected = frame.groupby("question_id")["hit_rate"].max() > 0
    n_questions = int(detected.size)
    rate = float(detected.mean())
    half_width = 1.96 * (rate * (1 - rate) / n_questions) ** 0.5
    hits = frame["hits"].explode()
    detected_hits = frame[frame["question_id"].isin(detected[detected].index)]["hits"]
    return {
        "n_questions": n_questions,
        "n_cells": int(len(frame)),
        "detected": int(detected.sum()),
        "detection_rate": rate,
        "detection_rate_ci95": half_width,
        "generation_hit_rate": float(hits.mean()),
        # None, not NaN: a run that detected nothing has no such rate, and
        # json.dumps writes bare NaN, which strict JSON readers reject.
        "generation_hit_rate_detected_only": (
            float(detected_hits.explode().mean()) if len(detected_hits) else None
        ),
        "detected_question_ids": sorted(detected[detected].index),
    }


def alignment_offsets(frames: dict[str, pd.DataFrame]) -> pd.Series:
    """Per question, the token offset the heat maps put at position 0.

    That is the earliest token, at any layer in any run, where the entity is
    detected at all -- the shared crossing point the paper aligns on.

    :param frames: each run's cells, keyed by run name
    :return: the alignment offset per question id
    """
    pooled = pd.concat(frames.values())
    return pooled[pooled["hit_rate"] > 0].groupby("question_id")["offset"].min()


def detection_grid(frame: pd.DataFrame, offsets: pd.Series) -> pd.DataFrame:
    """Mean hit rate per (layer, aligned position), over questions.

    :param frame: one run's cells
    :param offsets: `alignment_offsets`' output
    :return: layers as rows, aligned positions as columns
    """
    aligned = frame[frame["question_id"].isin(offsets.index)].copy()
    aligned["aligned"] = aligned["offset"] - aligned["question_id"].map(offsets)
    return aligned.pivot_table(
        index="layer", columns="aligned", values="hit_rate", aggfunc="mean"
    )


def plot_detection_grids(grids: dict[str, pd.DataFrame], plot_dir: Path) -> Path:
    """One heat map per run, side by side on shared axes and a shared colour scale.

    Runs are reindexed onto the union of their layers and aligned positions
    first: without that each panel gets whatever range its own hits happened
    to span, and two panels that cannot be read off against each other defeat
    the point of drawing them together.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted({layer for grid in grids.values() for layer in grid.index})
    columns = sorted({column for grid in grids.values() for column in grid.columns})
    grids = {
        name: grid.reindex(index=layers, columns=columns)
        for name, grid in grids.items()
    }
    # Reindexing leaves NaN where no question reached that far back, which a
    # plain numpy max() would propagate into the colour scale; pandas skips it.
    vmax = max(float(grid.max().max()) for grid in grids.values())
    fig, axes = plt.subplots(
        1, len(grids), sharey=True, figsize=(6 * len(grids), 6), squeeze=False
    )
    image = None
    for ax, (name, grid) in zip(axes[0], grids.items()):
        image = ax.imshow(grid, vmin=0, vmax=vmax, aspect="auto", origin="lower")
        ax.set_xticks(range(len(grid.columns)), labels=grid.columns, rotation=90)
        ax.set_yticks(range(len(grid.index)), labels=grid.index)
        ax.set_xlabel("token position (0 = first detected)")
        ax.set_title(name)
    axes[0][0].set_ylabel("extracted layer")
    fig.colorbar(image, ax=axes[0], label="bridge-entity detection fraction")
    path = plot_dir / "bridge_entity_detection.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main(args) -> dict:
    frames = {name: load_cells(directory) for name, directory in args.runs}
    report = {name: summarize(frame) for name, frame in frames.items()}
    for name, summary in report.items():
        print(
            f"{name}: bridge entity detected in {summary['detected']}/"
            f"{summary['n_questions']} questions "
            f"({summary['detection_rate']:.1%} +- {summary['detection_rate_ci95']:.1%}), "
            f"{summary['generation_hit_rate']:.2%} of generations"
        )

    offsets = alignment_offsets(frames)
    if offsets.empty:
        print("No run detected any bridge entity -- nothing to align a heat map on")
    else:
        grids = {name: detection_grid(frame, offsets) for name, frame in frames.items()}
        print(f"Wrote {plot_detection_grids(grids, args.plot_dir)}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main(args)
    print(f"Wrote {args.report}")
