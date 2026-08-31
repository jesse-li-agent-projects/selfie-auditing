"""Compare taboo_baseline (SelfIE paper adapter) vs taboo_armB (unverbalized-thought
adapter) hit rates, across model organism and taboo word.

    python compare_taboo_arms.py

Reads the `run_pipeline.py` sweeps at outputs/taboo_baseline/ and
outputs/taboo_armB/ and writes one hit-rate heat map per (model organism,
taboo word) to plot/, each with a baseline subplot and an armB subplot on a
shared color scale.
"""

import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir", type=Path, default=Path("outputs/taboo_baseline")
    )
    parser.add_argument("--arm-b-dir", type=Path, default=Path("outputs/taboo_armB"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plot"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

import matplotlib.pyplot as plt
import pandas as pd

from results_store import cell_key, read_cells


def load_cells(adapter_dir: Path) -> pd.DataFrame:
    """Load every cell of an adapter's sweep into a flat frame.

    Searches recursively for shard files, since `taboo_baseline` splits its
    sweep into one word per subdirectory while `taboo_armB` keeps all words
    in one. Raises if a directory holds more than one shard per cell: those
    need `merge_results.py` first, not naive concatenation.

    :param adapter_dir: an adapter's `run_pipeline.py` output directory
    :return: one row per (arm, word, layer, position) cell
    :raises ValueError: if any cell appears in more than one shard
    """
    rows = []
    seen = set()
    for shard_path in sorted(adapter_dir.rglob("results_*_*.jsonl")):
        for cell in read_cells(shard_path):
            key = cell_key(cell)
            if key in seen:
                raise ValueError(
                    f"{adapter_dir} holds more than one shard for cell {key} -- "
                    "run merge_results.py first"
                )
            seen.add(key)
            rows.append(cell)
    frame = pd.DataFrame(rows)
    frame["position_offset"] = (
        frame["position"].str.removeprefix("pos-").astype(int) * -1
    )
    return frame


def plot_hit_rate_heatmaps(
    baseline: pd.DataFrame, arm_b: pd.DataFrame, plot_dir: Path
) -> None:
    """Write one hit-rate heat map (layer x position) per (arm, word).

    Each figure has a baseline and an armB subplot sharing one color scale, so
    the two adapters are visually comparable.

    :param baseline: `load_cells` output for taboo_baseline
    :param arm_b: `load_cells` output for taboo_armB
    :param plot_dir: directory to write PNGs into
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    organisms = sorted(set(baseline["arm"]) & set(arm_b["arm"]))
    words = sorted(set(baseline["word"]) & set(arm_b["word"]))

    for organism in organisms:
        for word in words:
            base_cell = baseline[
                (baseline["arm"] == organism) & (baseline["word"] == word)
            ]
            arm_b_cell = arm_b[(arm_b["arm"] == organism) & (arm_b["word"] == word)]
            base_grid = base_cell.pivot(
                index="layer", columns="position_offset", values="hit_rate"
            )
            arm_b_grid = arm_b_cell.pivot(
                index="layer", columns="position_offset", values="hit_rate"
            )

            vmax = max(base_grid.to_numpy().max(), arm_b_grid.to_numpy().max())

            fig, (ax_base, ax_arm_b) = plt.subplots(1, 2, sharey=True)
            image = None
            for ax, grid, title in (
                (ax_base, base_grid, "baseline"),
                (ax_arm_b, arm_b_grid, "armB"),
            ):
                image = ax.imshow(grid, vmin=0, vmax=vmax)
                ax.set_xticks(
                    range(len(grid.columns)), labels=grid.columns, rotation=90
                )
                ax.set_yticks(range(len(grid.index)), labels=grid.index)
                ax.set_xlabel("extracted token position")
                ax.set_title(title)
            ax_base.set_ylabel("extracted layer")
            fig.colorbar(image, ax=(ax_base, ax_arm_b), label="hit rate")
            fig.suptitle(f"organism={organism}, word={word}")
            fig.savefig(
                plot_dir / f"hit_rate_{organism}_{word}.png", bbox_inches="tight"
            )
            plt.close(fig)


if __name__ == "__main__":
    baseline_cells = load_cells(args.baseline_dir)
    arm_b_cells = load_cells(args.arm_b_dir)
    plot_hit_rate_heatmaps(baseline_cells, arm_b_cells, args.plot_dir)
