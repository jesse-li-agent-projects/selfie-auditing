"""Browse one bridge-entity sweep question by question, generations included.

    python -m bridge_entity.explore_bridge_entity --run bridge_entity/bg_think
    cd outputs/bridge_entity/bg_think_explorer && python -m http.server

`report_bridge_entity.py` averages over questions and drops the generations;
this bakes the level below that into a small static site -- one question's
layer x token heat map at a time, with every generation behind a cell readable
on hover and the ones that recovered the bridge entity marked.

A sweep is tens of megabytes of generations, far too much for one page, so
each question is written as its own JSON file and fetched only when picked.
That is also why opening the page as a plain file does not work: browsers
refuse `fetch` from a `file://` page, hence the one-line server above.
"""

import argparse
import json
from pathlib import Path

# Light import: config.py pulls in no heavy dependencies, so --help stays fast.
from config import outputs_relative


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run",
        type=outputs_relative,
        default="bridge_entity/bg_think",
        help="a run_bridge_entity.py output directory, under outputs/ "
        "(implicitly prepended)",
    )
    parser.add_argument(
        "--questions",
        type=outputs_relative,
        default="bridge_entity/questions.jsonl",
        help="filter_bridge_questions.py's output, under outputs/ "
        "(implicitly prepended)",
    )
    parser.add_argument(
        "--output-dir",
        type=outputs_relative,
        default="bridge_entity/bg_think_explorer",
        help="where the site goes, under outputs/ (implicitly prepended)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

import shutil  # noqa: E402
from itertools import groupby  # noqa: E402

from bridge_entity.bridge_dataset import (  # noqa: E402
    BridgeQuestion,
    read_question_file,
)
from results_store import CELLS_FILE, read_cells, read_metadata  # noqa: E402

PAGE_FILE = "index.html"
INDEX_FILE = "index.json"
DATA_DIR = "data"


def grouped_cells(cells_path: Path):
    """Stream a cells file one question at a time.

    Groups rather than indexes, so a sweep is never held in memory whole. A
    sweep writes each question's cells consecutively; if that stops being true
    this raises rather than quietly writing half a question.

    :param cells_path: a run's cells file
    :return: (question id, its cells) pairs, in file order
    :raises ValueError: if one question's cells are not consecutive
    """
    seen = set()
    for question_id, cells in groupby(
        read_cells(cells_path), key=lambda cell: cell["question_id"]
    ):
        if question_id in seen:
            raise ValueError(
                f"{cells_path} splits question {question_id} across the file; it "
                "can only be read a question at a time if each one is contiguous"
            )
        seen.add(question_id)
        yield question_id, list(cells)


def cell_offset(cell: dict) -> int:
    """A cell's token position as its end-relative integer offset."""
    return int(cell["position"].removeprefix("pos"))


def column_label(token: str, offset: int) -> str:
    """A token's x-axis label, made unique by its offset.

    A statement can use the same word twice, and a categorical axis merges two
    columns that share a label.
    """
    return f"{token.strip() or '_'} {offset}"


def question_payload(cells: list[dict], question: BridgeQuestion | None) -> dict:
    """One question's heat map, and the generations behind every cell of it.

    Columns run in prompt order: offsets are end-relative, so ascending is
    left to right. A cell the sweep never wrote is null in both grids.

    :param cells: every cell of one question
    :param question: its entry in the question set, if the set still holds one
    :return: the question's `data/<id>.json` contents
    """
    layers = sorted({cell["layer"] for cell in cells})
    offsets = sorted({cell_offset(cell) for cell in cells})
    tokens = {cell_offset(cell): cell["token"] for cell in cells}
    indexed = {(cell["layer"], cell_offset(cell)): cell for cell in cells}
    grid = [[indexed.get((layer, offset)) for offset in offsets] for layer in layers]
    return {
        "question_id": cells[0]["question_id"],
        "statement": question.statement if question else "",
        "bridge_entity": cells[0]["bridge_entity"],
        "layers": layers,
        "columns": [
            {
                "offset": offset,
                "token": tokens[offset],
                "label": column_label(tokens[offset], offset),
            }
            for offset in offsets
        ],
        "hit_rate": [
            [None if cell is None else cell["hit_rate"] for cell in row] for row in grid
        ],
        "cells": [
            [
                (
                    None
                    if cell is None
                    else {"generations": cell["generations"], "hits": cell["hits"]}
                )
                for cell in row
            ]
            for row in grid
        ],
    }


def index_entry(payload: dict) -> dict:
    """A question's line in the picker, and what the picker sorts on.

    :param payload: `question_payload`'s output
    :return: the question's `index.json` entry
    """
    best = max(
        (rate for row in payload["hit_rate"] for rate in row if rate is not None),
        default=0.0,
    )
    return {
        "id": payload["question_id"],
        "statement": payload["statement"],
        "bridge_entity": payload["bridge_entity"],
        "best_hit_rate": best,
    }


def sort_index(entries: list[dict]) -> list[dict]:
    """Best-detected questions first, undetected ones last but still listed.

    A sweep detects the entity in a minority of its questions and those are
    the ones worth reading; the rest stay in the picker, since a question that
    never fires is itself something to look at.
    """
    return sorted(entries, key=lambda entry: (-entry["best_hit_rate"], entry["id"]))


def write_site(output_dir: Path, index: dict, payloads: list[dict]) -> None:
    """Write the page, the question picker's index, and one file per question."""
    import plotly

    data_dir = output_dir / DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        (data_dir / f"{payload['question_id']}.json").write_text(json.dumps(payload))
    (output_dir / INDEX_FILE).write_text(json.dumps(index, indent=2))
    (output_dir / PAGE_FILE).write_text(PAGE)
    # Bundled rather than pulled from a CDN: the page is served locally, on a
    # machine that may have no route out.
    shutil.copy(
        Path(plotly.__file__).parent / "package_data" / "plotly.min.js",
        output_dir / "plotly.min.js",
    )


def main(args) -> Path:
    cells_path = args.run / CELLS_FILE
    questions = {
        question.id: question for question in read_question_file(args.questions)
    }
    payloads = [
        question_payload(cells, questions.get(question_id))
        for question_id, cells in grouped_cells(cells_path)
    ]
    index = {
        "run": str(args.run),
        "adapter": read_metadata(cells_path).get("adapter", ""),
        "questions": sort_index([index_entry(payload) for payload in payloads]),
    }
    write_site(args.output_dir, index, payloads)
    return args.output_dir


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bridge-entity explorer</title>
<script src="plotly.min.js"></script>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb;
    --surface-raised: #ffffff;
    --line: #e2e1dc;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #86847e;
    --hit: #1baf7a;
  }
  body {
    margin: 0; padding: 20px 24px; background: var(--surface);
    color: var(--text-primary);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  h1 { font-size: 17px; margin: 0 0 4px; font-weight: 600; }
  #provenance { color: var(--text-muted); font-size: 12px; margin-bottom: 14px; }
  select { font: inherit; padding: 5px 8px; width: min(900px, 100%);
           border: 1px solid var(--line); border-radius: 6px;
           background: var(--surface-raised); color: var(--text-primary); }
  #statement { margin: 10px 0 14px; color: var(--text-secondary); }
  #statement b { color: var(--text-primary); }
  .split { display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 20px;
           align-items: start; }
  #panel { border: 1px solid var(--line); border-radius: 8px;
           background: var(--surface-raised); padding: 14px 16px;
           position: sticky; top: 20px; max-height: 82vh; overflow-y: auto; }
  #panel h2 { font-size: 13px; margin: 0 0 2px; font-weight: 600; }
  #panel .sub { color: var(--text-secondary); font-size: 12px; margin-bottom: 10px; }
  #panel.pinned { border-color: var(--hit); }
  ol { margin: 0; padding-left: 22px; }
  li { margin-bottom: 6px; color: var(--text-secondary); }
  li.hit { color: var(--text-primary); }
  li.hit::marker { color: var(--hit); font-weight: 700; }
  .hint { color: var(--text-muted); font-size: 12px; }
</style>
</head>
<body>
<h1>Bridge-entity detection, per question</h1>
<div id="provenance"></div>
<select id="picker"></select>
<div id="statement"></div>
<div class="split">
  <div id="plot"></div>
  <div id="panel"></div>
</div>
<script>
// The sequential blue ramp, light -> dark, so a zero hit rate recedes into
// the page surface rather than reading as a colour of its own.
const BLUES = [
  [0.0, "#fcfcfb"], [0.15, "#cde2fb"], [0.3, "#9ec5f4"], [0.5, "#5598e7"],
  [0.7, "#2a78d6"], [0.85, "#1c5cab"], [1.0, "#0d366b"]
];
const HINT = "Hover a cell to read its generations. Click to pin it, click again to release.";

const cache = {};
let current = null;
let pinned = null;

const picker = document.getElementById("picker");
const panel = document.getElementById("panel");
const plot = document.getElementById("plot");

function percent(rate) { return Math.round(rate * 100) + "%"; }

async function boot() {
  const index = await (await fetch("index.json")).json();
  document.getElementById("provenance").textContent =
    index.run + "  |  adapter: " + index.adapter;
  for (const entry of index.questions) {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = "[best " + percent(entry.best_hit_rate) + "]  " +
      entry.statement + " \\u2192 " + entry.bridge_entity;
    picker.appendChild(option);
  }
  picker.addEventListener("change", () => show(picker.value));
  await show(index.questions[0].id);
  // Only after the first draw: plotly attaches .on to the div when it plots.
  plot.on("plotly_hover", event => { if (!pinned) render(event.points[0].pointIndex); });
  plot.on("plotly_unhover", () => { if (!pinned) reset(); });
  plot.on("plotly_click", event => {
    const at = event.points[0].pointIndex;
    const same = pinned && pinned[0] === at[0] && pinned[1] === at[1];
    pinned = same ? null : at;
    panel.classList.toggle("pinned", !same);
    render(at);
  });
}

async function show(questionId) {
  if (!cache[questionId]) {
    cache[questionId] = await (await fetch("data/" + questionId + ".json")).json();
  }
  current = cache[questionId];
  pinned = null;
  panel.classList.remove("pinned");
  document.getElementById("statement").innerHTML =
    "Question " + current.question_id + ": &ldquo;" + current.statement +
    "&rdquo; &mdash; bridge entity <b>" + current.bridge_entity + "</b>";
  draw();
  reset();
}

function draw() {
  const trace = {
    type: "heatmap",
    z: current.hit_rate,
    x: current.columns.map(column => column.label),
    y: current.layers.map(String),
    zmin: 0, zmax: 1,
    colorscale: BLUES,
    xgap: 1, ygap: 1,
    hoverinfo: "none",
    colorbar: { title: { text: "hit rate", side: "right" }, thickness: 12,
                tickformat: ".0%", outlinewidth: 0 }
  };
  const layout = {
    height: 660,
    margin: { l: 60, r: 10, t: 10, b: 140 },
    paper_bgcolor: "#fcfcfb",
    plot_bgcolor: "#fcfcfb",
    font: { color: "#52514e", size: 12 },
    xaxis: { title: "prompt token, statement start to end", type: "category",
             tickangle: -60, ticks: "", showgrid: false, zeroline: false },
    yaxis: { title: "extracted layer", type: "category", ticks: "",
             showgrid: false, zeroline: false, dtick: 2 }
  };
  Plotly.react(plot, [trace], layout, { displayModeBar: false, responsive: true });
}

function reset() {
  const hint = document.createElement("span");
  hint.className = "hint";
  hint.textContent = HINT;
  panel.replaceChildren(hint);
}

function render([row, column]) {
  const cell = current.cells[row][column];
  const heading = document.createElement("h2");
  heading.textContent = "Layer " + current.layers[row] + ", token " +
    JSON.stringify(current.columns[column].token);
  const sub = document.createElement("div");
  sub.className = "sub";
  panel.replaceChildren(heading, sub);
  if (!cell) {
    sub.textContent = "no data for this cell";
    return;
  }
  const hits = cell.hits.filter(Boolean).length;
  sub.textContent = "recovered " + current.bridge_entity + " in " + hits + " of " +
    cell.hits.length + " generations" + (pinned ? " (pinned)" : "");
  const list = document.createElement("ol");
  cell.generations.forEach((text, index) => {
    const item = document.createElement("li");
    if (cell.hits[index]) { item.className = "hit"; }
    item.textContent = text;
    list.appendChild(item);
  });
  panel.appendChild(list);
}

boot();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(f"Wrote {main(args)}; serve it with `python -m http.server`")
