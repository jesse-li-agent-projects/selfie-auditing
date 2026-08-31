"""selfie_on_assistant.py's resume bookkeeping.

The part of a resume that has to be right without a GPU: what an interrupted
shard is judged to have finished, and when a directory counts as the same run
at all. Everything here is file-level, so none of it loads weights.
"""

import json

import pytest

from results_store import append_cell, write_metadata
from selfie_on_assistant import completed_keys, recorded_responses

CELL = {"arm": "control", "word": "book", "layer": 0, "position": "resp0"}
METADATA = {
    "sample_range": [0, 2],
    "layers": [0],
    "responses": {"control": {"book": {"text": "hi"}}},
}


@pytest.fixture
def cells_path(tmp_path):
    return tmp_path / "results_000000_000002.jsonl"


def test_a_shard_that_never_started_has_finished_nothing(cells_path):
    assert completed_keys(cells_path) == set()


def test_written_cells_are_not_regenerated(cells_path):
    with open(cells_path, "w") as handle:
        append_cell(handle, CELL)
        append_cell(handle, {**CELL, "layer": 3})

    assert completed_keys(cells_path) == {
        ("control", "book", 0, "resp0"),
        ("control", "book", 3, "resp0"),
    }


def test_a_half_written_line_is_reported_not_swallowed(cells_path):
    """A crash mid-write truncates the last line. Silently dropping it would
    be worse than stopping: the cell is neither complete nor regenerated."""
    cells_path.write_text(json.dumps(CELL) + "\n" + json.dumps(CELL)[:20])

    with pytest.raises(SystemExit, match="malformed line"):
        completed_keys(cells_path)


def test_a_fresh_directory_has_no_recorded_replies(cells_path):
    assert recorded_responses(cells_path, METADATA) == {}


def test_a_resume_reuses_the_replies_it_was_interrupted_on(cells_path):
    write_metadata(cells_path, METADATA)

    assert recorded_responses(cells_path, {**METADATA, "responses": {}}) == (
        METADATA["responses"]
    )


def test_a_directory_written_by_a_different_run_is_refused(cells_path):
    """Resuming across a settings change would mix two experiments into one
    file, with nothing on disk saying so."""
    write_metadata(cells_path, METADATA)

    with pytest.raises(SystemExit, match="layers"):
        recorded_responses(cells_path, {**METADATA, "layers": [0, 1]})
