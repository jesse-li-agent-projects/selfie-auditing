"""Bridge-entity experiment: dataset parsing, alias matching, and the parts of
the sweep that decide what gets regenerated after a crash.

The generation itself needs a real model and is covered by the GPU sweep
tests; what is pinned here is everything that decides *which* generations run
and how they are scored.
"""

import json

import pytest

from bridge_entity.bridge_dataset import (
    BridgeQuestion,
    answer_hop_messages,
    answer_matches,
    bridge_hop_messages,
    parse_aliases,
    shuffled,
)
from bridge_entity.explore_bridge_entity import (
    grouped_cells,
    index_entry,
    question_payload,
    sort_index,
)
from bridge_entity.run_bridge_entity import (
    check_settings,
    completed_cells,
    question_seed,
)
from scoring import contains_alias


def make_question(**overrides) -> BridgeQuestion:
    fields = {
        "id": "0",
        "statement": "The author of the novel Nineteen Eighty-Four was born in the city of",
        "bridge_statement": "The author of the novel Nineteen Eighty-Four is",
        "bridge_value": "George Orwell",
        "bridge_aliases": ("George Orwell", "Eric Blair"),
        "bridge_category": "person",
        "answer_value": "Motihari",
        "answer_aliases": ("Motihari",),
        "answer_category": "city",
        "category": "novel-author-birthcity",
        "fact_comp_type": "birthcity of novel's author",
    }
    return BridgeQuestion(**{**fields, **overrides})


# --- alias parsing ----------------------------------------------------------


def test_parse_aliases_unwraps_the_nested_tuple():
    raw = "(('George Orwell', 'Eric Blair'),)"
    assert parse_aliases(raw, "George Orwell") == ("George Orwell", "Eric Blair")


def test_parse_aliases_puts_the_canonical_name_first_without_duplicating_it():
    raw = "(('Eric Blair', 'George Orwell'),)"
    assert parse_aliases(raw, "George Orwell") == ("George Orwell", "Eric Blair")


@pytest.mark.parametrize("raw", ["", "not a tuple", "((),)"])
def test_parse_aliases_falls_back_to_the_canonical_name(raw):
    """An unusable alias cell must not leave a question that matches nothing."""
    assert parse_aliases(raw, "Motihari") == ("Motihari",)


# --- matching an entity in free text ----------------------------------------


def test_contains_alias_matches_any_alias_case_insensitively():
    assert contains_alias(
        "probably eric blair, the writer", ("George Orwell", "Eric Blair")
    )


def test_contains_alias_requires_a_whole_phrase():
    assert not contains_alias("Georgetown Orwellian", ("George Orwell",))


def test_contains_alias_tolerates_spacing_around_periods():
    assert contains_alias("written by J. D. Salinger", ("J.D. Salinger",))
    assert contains_alias("written by J.D. Salinger", ("J. D. Salinger",))


def test_contains_alias_of_nothing_is_false():
    assert not contains_alias("George Orwell", ())


# --- the dataset filter's own judgement -------------------------------------


@pytest.mark.parametrize(
    "reply", ["Motihari", "motihari.", "The answer is Motihari", "Motihari, India"]
)
def test_answer_matches_accepts_a_wrapped_or_punctuated_reply(reply):
    assert answer_matches(reply, ("Motihari",))


def test_answer_matches_rejects_a_different_entity():
    assert not answer_matches("Athens", ("Motihari",))


def test_answer_matches_rejects_an_empty_reply():
    """Substring matching runs both ways, so an empty reply would otherwise be
    a substring of every alias."""
    assert not answer_matches("   ", ("Motihari",))


def test_filter_prompts_ask_each_hop_after_its_own_exemplar():
    question = make_question()
    bridge, answer = bridge_hop_messages(question), answer_hop_messages(question)
    assert [message["role"] for message in bridge] == ["user", "assistant", "user"]
    assert bridge[1]["content"] == "George Orwell"
    assert bridge[2]["content"].endswith(question.bridge_statement)
    assert answer[1]["content"] == "Ottawa"
    assert answer[2]["content"].endswith(question.statement)


def test_filter_prompt_renames_the_person_category():
    """The dataset says "person"; the reference filter asks for "a human"."""
    assert "name of a human" in bridge_hop_messages(make_question())[2]["content"]


def test_shuffled_order_is_reproducible_and_a_permutation():
    questions = [make_question(id=str(i)) for i in range(20)]
    first = [q.id for q in shuffled(questions, seed=7)]
    assert first == [q.id for q in shuffled(questions, seed=7)]
    assert sorted(first, key=int) == [q.id for q in questions]
    assert first != [q.id for q in shuffled(questions, seed=8)]


# --- resume -----------------------------------------------------------------


def write_cells(path, cells):
    path.write_text("".join(json.dumps(cell) + "\n" for cell in cells))


def test_completed_cells_is_empty_before_the_first_run(tmp_path):
    assert completed_cells(tmp_path / "cells.jsonl") == set()


def test_completed_cells_reads_back_every_written_key(tmp_path):
    path = tmp_path / "cells.jsonl"
    write_cells(
        path,
        [
            {"question_id": "0", "layer": 3, "position": "pos-5", "hits": [True]},
            {"question_id": "0", "layer": 4, "position": "pos-5", "hits": [False]},
        ],
    )
    assert completed_cells(path) == {("0", 3, "pos-5"), ("0", 4, "pos-5")}


def test_completed_cells_names_the_repair_for_a_truncated_file(tmp_path):
    path = tmp_path / "cells.jsonl"
    write_cells(path, [{"question_id": "0", "layer": 3, "position": "pos-5"}])
    with open(path, "a") as handle:
        handle.write('{"question_id": "0", "layer": 4, "pos')
    with pytest.raises(ValueError, match="incomplete line"):
        completed_cells(path)


def test_check_settings_accepts_a_resume_with_the_same_settings(tmp_path):
    path = tmp_path / "cells.jsonl"
    settings = {
        "model": "m",
        "adapter": "a.pt",
        "questions": "q.jsonl",
        "n_samples": 10,
        "max_new_tokens": 20,
        "temperature": 0.7,
    }
    check_settings(path, settings)
    check_settings(path, {**settings, "batch_size": 64})


def test_check_settings_refuses_to_mix_two_adapters_into_one_file(tmp_path):
    path = tmp_path / "cells.jsonl"
    settings = {
        "model": "m",
        "adapter": "a.pt",
        "questions": "q.jsonl",
        "n_samples": 10,
        "max_new_tokens": 20,
        "temperature": 0.7,
    }
    check_settings(path, settings)
    with pytest.raises(ValueError, match="different settings"):
        check_settings(path, {**settings, "adapter": "b.pt"})


def test_question_seed_is_stable_and_question_specific():
    assert question_seed("124088") == question_seed("124088")
    assert question_seed("124088") != question_seed("124089")


# --- shaping a sweep for the explorer ---------------------------------------


def make_cell(**overrides) -> dict:
    fields = {
        "question_id": "0",
        "layer": 0,
        "position": "pos-3",
        "token": " The",
        "bridge_entity": "George Orwell",
        "generations": ["the author", "a novel"],
        "hits": [False, False],
        "hit_rate": 0.0,
    }
    return {**fields, **overrides}


def test_grouped_cells_yields_each_question_whole(tmp_path):
    path = tmp_path / "cells.jsonl"
    write_cells(
        path,
        [
            make_cell(question_id="a", layer=0),
            make_cell(question_id="a", layer=1),
            make_cell(question_id="b", layer=0),
        ],
    )
    assert [(qid, len(cells)) for qid, cells in grouped_cells(path)] == [
        ("a", 2),
        ("b", 1),
    ]


def test_grouped_cells_refuses_a_question_split_across_the_file(tmp_path):
    path = tmp_path / "cells.jsonl"
    write_cells(
        path,
        [
            make_cell(question_id="a"),
            make_cell(question_id="b"),
            make_cell(question_id="a", layer=1),
        ],
    )
    with pytest.raises(ValueError, match="splits question a"):
        list(grouped_cells(path))


def test_question_payload_orders_columns_from_statement_start_to_end():
    cells = [
        make_cell(position="pos-1", token=" of"),
        make_cell(position="pos-3", token=" The"),
    ]
    payload = question_payload(cells, make_question())
    assert [column["offset"] for column in payload["columns"]] == [-3, -1]
    assert [column["token"] for column in payload["columns"]] == [" The", " of"]


def test_question_payload_labels_a_repeated_token_distinctly():
    cells = [
        make_cell(position="pos-4", token=" the"),
        make_cell(position="pos-2", token=" the"),
    ]
    labels = [column["label"] for column in question_payload(cells, None)["columns"]]
    assert len(set(labels)) == 2


def test_question_payload_leaves_an_unswept_cell_null():
    cells = [
        make_cell(layer=0, position="pos-2", hit_rate=0.5, hits=[True, False]),
        make_cell(layer=0, position="pos-1"),
        make_cell(layer=1, position="pos-1"),
    ]
    payload = question_payload(cells, make_question())
    assert payload["layers"] == [0, 1]
    # Rows are layers, columns are token offsets ascending: layer 1 was only
    # swept at the last token, so its first column has neither rate nor cells.
    assert payload["hit_rate"] == [[0.5, 0.0], [None, 0.0]]
    assert payload["cells"][1][0] is None
    assert payload["cells"][0][0]["hits"] == [True, False]


def test_question_payload_takes_the_statement_from_the_question_set():
    payload = question_payload([make_cell()], make_question())
    assert payload["statement"].startswith("The author of the novel")
    assert payload["bridge_entity"] == "George Orwell"


def test_question_payload_survives_a_question_the_set_no_longer_has():
    assert question_payload([make_cell()], None)["statement"] == ""


def test_index_entry_reports_the_question_s_best_cell():
    payload = question_payload(
        [
            make_cell(layer=0, hit_rate=0.0),
            make_cell(layer=1, hit_rate=0.3),
        ],
        make_question(),
    )
    assert index_entry(payload)["best_hit_rate"] == 0.3


def test_sort_index_puts_the_detected_questions_first():
    entries = [
        {"id": "c", "best_hit_rate": 0.0},
        {"id": "a", "best_hit_rate": 0.2},
        {"id": "b", "best_hit_rate": 0.6},
    ]
    assert [entry["id"] for entry in sort_index(entries)] == ["b", "a", "c"]
