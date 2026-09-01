"""Bridge-entity experiment: dataset parsing, alias matching, and the parts of
the sweep that decide what gets regenerated after a crash.

The generation itself needs a real model and is covered by the GPU sweep
tests; what is pinned here is everything that decides *which* generations run
and how they are scored.
"""

import json

import pytest

from bridge_dataset import (
    BridgeQuestion,
    answer_hop_messages,
    answer_matches,
    bridge_hop_messages,
    parse_aliases,
    shuffled,
)
from run_bridge_entity import check_settings, completed_cells, question_seed
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
