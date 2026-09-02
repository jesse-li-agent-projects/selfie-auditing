"""Regression test for the measured token span (plan S2/S6.2).

Tokenizer only -- no model weights, no GPU. The pinned values live in
preflight.py, which is also what a run checks them with; this file is the
pytest-side half of the same ratchet, so a pin edit has to face both. The 1B
dummy tokenizer gives the same answer as the gated 8B (plan S2), so this needs
no 8B access.
"""

import pytest
import torch

from config import SECRET_PROMPT, ModelOrganism, DUMMY_BASE_MODEL
from config import sweep_config as _sweep_config
from extract import build_prompt, user_prompt_span
from model_loading import load_tokenizer, system_prompt_for
from preflight import (
    PIN_WORD,
    PINNED_PROMPT_LENGTHS,
    PINNED_SPAN,
    PINNED_SPAN_TOKENS,
    PreflightError,
    check_run_prompts,
    check_tokenization_pins,
)

pytestmark = pytest.mark.hf_cache


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(DUMMY_BASE_MODEL)


def sweep_config(output_dir):
    return _sweep_config(
        [PIN_WORD, "moon"], layers=list(range(16)), output_dir=output_dir
    )


def prompt_ids(tokenizer, organism: ModelOrganism) -> torch.Tensor:
    formatted = build_prompt(tokenizer, SECRET_PROMPT, system_prompt_for(organism, PIN_WORD))
    return tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]


@pytest.mark.parametrize("organism", list(ModelOrganism))
def test_span_matches_the_measured_offsets(tokenizer, organism):
    ids = prompt_ids(tokenizer, organism)

    span = user_prompt_span(tokenizer, ids, SECRET_PROMPT)

    assert span == PINNED_SPAN
    assert [tokenizer.decode([ids[o]]) for o in span] == PINNED_SPAN_TOKENS
    assert tokenizer.decode(ids[len(ids) + span[0] :]).startswith(SECRET_PROMPT)


@pytest.mark.parametrize("organism", list(ModelOrganism))
def test_prompt_length_matches_the_measured_length(tokenizer, organism):
    assert len(prompt_ids(tokenizer, organism)) == PINNED_PROMPT_LENGTHS[organism]


def test_span_is_identical_across_arms(tokenizer):
    spans = {
        organism: user_prompt_span(tokenizer, prompt_ids(tokenizer, organism), SECRET_PROMPT)
        for organism in ModelOrganism
    }

    assert len(set(map(tuple, spans.values()))) == 1


def test_preflight_accepts_the_current_tokenizer(tokenizer, tmp_path):
    check_tokenization_pins(tokenizer)
    check_run_prompts(tokenizer, sweep_config(tmp_path))


def test_preflight_catches_pin_drift(tokenizer, monkeypatch):
    # The failure the cross-organism and cross-shard checks structurally cannot see:
    # every organism agreeing on a span that is no longer the measured one.
    monkeypatch.setattr("preflight.PINNED_SPAN", list(range(-9, 0)))

    with pytest.raises(PreflightError, match="pinned measurement"):
        check_tokenization_pins(tokenizer)


@pytest.mark.parametrize("position", [999, -999])
def test_preflight_rejects_a_position_outside_the_prompt(tokenizer, tmp_path, position):
    # Without this the offset survives to the forward pass and dies on an
    # IndexError, after the base model has already been downloaded and loaded.
    config = sweep_config(tmp_path)
    config.positions = [position]

    with pytest.raises(PreflightError, match="outside the prompt"):
        check_run_prompts(tokenizer, config)


def test_preflight_accepts_a_position_inside_the_shortest_prompt(tokenizer, tmp_path):
    # Prompt length differs by organism, so the shortest one is what binds.
    config = sweep_config(tmp_path)
    config.positions = [-min(PINNED_PROMPT_LENGTHS.values())]

    check_run_prompts(tokenizer, config)


def test_preflight_rejects_a_prompt_the_template_alters(tokenizer, tmp_path):
    # The chat template trims message content, so a prompt with trailing
    # whitespace never appears verbatim in the rendered text -- the class of
    # silent alteration that would leave the span pointing somewhere else.
    config = sweep_config(tmp_path)
    config.secret_prompt = SECRET_PROMPT + "   "

    with pytest.raises((PreflightError, ValueError)):
        check_run_prompts(tokenizer, config)
