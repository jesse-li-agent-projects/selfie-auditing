"""Regression test for the measured token span (plan S2/S6.2).

Tokenizer only -- no model weights, no GPU. This is the one place hardcoding
the span's length and contents is correct: pinning the measured value against
template drift is the whole purpose. The 1B smoke tokenizer gives the same
answer as the gated 8B (plan S2), so this needs no 8B access.

Prompt *lengths* are pinned too. The span offsets alone would not have caught
the mirror-model discrepancy S2 describes, where every absolute index moved by
20-25 tokens while the span stayed at -11 .. -1.
"""

import pytest
import torch

from config import SECRET_PROMPT, Arm
from extract import build_prompt, user_prompt_span
from model_loading import load_tokenizer, system_prompt_for
from smoke.small_llama_config import SMOKE_MODEL

pytestmark = pytest.mark.hf_cache

EXPECTED_SPAN = list(range(-11, 0))
EXPECTED_TOKENS = [
    "What",
    " is",
    " the",
    " secret",
    " word",
    "?",
    "<|eot_id|>",
    "<|start_header_id|>",
    "assistant",
    "<|end_header_id|>",
    "\n\n",
]
EXPECTED_PROMPT_LENGTHS = {Arm.CONTROL: 62, Arm.PROMPTED: 66, Arm.FINETUNED: 41}
WORD = "gold"


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(SMOKE_MODEL)


def prompt_ids(tokenizer, arm: Arm) -> torch.Tensor:
    formatted = build_prompt(tokenizer, SECRET_PROMPT, system_prompt_for(arm, WORD))
    return tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]


@pytest.mark.parametrize("arm", list(Arm))
def test_span_matches_the_measured_offsets(tokenizer, arm):
    ids = prompt_ids(tokenizer, arm)

    span = user_prompt_span(tokenizer, ids, SECRET_PROMPT)

    assert span == EXPECTED_SPAN
    assert [tokenizer.decode([ids[o]]) for o in span] == EXPECTED_TOKENS
    assert tokenizer.decode(ids[len(ids) + span[0] :]).startswith(SECRET_PROMPT)


@pytest.mark.parametrize("arm", list(Arm))
def test_prompt_length_matches_the_measured_length(tokenizer, arm):
    assert len(prompt_ids(tokenizer, arm)) == EXPECTED_PROMPT_LENGTHS[arm]


def test_span_is_identical_across_arms(tokenizer):
    spans = {
        arm: user_prompt_span(tokenizer, prompt_ids(tokenizer, arm), SECRET_PROMPT)
        for arm in Arm
    }

    assert len(set(map(tuple, spans.values()))) == 1
