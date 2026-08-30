"""Tests for the batching utilities shared by the pangram and baseline
extraction scripts."""

import pytest
import torch

from adapter_training.extract_common import left_pad, position_ids_from_mask
from config import DUMMY_BASE_MODEL


def test_left_pad_puts_content_at_the_end():
    input_ids, mask = left_pad([[5, 6], [7, 8, 9]], pad_id=0)

    assert input_ids.tolist() == [[0, 5, 6], [7, 8, 9]]
    assert mask.tolist() == [[0, 1, 1], [1, 1, 1]]


def test_position_ids_ignore_left_padding():
    _, mask = left_pad([[5, 6], [7, 8, 9]], pad_id=0)

    # The short row's real tokens must still be positions 0 and 1, not 1 and 2.
    assert position_ids_from_mask(mask).tolist() == [[0, 0, 1], [0, 1, 2]]


@pytest.mark.hf_cache
def test_naive_position_ids_are_equivalent_under_rope():
    """`position_ids_from_mask` is not fixing a live bug.

    A plain forward pass with `position_ids=None` gives every row the same
    `arange(seq_len)`, so under left padding a short row's real tokens land
    at positions shifted by that row's pad count relative to running it
    alone -- what upstream's own extraction does, and what §9.4 of
    `plans/pangram_extraction_adapter.md` once suspected of explaining the
    step-0 gate's 1.7800-vs-1.3662 gap. It doesn't: RoPE attention scores
    are a function of the *relative* offset between query and key, and
    padding is excluded from attention by the causal mask, so a constant
    per-row shift cancels out -- only the spacing among a row's own real
    tokens ever affects the output, and that spacing is identical either
    way. Confirmed here against the real model, not just derived; see
    `plans/notes/step0_findings.md`.
    """
    from model_loading import load_base_model, load_tokenizer

    tokenizer = load_tokenizer(DUMMY_BASE_MODEL)
    model = load_base_model(DUMMY_BASE_MODEL, device="cpu", dtype="float32")
    sequences = [
        tokenizer("Tell me about Ada Lovelace.", add_special_tokens=False).input_ids,
        tokenizer(
            "Tell me about the Peloponnesian War and its immediate causes.",
            add_special_tokens=False,
        ).input_ids,
    ]
    input_ids, attention_mask = left_pad(sequences, tokenizer.pad_token_id)

    with torch.no_grad():
        mask_aware = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids_from_mask(attention_mask),
            output_hidden_states=True,
        ).hidden_states[-1][:, -1, :]
        naive = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            output_hidden_states=True,
        ).hidden_states[-1][:, -1, :]

    similarity = torch.nn.functional.cosine_similarity(
        mask_aware.float(), naive.float(), dim=-1
    )
    assert similarity.min() > 0.999
