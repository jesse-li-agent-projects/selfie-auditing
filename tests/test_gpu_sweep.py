"""Tier-2 tests: the things a fake tokenizer cannot check (plan S6.1).

Real 1B smoke model on a real CUDA device, through the ordinary load paths.
These say nothing about whether the sweep *finds* anything -- the adapter and
the LoRA are untrained by construction -- only that negative offsets address
the tokens the design claims, and that two shards really produce two
different halves of a cell.
"""

from dataclasses import replace

import pytest
import torch

from config import (
    DUMMY_ADAPTER_FILE,
    DUMMY_BASE_MODEL,
    DUMMY_WORD,
    SECRET_PROMPT,
    Arm,
    Position,
)
from dummy_weights import create_random_selfie_adapter, embedding_norm
from extract import build_prompt, extract_hidden_states, user_prompt_span
from model_loading import load_base_model, load_tokenizer
from run_pipeline import run, smoke_config

pytestmark = [pytest.mark.gpu, pytest.mark.hf_cache]

DEVICE = "cuda:0"
LAYER = 8


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(DUMMY_BASE_MODEL)


@pytest.fixture(scope="module")
def model():
    return load_base_model(DUMMY_BASE_MODEL, device=DEVICE, dtype="bfloat16")


@pytest.fixture(scope="module")
def adapter(model, tmp_path_factory):
    from selfie_adapters import load_adapter

    path = create_random_selfie_adapter(
        model.config.hidden_size,
        tmp_path_factory.mktemp("adapter") / DUMMY_ADAPTER_FILE,
        embedding_norm(model),
    )
    return load_adapter(str(path), device=DEVICE)


def test_span_reads_the_intended_tokens(model, tokenizer):
    """Negative offsets address what plan S3 says they do, end to end through
    a real forward pass rather than through a fake tokenizer."""
    ids = tokenizer(
        build_prompt(tokenizer, SECRET_PROMPT, None),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    span = user_prompt_span(tokenizer, ids, SECRET_PROMPT)

    extraction = extract_hidden_states(
        model,
        tokenizer,
        SECRET_PROMPT,
        None,
        [LAYER],
        [Position.USER_PROMPT_SPAN],
        DEVICE,
    )
    absolute = [len(ids) + offset for offset in span]
    by_absolute = extract_hidden_states(
        model, tokenizer, SECRET_PROMPT, None, [LAYER], absolute, DEVICE
    )

    assert extraction.positions == span
    assert [extraction.tokens[f"pos{o}"] for o in span] == [
        tokenizer.decode([ids[o]]) for o in span
    ]
    for offset, index in zip(span, absolute):
        assert torch.equal(
            extraction.hidden_states[(LAYER, offset)],
            by_absolute.hidden_states[(LAYER, index)],
        )


def shard_config(tmp_path, sample_start, n_samples):
    """A one-cell, control-only slice of the smoke config -- enough to
    exercise the shard path without paying for the whole smoke sweep.

    :param tmp_path: output directory for this shard
    :param sample_start: index of this shard's first generation
    :param n_samples: generations per cell for this shard
    :return: a one-cell smoke pipeline config
    """
    return replace(
        smoke_config(tmp_path),
        arms=[Arm.CONTROL],
        layers=[LAYER],
        positions=[Position.USER_PROMPT_SPAN],
        n_samples=n_samples,
        sample_start=sample_start,
        device=DEVICE,
    )


def test_two_shards_produce_different_generations(model, tokenizer, adapter, tmp_path):
    """If cell_seed were ignored, or both shards seeded identically, a
    "2n-sample" cell would really be n samples counted twice -- and the merged
    output would look perfectly healthy. Nothing in tier 1 can catch that."""
    from merge_results import merge
    from results_store import read_cells

    n = 4
    paths = [
        run(
            shard_config(tmp_path, start, n),
            adapter=adapter,
            tokenizer=tokenizer,
            peft_model=model,
        )
        for start in (0, n)
    ]

    def generations(path):
        cells = read_cells(path)
        return {
            cell["position"]: cell["generations"]
            for cell in cells
            if cell["word"] == DUMMY_WORD and cell["layer"] == LAYER
        }

    first, second = generations(paths[0]), generations(paths[1])
    assert first.keys() == second.keys()
    assert all(len(g) == n for g in first.values())
    assert first != second

    merged = merge(tmp_path, total=2 * n)
    assert all(len(cell["generations"]) == 2 * n for cell in read_cells(merged))
