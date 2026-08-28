"""Baseline extraction logic tests (plan step 1).

The fast test drives `extract_baseline_vectors` with a fake model and
tokenizer. The `hf_cache` tests pin what only a real tokenizer/model can
answer: that batching does not change the vectors, and the output shapes.
"""

import pytest
import torch

from adapter_training.extract_baseline_vectors import extract_baseline_vectors
from config import DUMMY_BASE_MODEL
from conftest import topic


def test_baseline_extraction_keeps_the_last_prompt_token(fake):
    tokenizer, model = fake
    entry = topic("Alpha")
    prompt_ids = tokenizer(f"USER {entry.prompt} ASSISTANT").input_ids

    result = extract_baseline_vectors(model, tokenizer, [entry], layer=1, device="cpu")

    assert result.vectors.shape == (1, 8)
    assert result.vectors[0, 0].float().item() == float(prompt_ids[-1] + 2)


# --- real tokenizer / real model ------------------------------------------


@pytest.mark.hf_cache
def test_real_model_batching_matches_unbatched():
    """The padding-aware position ids of `position_ids_from_mask`, end to end.

    Without them the shorter prompt's rotary positions shift by the pad count
    and its vectors depend on what it shared a batch with.
    """
    from model_loading import load_base_model, load_tokenizer

    tokenizer = load_tokenizer(DUMMY_BASE_MODEL)
    model = load_base_model(DUMMY_BASE_MODEL, device="cpu", dtype="float32")
    topics = [
        topic("Ada Lovelace"),
        topic("The Peloponnesian War and its immediate causes in Greek history"),
    ]

    one = extract_baseline_vectors(
        model, tokenizer, topics, layer=8, batch_size=1, device="cpu"
    )
    two = extract_baseline_vectors(
        model, tokenizer, topics, layer=8, batch_size=2, device="cpu"
    )

    assert one.vectors.shape == (len(topics), model.config.hidden_size)
    assert one.vectors.shape == two.vectors.shape
    similarity = torch.nn.functional.cosine_similarity(
        one.vectors.float(), two.vectors.float(), dim=-1
    )
    assert similarity.min() > 0.999


@pytest.mark.hf_cache
def test_real_model_extraction_has_the_expected_shapes(tmp_path):
    from adapter_training.extract_baseline_vectors import write_outputs
    from model_loading import load_base_model, load_tokenizer

    tokenizer = load_tokenizer(DUMMY_BASE_MODEL)
    model = load_base_model(DUMMY_BASE_MODEL, device="cpu", dtype="float32")

    result = extract_baseline_vectors(
        model, tokenizer, [topic("Ada Lovelace", split="val")], layer=8, device="cpu"
    )
    write_outputs(tmp_path, result, 8, DUMMY_BASE_MODEL)

    assert result.vectors.shape == (1, model.config.hidden_size)
    assert result.vectors.dtype == torch.bfloat16
    assert (tmp_path / "vectors.pt").exists()
    assert (tmp_path / "topics.json").exists()
    assert (tmp_path / "positions.json").exists()
    assert (tmp_path / "position_means.pt").exists()
    assert not (tmp_path / "filter_report.json").exists()
