"""Tests for adapter_training.loss (plan step 2a, tests 5-7 and 9-11).

Tests 5-7 drive `SoftPromptLoss` with a fake tokenizer/model, at no cost.
`FakeCharTokenizer` (conftest.py) tokenizes `<|...|>`-style tags as one
atomic token and everything else character by character, which is enough to
isolate `SELFIE_TEMPLATE`'s two `RESERVED_TOKEN` slots without a real BPE
tokenizer. The stub model's base transformer ignores `inputs_embeds` and
returns whatever `last_hidden_state` the test sets, with `lm_head` the
identity -- so the test controls "logits" directly and can hand-verify the
loss.

Tests 9-11 need the real Llama-3.2-1B tokenizer/model (`hf_cache`); test 10
is the one the plan calls out as protecting the eventual 1.3662 check.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter_training.loss import LossConfig, SoftPromptLoss, target_text
from conftest import FakeCharTokenizer
from interpret import RESERVED_TOKEN, SELFIE_TEMPLATE
from selfie_adapters.projection import create_projection_module

VOCAB = 300


class StubBaseModel:
    """Returns whatever `hidden` the test set, ignoring the actual input --
    stands in for the base transformer `model.model` that `SoftPromptLoss`
    calls to get `last_hidden_state` without paying for a full `lm_head`."""

    def __init__(self):
        self.hidden = None

    def __call__(self, inputs_embeds, attention_mask, use_cache=False):
        assert (
            self.hidden is not None
        ), "test must set .hidden before calling the scorer"
        assert self.hidden.shape[:2] == inputs_embeds.shape[:2]
        return SimpleNamespace(last_hidden_state=self.hidden)


@pytest.fixture
def loss_setup():
    tokenizer = FakeCharTokenizer()
    embed = nn.Embedding(2000, VOCAB)
    base = StubBaseModel()
    model = SimpleNamespace(
        model=base, lm_head=nn.Identity(), get_input_embeddings=lambda: embed
    )
    projection = create_projection_module(
        "scalar_affine", dim=VOCAB, normalize_input=False, device="cpu", init_scale=1.0
    )
    config = LossConfig(
        max_loss=100.0, label_smoothing=0.0, strip_labels=True, eos_token="<|eot_id|>"
    )
    scorer = SoftPromptLoss(model, tokenizer, projection, config)
    return scorer, tokenizer, base, config


def build_hidden(scorer, tokenizer, labels, config, peak_logit_fn):
    """Construct a `(hidden, expected_token_losses, target_lens)` triple: a
    fixed hidden-state tensor whose target-window rows peak at one class each
    (`peak_logit_fn(row, k, correct_id) -> (peak_class, peak_logit)`), plus
    the cross-entropy that peak implies for every valid (unpadded) position.
    """
    targets = [target_text(label, config) for label in labels]
    target_id_lists = [tokenizer(text).input_ids for text in targets]
    target_lens = [len(ids) for ids in target_id_lists]
    max_len = max(target_lens)
    batch = len(labels)

    hidden = torch.zeros(batch, scorer.template_len + max_len, VOCAB)
    expected = torch.zeros(batch, max_len)
    for i, ids in enumerate(target_id_lists):
        for k, correct in enumerate(ids):
            peak_class, peak_logit = peak_logit_fn(i, k, correct)
            row = torch.zeros(VOCAB)
            row[peak_class] = peak_logit
            hidden[i, scorer.template_len - 1 + k, :] = row
            expected[i, k] = F.cross_entropy(row.unsqueeze(0), torch.tensor([correct]))
    return hidden, expected, target_lens


# --- test 5: target construction -------------------------------------------


def test_target_text_appends_quote_and_eos_and_strips():
    config = LossConfig(strip_labels=True, eos_token="<|eot_id|>")
    assert target_text("a label ", config) == 'a label"<|eot_id|>'


def test_target_text_does_not_strip_when_disabled():
    config = LossConfig(strip_labels=False, eos_token="<|eot_id|>")
    assert target_text("a label ", config) == 'a label "<|eot_id|>'


# --- test 6: loss reduction --------------------------------------------------


def test_loss_reduction_matches_a_hand_computed_masked_mean(loss_setup):
    scorer, tokenizer, base, config = loss_setup
    labels = ["ab", "abcd"]  # different target lengths -> padding exists

    def wrong_class(row, k, correct):
        return (correct + 7) % VOCAB, 6.0

    hidden, expected_token_losses, target_lens = build_hidden(
        scorer, tokenizer, labels, config, wrong_class
    )
    base.hidden = hidden

    vectors = torch.zeros(len(labels), VOCAB)
    loss, stats = scorer(vectors, labels)

    expected_per_seq = torch.stack(
        [expected_token_losses[i, : target_lens[i]].mean() for i in range(len(labels))]
    )
    assert torch.isclose(loss, expected_per_seq.mean(), atol=1e-5)
    assert stats["total_valid_tokens"] == sum(target_lens)
    assert stats["total_clamped_tokens"] == 0


def test_padding_contributes_nothing_to_the_loss(loss_setup):
    """The shorter sequence's padded positions must not enter the mean --
    check by comparing against a batch of the shorter sequence alone."""
    scorer, tokenizer, base, config = loss_setup

    def wrong_class(row, k, correct):
        return (correct + 3) % VOCAB, 5.0

    labels = ["ab", "ab"]  # identical, so the batched and solo losses must match
    hidden, _, target_lens = build_hidden(
        scorer, tokenizer, labels, config, wrong_class
    )
    base.hidden = hidden
    vectors = torch.zeros(2, VOCAB)
    batched_loss, _ = scorer(vectors, labels)

    single_labels = ["ab"]
    single_hidden, _, _ = build_hidden(
        scorer, tokenizer, single_labels, config, wrong_class
    )
    base.hidden = single_hidden
    single_loss, _ = scorer(vectors[:1], single_labels)

    assert torch.isclose(batched_loss, single_loss, atol=1e-5)


# --- test 7: max_loss clamps per token, not per sequence --------------------


def test_max_loss_clamps_per_token(loss_setup):
    scorer, tokenizer, base, config = loss_setup
    config = LossConfig(
        max_loss=50.0, label_smoothing=0.0, strip_labels=True, eos_token="<|eot_id|>"
    )
    scorer.config = config
    labels = ["abc"]

    def one_huge_token(row, k, correct):
        # Token 1 gets a wildly wrong, huge-magnitude prediction; the others
        # are ordinary mismatches well under the clamp.
        if k == 1:
            return (correct + 1) % VOCAB, 500.0
        return (correct + 5) % VOCAB, 4.0

    hidden, expected_token_losses, target_lens = build_hidden(
        scorer, tokenizer, labels, config, one_huge_token
    )
    base.hidden = hidden
    vectors = torch.zeros(1, VOCAB)

    loss, stats = scorer(vectors, labels)

    clamped_by_token = expected_token_losses[0, : target_lens[0]].clamp(
        max=config.max_loss
    )
    expected_loss = clamped_by_token.mean()
    # Clamping the already-averaged sequence loss instead would give a
    # different (smaller) number -- this is what distinguishes per-token
    # clamping from per-sequence clamping.
    clamped_after_mean = (
        expected_token_losses[0, : target_lens[0]].mean().clamp(max=config.max_loss)
    )
    assert not torch.isclose(expected_loss, clamped_after_mean)

    assert torch.isclose(loss, expected_loss, atol=1e-5)
    assert stats["total_clamped_tokens"] == 1


# --- tests 9-11: real Llama-3.2-1B tokenizer/model --------------------------


@pytest.fixture(scope="module")
def real_model_and_tokenizer():
    from config import DUMMY_BASE_MODEL
    from model_loading import load_base_model, load_tokenizer

    tokenizer = load_tokenizer(DUMMY_BASE_MODEL)
    model = load_base_model(DUMMY_BASE_MODEL, device="cpu", dtype="float32")
    return model, tokenizer


@pytest.mark.hf_cache
def test_selfie_template_tokenizes_to_26_tokens_with_two_slots(
    real_model_and_tokenizer,
):
    _, tokenizer = real_model_and_tokenizer

    template_ids = tokenizer(SELFIE_TEMPLATE, add_special_tokens=False).input_ids
    reserved_id = tokenizer.convert_tokens_to_ids(RESERVED_TOKEN)
    inject_positions = [i for i, tid in enumerate(template_ids) if tid == reserved_id]

    assert len(template_ids) == 26
    assert len(inject_positions) == 2
    assert inject_positions == [11, 22]


def naive_compute_loss(
    model, tokenizer, projection, config: LossConfig, vectors, labels
):
    """Reproduces `SelfIEModel.compute_loss` directly: full per-position
    logits via the ordinary CausalLM forward, a Python loop over the batch --
    what `SoftPromptLoss`'s sliced-logit batched path is claimed to equal."""
    device = next(model.parameters()).device
    template_ids = tokenizer(SELFIE_TEMPLATE, add_special_tokens=False).input_ids
    reserved_id = tokenizer.convert_tokens_to_ids(RESERVED_TOKEN)
    inject_positions = [i for i, tid in enumerate(template_ids) if tid == reserved_id]
    template_len = len(template_ids)

    embed = model.get_input_embeddings()
    with torch.no_grad():
        template_embeds = embed(torch.tensor(template_ids, device=device).unsqueeze(0))

    vectors = vectors.to(device=device, dtype=torch.float32)
    soft_tokens = projection(vectors).to(dtype=template_embeds.dtype)

    sequences, target_lens, target_id_tensors = [], [], []
    for i, label in enumerate(labels):
        target_ids = tokenizer(
            target_text(label, config), add_special_tokens=False
        ).input_ids
        target_ids_t = torch.tensor(target_ids, device=device)
        with torch.no_grad():
            target_embeds = embed(target_ids_t.unsqueeze(0))
        modified = template_embeds.clone()
        for pos in inject_positions:
            modified[0, pos, :] = soft_tokens[i]
        sequences.append(torch.cat([modified, target_embeds], dim=1))
        target_lens.append(len(target_ids))
        target_id_tensors.append(target_ids_t)

    max_len = max(seq.shape[1] for seq in sequences)
    padded, masks = [], []
    for seq in sequences:
        n = seq.shape[1]
        pad = torch.zeros(1, max_len - n, seq.shape[2], device=device, dtype=seq.dtype)
        padded.append(torch.cat([seq, pad], dim=1) if n < max_len else seq)
        masks.append(
            torch.cat(
                [
                    torch.ones(1, n, dtype=torch.long, device=device),
                    torch.zeros(1, max_len - n, dtype=torch.long, device=device),
                ],
                dim=1,
            )
        )
    batched_embeds = torch.cat(padded, dim=0)
    batched_masks = torch.cat(masks, dim=0)

    with torch.no_grad():
        outputs = model(inputs_embeds=batched_embeds, attention_mask=batched_masks)

    loss_fn = torch.nn.CrossEntropyLoss(
        reduction="none", label_smoothing=config.label_smoothing
    )
    per_seq = []
    for i in range(len(labels)):
        n = target_lens[i]
        target_logits = outputs.logits[i, template_len - 1 : template_len + n - 1]
        token_losses = loss_fn(target_logits, target_id_tensors[i])
        per_seq.append(token_losses.clamp(max=config.max_loss).mean())
    return torch.stack(per_seq).mean()


@pytest.mark.hf_cache
def test_sliced_logit_loss_matches_the_naive_full_logits_reference(
    real_model_and_tokenizer,
):
    """The test the plan calls out as protecting the eventual 1.3662 check:
    our batched, sliced-logit loss must equal the naive per-example, full-
    logits loss upstream's own `compute_loss` computes, within fp32 noise."""
    model, tokenizer = real_model_and_tokenizer
    hidden_size = model.config.hidden_size
    config = LossConfig(
        max_loss=100.0, label_smoothing=0.0, strip_labels=True, eos_token="<|eot_id|>"
    )

    torch.manual_seed(0)
    vectors = torch.randn(8, hidden_size)
    labels = [
        "a short label",
        "a considerably longer description of a topic than the others",
        "x",
        "another label of medium length",
        "y" * 20,
        "the quick brown fox",
        "z",
        "one more label, this one with punctuation!",
    ]
    projection = create_projection_module(
        "scalar_affine",
        dim=hidden_size,
        normalize_input=True,
        device="cpu",
        init_scale=5.0,
    )

    scorer = SoftPromptLoss(model, tokenizer, projection, config)
    sliced_loss, _ = scorer(vectors, labels)
    naive_loss = naive_compute_loss(
        model, tokenizer, projection, config, vectors, labels
    )

    assert torch.isclose(sliced_loss, naive_loss, atol=1e-3)


@pytest.mark.hf_cache
def test_loss_is_invariant_to_batch_composition(real_model_and_tokenizer):
    model, tokenizer = real_model_and_tokenizer
    hidden_size = model.config.hidden_size
    config = LossConfig(
        max_loss=100.0, label_smoothing=0.0, strip_labels=True, eos_token="<|eot_id|>"
    )
    projection = create_projection_module(
        "scalar_affine",
        dim=hidden_size,
        normalize_input=True,
        device="cpu",
        init_scale=5.0,
    )
    scorer = SoftPromptLoss(model, tokenizer, projection, config)

    torch.manual_seed(1)
    vectors = torch.randn(3, hidden_size)
    labels = ["a short label", "a much longer label with several more words in it", "z"]

    individual = torch.stack(
        [scorer(vectors[i : i + 1], labels[i : i + 1])[0] for i in range(3)]
    )
    batched_loss, _ = scorer(vectors, labels)

    assert torch.isclose(batched_loss, individual.mean(), atol=1e-4)
