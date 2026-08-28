"""Pangram extraction logic tests (plan step 1).

The fast tests drive `extract_pangram_vectors` with a fake model and
tokenizer, which is what lets them assert on the filter, the split
inheritance and the index arithmetic without weights. The `hf_cache` test
pins the thing only a real tokenizer can answer: how many tokens the pangram
is.
"""

import pytest
import torch

from adapter_training.extract_pangram_vectors import (
    DEFAULT_RESPONSE,
    PANGRAM,
    PANGRAM_PROMPT_TEMPLATE,
    Compliance,
    check_forced_greedy,
    extract_pangram_vectors,
    mismatch_histogram,
    response_token_ids,
    response_variants,
)
from config import DUMMY_BASE_MODEL
from conftest import (
    FakeModel,
    FakeTokenizer,
    FakeTokenizerSplitsPunctuation,
    NoStopModel,
    topic,
)

VOCAB = 4096


def test_pangram_prompt_is_the_requested_wording():
    # Verbatim from the user's request (plan S1) -- do not reword.
    prompt = PANGRAM_PROMPT_TEMPLATE.format(pangram=PANGRAM, topic="Alpha")

    assert prompt == (
        'Write "The quick brown fox jumps over the lazy dog". Think about the '
        'topic "Alpha" while writing the sentence. Do not write anything else '
        "or change the words."
    )


def test_compliance_passes_when_every_argmax_matches():
    forced = [11, 12, 13]
    logits = torch.zeros(3, VOCAB)
    for row, token in enumerate(forced):
        logits[row, token] = 1.0

    assert check_forced_greedy(logits, forced, FakeTokenizer()) == Compliance(ok=True)


def test_compliance_reports_the_first_divergence():
    forced = [11, 12, 13]
    logits = torch.zeros(3, VOCAB)
    for row, token in enumerate(forced):
        logits[row, token] = 1.0
    logits[1, 12] = 0.0
    logits[1, 99] = 1.0

    verdict = check_forced_greedy(logits, forced, FakeTokenizer())

    assert not verdict.ok
    assert verdict.mismatch_index == 1


def test_mismatch_histogram_orders_by_position():
    failures = [{"mismatch_index": 9}, {"mismatch_index": 0}, {"mismatch_index": 9}]

    assert mismatch_histogram(failures) == {"0": 1, "9": 2}


def test_pangram_extraction_keeps_one_vector_per_sentence_token(fake):
    tokenizer, model = fake
    sentence_ids, forced_ids = response_token_ids(tokenizer, DEFAULT_RESPONSE)

    result = extract_pangram_vectors(
        model, tokenizer, [topic("Alpha"), topic("Bravo")], layer=1, device="cpu"
    )

    assert len(forced_ids) == len(sentence_ids) + 1  # the eot is forced too
    assert result.vectors.shape == (2 * len(sentence_ids), 8)
    assert result.position_tokens == [tokenizer.decode([i]) for i in sentence_ids]
    # The fake's hidden_states[L + 1] is (token id + L + 1) in every channel,
    # so this asserts the harvested positions are the sentence tokens -- not
    # the eot, and not the last prompt tokens.
    harvested = result.vectors[: len(sentence_ids), 0].float().tolist()
    assert harvested == [float(i + 2) for i in sentence_ids]


def test_filter_drops_topics_that_would_not_reproduce_the_sentence():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, fail_titles={"Bravo"})
    topics = [topic("Alpha"), topic("Bravo"), topic("Charlie")]

    result = extract_pangram_vectors(model, tokenizer, topics, layer=1, device="cpu")

    assert [record.title for record in result.records] == ["Alpha", "Charlie"]
    assert result.n_seen == 3
    assert [failure["title"] for failure in result.failures] == ["Bravo"]
    assert result.failures[0]["mismatch_index"] == 2
    # A rejected topic must leave no gap: the survivors' ranges stay contiguous.
    assert [record.start for record in result.records] == [0, result.records[0].count]


def test_response_variants_derives_the_no_stop_candidate():
    tokenizer = FakeTokenizerSplitsPunctuation()

    variants = response_variants(tokenizer)

    assert len(variants) == 2
    with_stop_text, with_stop_ids, with_stop_forced = variants[0]
    no_stop_text, no_stop_ids, no_stop_forced = variants[1]
    assert with_stop_text == DEFAULT_RESPONSE
    assert no_stop_text == PANGRAM
    assert len(with_stop_ids) == len(no_stop_ids) + 1
    # The no-stop candidate's tokens are a genuine prefix of the with-stop
    # one's -- both point at the same position for the shared words.
    assert no_stop_ids == with_stop_ids[: len(no_stop_ids)]
    assert len(with_stop_forced) == len(with_stop_ids) + 1  # + eot
    assert len(no_stop_forced) == len(no_stop_ids) + 1


def test_response_variants_falls_back_to_one_candidate_without_a_genuine_prefix():
    # FakeTokenizer fuses "dog." into one token, so stripping the string's
    # trailing "." does not recover a prefix of the original ids -- the
    # derivation must not offer a bogus second candidate in that case.
    tokenizer = FakeTokenizer()

    variants = response_variants(tokenizer)

    assert len(variants) == 1
    assert variants[0][0] == DEFAULT_RESPONSE


def test_pangram_extraction_accepts_the_no_stop_variant():
    tokenizer = FakeTokenizerSplitsPunctuation()
    model = NoStopModel(tokenizer, no_stop_titles={"Bravo"}, fail_titles={"Charlie"})
    topics = [topic("Alpha"), topic("Bravo"), topic("Charlie")]
    sentence_ids, _ = response_token_ids(tokenizer, DEFAULT_RESPONSE)
    no_stop_ids, _ = response_token_ids(tokenizer, PANGRAM)

    result = extract_pangram_vectors(model, tokenizer, topics, layer=1, device="cpu")

    assert [r.title for r in result.records] == ["Alpha", "Bravo"]
    alpha, bravo = result.records
    assert alpha.count == len(sentence_ids)  # matched the with-stop variant
    assert alpha.variant == DEFAULT_RESPONSE
    assert bravo.count == len(no_stop_ids)  # one fewer: no full-stop vector
    assert bravo.variant == PANGRAM
    assert bravo.start == alpha.start + alpha.count  # still contiguous
    assert [f["title"] for f in result.failures] == ["Charlie"]


def test_position_means_only_average_positions_that_were_actually_kept():
    tokenizer = FakeTokenizerSplitsPunctuation()
    model = NoStopModel(tokenizer, no_stop_titles={"Bravo"})
    topics = [topic("Alpha"), topic("Bravo")]
    sentence_ids, _ = response_token_ids(tokenizer, DEFAULT_RESPONSE)
    last_position = len(sentence_ids) - 1  # the full-stop token

    result = extract_pangram_vectors(model, tokenizer, topics, layer=1, device="cpu")

    # Alpha (with-stop) occupies vectors[0:10]; Bravo (no-stop) only
    # contributes positions 0-8. The final position's mean must therefore
    # equal Alpha's vector at that position exactly, not an average diluted
    # by a Bravo contribution that does not exist.
    alpha_last = result.vectors[last_position].float()
    assert torch.allclose(result.position_means[last_position], alpha_last, atol=1e-4)


def test_records_inherit_the_topics_split_for_every_position(fake):
    tokenizer, model = fake
    topics = [topic("Alpha", split="train"), topic("Bravo", split="val")]

    result = extract_pangram_vectors(model, tokenizer, topics, layer=1, device="cpu")

    assert [record.split for record in result.records] == ["train", "val"]
    # Every position of a topic lives inside that topic's own range, so a
    # per-vector split can never disagree with the topic's.
    assert all(record.count == len(result.position_tokens) for record in result.records)
    assert result.records[1].start == result.records[0].count


def test_position_means_are_per_position_not_global(fake):
    tokenizer, model = fake
    topics = [topic("Alpha"), topic("Bravo"), topic("Charlie")]

    result = extract_pangram_vectors(model, tokenizer, topics, layer=1, device="cpu")

    stacked = result.vectors.view(len(topics), -1, 8).float()
    assert torch.allclose(result.position_means, stacked.mean(dim=0), atol=1e-2)
    # The pangram tokens differ from each other, so a global mean would not
    # have cancelled "which word of the sentence this is".
    assert not torch.allclose(
        result.position_means, result.position_means.mean(dim=0, keepdim=True)
    )


def test_batching_does_not_change_which_positions_are_harvested(fake):
    tokenizer, model = fake
    # Prompt lengths differ, so the second batch pads and the first does not.
    topics = [topic("Alpha"), topic("Bravo with a much longer title than the first")]

    one = extract_pangram_vectors(
        model, tokenizer, topics, layer=1, batch_size=1, device="cpu"
    )
    two = extract_pangram_vectors(
        model, tokenizer, topics, layer=1, batch_size=2, device="cpu"
    )

    assert torch.equal(one.vectors, two.vectors)


# --- real tokenizer --------------------------------------------------------


@pytest.mark.hf_cache
def test_pangram_is_ten_tokens():
    """Pins plan S4.2b's count, which the whole cost model is built on."""
    from model_loading import load_tokenizer

    tokenizer = load_tokenizer(DUMMY_BASE_MODEL)
    sentence_ids, forced_ids = response_token_ids(tokenizer, DEFAULT_RESPONSE)

    assert [tokenizer.decode([i]) for i in sentence_ids] == [
        "The",
        " quick",
        " brown",
        " fox",
        " jumps",
        " over",
        " the",
        " lazy",
        " dog",
        ".",
    ]
    assert len(forced_ids) == 11
    assert tokenizer.decode([forced_ids[-1]]) == "<|eot_id|>"
    assert PANGRAM in DEFAULT_RESPONSE
