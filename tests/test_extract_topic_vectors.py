"""Extraction logic tests (plan step 1).

The fast tests drive `extract_topics` with a fake model and tokenizer, which is
what lets them assert on the filter, the split inheritance and the index
arithmetic without weights. The `hf_cache` tests pin the things only a real
tokenizer/model can answer: how many tokens the pangram is, and that batching
does not change the vectors.
"""

from types import SimpleNamespace

import pytest
import torch

from adapter_training.extract_topic_vectors import (
    DEFAULT_RESPONSE,
    PANGRAM,
    Compliance,
    PromptStyle,
    Topic,
    build_user_prompt,
    check_forced_greedy,
    extract_topics,
    left_pad,
    mismatch_histogram,
    position_ids_from_mask,
    response_token_ids,
    response_variants,
)
from config import DUMMY_BASE_MODEL

HIDDEN = 8
VOCAB = 4096


class FakeTokenizer:
    """Whitespace tokenizer with a Llama-shaped chat template.

    Ids are assigned on first sight, so a token's id is stable within one test
    but means nothing across tests.
    """

    pad_token_id = 0
    eot_id = 3

    def __init__(self):
        self._ids = {"<|eot_id|>": self.eot_id}
        self._tokens = {self.eot_id: "<|eot_id|>"}

    def _id(self, token: str) -> int:
        if token not in self._ids:
            new_id = len(self._ids) + 4
            self._ids[token] = new_id
            self._tokens[new_id] = token
        return self._ids[token]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"USER {messages[-1]['content']} ASSISTANT"

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return SimpleNamespace(input_ids=[self._id(t) for t in text.split()])

    def convert_tokens_to_ids(self, token):
        return self._id(token)

    def decode(self, ids, **kwargs):
        return " ".join(self._tokens.get(int(i), "?") for i in ids)


class FakeTokenizerSplitsPunctuation(FakeTokenizer):
    """Like `FakeTokenizer`, but a trailing full stop is its own token --
    what a real BPE tokenizer does, and what `response_variants` needs to
    find a genuine token-level prefix relationship between the two
    candidates."""

    def __call__(self, text, add_special_tokens=False, **kwargs):
        parts = text.replace(".", " .").split()
        return SimpleNamespace(input_ids=[self._id(t) for t in parts])


class FakeModel:
    """Returns hidden states that encode their own token id, and greedy-compliant
    logits except for topics named in `fail_titles`.

    Encoding the token id into the activation is what lets a test assert *which*
    positions were harvested, not merely how many.
    """

    def __init__(self, tokenizer, fail_titles=(), n_layers=4):
        self.config = SimpleNamespace(hidden_size=HIDDEN, num_hidden_layers=n_layers)
        self.tokenizer = tokenizer
        self.fail_titles = set(fail_titles)
        self.n_layers = n_layers
        self.seen_position_ids = []

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        output_hidden_states,
        logits_to_keep,
    ):
        self.seen_position_ids.append(position_ids.clone())
        batch, seq = input_ids.shape
        base = input_ids.unsqueeze(-1).float().expand(batch, seq, HIDDEN)
        hidden_states = tuple(base + layer for layer in range(self.n_layers + 1))

        logits = torch.zeros(batch, logits_to_keep, VOCAB)
        for row in range(batch):
            text = self.tokenizer.decode(input_ids[row].tolist())
            fails = any(title in text for title in self.fail_titles)
            for kept in range(logits_to_keep):
                position = seq - logits_to_keep + kept
                if position + 1 >= seq:
                    continue
                next_id = int(input_ids[row, position + 1])
                # Diverge in the middle of the sentence, so the test exercises
                # a mismatch index that is neither the first nor the last.
                if fails and kept == 2:
                    next_id = (next_id + 1) % VOCAB
                logits[row, kept, next_id] = 10.0
        return SimpleNamespace(hidden_states=hidden_states, logits=logits)


class NoStopModel(FakeModel):
    """Complies fully with the shorter (no full-stop) forced variant for
    `no_stop_titles`, but diverges from the longer variant exactly at the
    full-stop position -- what the step-0 probe found ~27% of real topics do.
    Otherwise behaves like `FakeModel`.

    Distinguishing the two forced-sequence passes by `logits_to_keep` (12 for
    the 10-sentence-token + eot + lookback candidate, 11 for the
    9-token one) rather than by decoding the input keeps this independent of
    prompt length.
    """

    def __init__(self, tokenizer, no_stop_titles=(), fail_titles=(), n_layers=4):
        super().__init__(tokenizer, fail_titles=fail_titles, n_layers=n_layers)
        self.no_stop_titles = set(no_stop_titles)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        output_hidden_states,
        logits_to_keep,
    ):
        self.seen_position_ids.append(position_ids.clone())
        batch, seq = input_ids.shape
        base = input_ids.unsqueeze(-1).float().expand(batch, seq, HIDDEN)
        hidden_states = tuple(base + layer for layer in range(self.n_layers + 1))

        logits = torch.zeros(batch, logits_to_keep, VOCAB)
        for row in range(batch):
            text = self.tokenizer.decode(input_ids[row].tolist())
            fails = any(title in text for title in self.fail_titles)
            is_no_stop = any(title in text for title in self.no_stop_titles)
            for kept in range(logits_to_keep):
                position = seq - logits_to_keep + kept
                if position + 1 >= seq:
                    continue
                next_id = int(input_ids[row, position + 1])
                if fails and kept == 2:
                    next_id = (next_id + 1) % VOCAB
                elif is_no_stop and logits_to_keep == 12 and kept == 9:
                    # The with-stop pass: refuse the "." token, as if the
                    # model preferred to stop the sentence one token earlier.
                    next_id = self.tokenizer.eot_id
                logits[row, kept, next_id] = 10.0
        return SimpleNamespace(hidden_states=hidden_states, logits=logits)


def topic(title, split="train", n_labels=2):
    return Topic(
        title=title,
        prompt=f"Tell me about {title}.",
        labels=tuple(f"{title} label {i}" for i in range(n_labels)),
        split=split,
    )


def test_pangram_prompt_is_the_requested_wording():
    prompt = build_user_prompt(PromptStyle.PANGRAM, topic("Alpha"))

    assert prompt == (
        'Write "The quick brown fox jumps over the lazy dog". Think about the '
        'topic "Alpha" while writing the sentence. Do not write anything else '
        "or change the words."
    )


def test_baseline_prompt_is_the_topics_own_prompt():
    # Upstream's prompts are hand-written per topic, so the title is not
    # enough to rebuild them.
    entry = Topic("bits", "Tell me about bits (binary digits).", ("a bit",), "train")

    assert build_user_prompt(PromptStyle.BASELINE, entry) == entry.prompt


def test_left_pad_puts_content_at_the_end():
    input_ids, mask = left_pad([[5, 6], [7, 8, 9]], pad_id=0)

    assert input_ids.tolist() == [[0, 5, 6], [7, 8, 9]]
    assert mask.tolist() == [[0, 1, 1], [1, 1, 1]]


def test_position_ids_ignore_left_padding():
    _, mask = left_pad([[5, 6], [7, 8, 9]], pad_id=0)

    # The short row's real tokens must still be positions 0 and 1, not 1 and 2.
    assert position_ids_from_mask(mask).tolist() == [[0, 0, 1], [0, 1, 2]]


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


@pytest.fixture
def fake():
    tokenizer = FakeTokenizer()
    return tokenizer, FakeModel(tokenizer)


def test_pangram_extraction_keeps_one_vector_per_sentence_token(fake):
    tokenizer, model = fake
    sentence_ids, forced_ids = response_token_ids(tokenizer, DEFAULT_RESPONSE)

    result = extract_topics(
        model,
        tokenizer,
        [topic("Alpha"), topic("Bravo")],
        style=PromptStyle.PANGRAM,
        layer=1,
        device="cpu",
    )

    assert len(forced_ids) == len(sentence_ids) + 1  # the eot is forced too
    assert result.vectors.shape == (2 * len(sentence_ids), HIDDEN)
    assert result.position_tokens == [tokenizer.decode([i]) for i in sentence_ids]
    # The fake's hidden_states[L + 1] is (token id + L + 1) in every channel,
    # so this asserts the harvested positions are the sentence tokens -- not
    # the eot, and not the last prompt tokens.
    harvested = result.vectors[: len(sentence_ids), 0].float().tolist()
    assert harvested == [float(i + 2) for i in sentence_ids]


def test_baseline_extraction_keeps_the_last_prompt_token(fake):
    tokenizer, model = fake
    entry = topic("Alpha")
    prompt_ids = tokenizer(f"USER {entry.prompt} ASSISTANT").input_ids

    result = extract_topics(
        model, tokenizer, [entry], style=PromptStyle.BASELINE, layer=1, device="cpu"
    )

    assert result.vectors.shape == (1, HIDDEN)
    assert result.position_tokens == ["last_prompt_token"]
    assert result.vectors[0, 0].float().item() == float(prompt_ids[-1] + 2)


def test_filter_drops_topics_that_would_not_reproduce_the_sentence():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, fail_titles={"Bravo"})
    topics = [topic("Alpha"), topic("Bravo"), topic("Charlie")]

    result = extract_topics(
        model, tokenizer, topics, style=PromptStyle.PANGRAM, layer=1, device="cpu"
    )

    assert [record.title for record in result.records] == ["Alpha", "Charlie"]
    assert result.n_seen == 3
    assert [failure["title"] for failure in result.failures] == ["Bravo"]
    assert result.failures[0]["mismatch_index"] == 2
    # A rejected topic must leave no gap: the survivors' ranges stay contiguous.
    assert [record.start for record in result.records] == [
        0,
        result.records[0].count,
    ]


def test_response_variants_derives_the_no_stop_candidate():
    tokenizer = FakeTokenizerSplitsPunctuation()

    variants = response_variants(tokenizer, DEFAULT_RESPONSE)

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

    variants = response_variants(tokenizer, DEFAULT_RESPONSE)

    assert len(variants) == 1
    assert variants[0][0] == DEFAULT_RESPONSE


def test_pangram_extraction_accepts_the_no_stop_variant():
    tokenizer = FakeTokenizerSplitsPunctuation()
    model = NoStopModel(tokenizer, no_stop_titles={"Bravo"}, fail_titles={"Charlie"})
    topics = [topic("Alpha"), topic("Bravo"), topic("Charlie")]
    sentence_ids, _ = response_token_ids(tokenizer, DEFAULT_RESPONSE)
    no_stop_ids, _ = response_token_ids(tokenizer, PANGRAM)

    result = extract_topics(
        model, tokenizer, topics, style=PromptStyle.PANGRAM, layer=1, device="cpu"
    )

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

    result = extract_topics(
        model, tokenizer, topics, style=PromptStyle.PANGRAM, layer=1, device="cpu"
    )

    # Alpha (with-stop) occupies vectors[0:10]; Bravo (no-stop) only
    # contributes positions 0-8. The final position's mean must therefore
    # equal Alpha's vector at that position exactly, not an average diluted
    # by a Bravo contribution that does not exist.
    alpha_last = result.vectors[last_position].float()
    assert torch.allclose(result.position_means[last_position], alpha_last, atol=1e-4)


def test_records_inherit_the_topics_split_for_every_position(fake):
    tokenizer, model = fake
    topics = [topic("Alpha", split="train"), topic("Bravo", split="val")]

    result = extract_topics(
        model, tokenizer, topics, style=PromptStyle.PANGRAM, layer=1, device="cpu"
    )

    assert [record.split for record in result.records] == ["train", "val"]
    # Every position of a topic lives inside that topic's own range, so a
    # per-vector split can never disagree with the topic's.
    assert all(record.count == len(result.position_tokens) for record in result.records)
    assert result.records[1].start == result.records[0].count


def test_position_means_are_per_position_not_global(fake):
    tokenizer, model = fake
    topics = [topic("Alpha"), topic("Bravo"), topic("Charlie")]

    result = extract_topics(
        model, tokenizer, topics, style=PromptStyle.PANGRAM, layer=1, device="cpu"
    )

    stacked = result.vectors.view(len(topics), -1, HIDDEN).float()
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

    one = extract_topics(
        model,
        tokenizer,
        topics,
        style=PromptStyle.PANGRAM,
        layer=1,
        batch_size=1,
        device="cpu",
    )
    two = extract_topics(
        model,
        tokenizer,
        topics,
        style=PromptStyle.PANGRAM,
        layer=1,
        batch_size=2,
        device="cpu",
    )

    assert torch.equal(one.vectors, two.vectors)


# --- real tokenizer / real model ------------------------------------------


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


@pytest.mark.hf_cache
def test_real_model_batching_matches_unbatched():
    """The padding-aware position ids of `position_ids_from_mask`, end to end.

    Without them the shorter prompt's rotary positions shift by the pad count
    and its vectors depend on what it shared a batch with.

    Runs the baseline style deliberately: the pangram style would filter, and
    whether the 1B smoke model complies for a given topic is not something this
    test should depend on.
    """
    from model_loading import load_base_model, load_tokenizer

    tokenizer = load_tokenizer(DUMMY_BASE_MODEL)
    model = load_base_model(DUMMY_BASE_MODEL, device="cpu", dtype="float32")
    topics = [
        topic("Ada Lovelace"),
        topic("The Peloponnesian War and its immediate causes in Greek history"),
    ]

    one = extract_topics(
        model,
        tokenizer,
        topics,
        style=PromptStyle.BASELINE,
        layer=8,
        batch_size=1,
        device="cpu",
    )
    two = extract_topics(
        model,
        tokenizer,
        topics,
        style=PromptStyle.BASELINE,
        layer=8,
        batch_size=2,
        device="cpu",
    )

    assert one.vectors.shape == (len(topics), model.config.hidden_size)
    assert one.vectors.shape == two.vectors.shape
    similarity = torch.nn.functional.cosine_similarity(
        one.vectors.float(), two.vectors.float(), dim=-1
    )
    assert similarity.min() > 0.999


@pytest.mark.hf_cache
def test_real_model_extraction_has_the_expected_shapes(tmp_path):
    from adapter_training.extract_topic_vectors import write_outputs
    from model_loading import load_base_model, load_tokenizer

    tokenizer = load_tokenizer(DUMMY_BASE_MODEL)
    model = load_base_model(DUMMY_BASE_MODEL, device="cpu", dtype="float32")

    result = extract_topics(
        model,
        tokenizer,
        [topic("Ada Lovelace", split="val")],
        style=PromptStyle.BASELINE,
        layer=8,
        device="cpu",
    )
    write_outputs(
        tmp_path, result, PromptStyle.BASELINE, 8, DUMMY_BASE_MODEL, DEFAULT_RESPONSE
    )

    assert result.vectors.shape == (1, model.config.hidden_size)
    assert result.vectors.dtype == torch.bfloat16
    assert (tmp_path / "vectors.pt").exists()
    assert (tmp_path / "topics.json").exists()
    assert (tmp_path / "positions.json").exists()
    assert (tmp_path / "filter_report.json").exists()
    assert (tmp_path / "position_means.pt").exists()
