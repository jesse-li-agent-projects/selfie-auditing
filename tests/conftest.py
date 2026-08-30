"""Fakes shared across adapter_training's tests."""

import re
from types import SimpleNamespace

import pytest
import torch

from adapter_training.dataset import Topic

HIDDEN = 8
VOCAB = 4096


class FakeCharTokenizer:
    """Tokenizes `<|...|>`-style tags as one atomic token and everything else
    character by character -- enough to isolate `SELFIE_TEMPLATE`'s two
    `RESERVED_TOKEN` slots without a real BPE tokenizer, which a
    whitespace-splitting fake (`FakeTokenizer`, below) cannot do: the
    template embeds `RESERVED_TOKEN` directly against punctuation and other
    tags with no surrounding whitespace.
    """

    TAG = re.compile(r"<\|[^|]*\|>")

    def __init__(self):
        self._ids: dict[str, int] = {}
        self._tokens: dict[int, str] = {}

    def _id(self, token: str) -> int:
        if token not in self._ids:
            new_id = len(self._ids)
            self._ids[token] = new_id
            self._tokens[new_id] = token
        return self._ids[token]

    def _pieces(self, text: str) -> list[str]:
        pieces: list[str] = []
        pos = 0
        for match in self.TAG.finditer(text):
            pieces.extend(text[pos : match.start()])
            pieces.append(match.group())
            pos = match.end()
        pieces.extend(text[pos:])
        return pieces

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return SimpleNamespace(
            input_ids=[self._id(piece) for piece in self._pieces(text)]
        )

    def convert_tokens_to_ids(self, token):
        return self._id(token)

    def decode(self, ids, **kwargs):
        return "".join(self._tokens.get(int(i), "?") for i in ids)


class _FakeBatch:
    """Stands in for the `BatchEncoding` `tokenizer.pad` returns -- just
    enough (`.input_ids`/`.attention_mask`/`.to`) for `run_forward`."""

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class FakeTokenizer:
    """Whitespace tokenizer with a Llama-shaped chat template.

    Ids are assigned on first sight, so a token's id is stable within one test
    but means nothing across tests.
    """

    pad_token_id = 0
    eot_id = 3
    padding_side = "left"

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

    def pad(self, encoded, return_tensors="pt"):
        """Left-pads like the real tokenizer configured with
        `padding_side = "left"` (`run_forward` relies on this)."""
        sequences = encoded["input_ids"]
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), width), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, width - len(sequence) :] = torch.tensor(
                sequence, dtype=torch.long
            )
            attention_mask[row, width - len(sequence) :] = 1
        return _FakeBatch(input_ids, attention_mask)


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

    def __call__(
        self,
        input_ids,
        attention_mask,
        output_hidden_states,
        logits_to_keep,
    ):
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
        output_hidden_states,
        logits_to_keep,
    ):
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


@pytest.fixture
def fake():
    tokenizer = FakeTokenizer()
    return tokenizer, FakeModel(tokenizer)
