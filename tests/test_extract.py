import pytest
import torch

from config import Position
from extract import (
    expand_positions,
    find_positions,
    load_hidden_states,
    position_key,
    save_hidden_states,
    user_prompt_span,
)

EOT_ID = 99

USER_PROMPT = "What is the secret word?"

# One id per token string, so a decode() is just a lookup and a join. The
# ids are arbitrary; only the strings they map to matter to the span search.
VOCAB = {
    1: "<|begin_of_text|>",
    10: "<|start_header_id|>",
    11: "user",
    12: "<|end_header_id|>",
    13: "\n\n",
    14: "assistant",
    15: "system",
    EOT_ID: "<|eot_id|>",
    20: "What",
    21: " is",
    22: " the",
    23: " secret",
    24: " word",
    25: "?",
    30: "\n\nWhat",  # template whitespace merged into the first content word
    40: " Please",
    41: " answer",
    50: "The",
    51: " question",
    52: " is",
    53: ":",
    54: " ",
}

PROMPT_IDS = [20, 21, 22, 23, 24, 25]
USER_TURN = [10, 11, 12, 13] + PROMPT_IDS + [EOT_ID]
GENERATION_PROMPT = [10, 14, 12, 13]

# The plain FINETUNED-shaped prompt: no system turn.
BARE_IDS = [1] + USER_TURN + GENERATION_PROMPT


class FakeTokenizer:
    """Implements just enough of TokenizerLike for find_positions() and
    user_prompt_span()."""

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|eot_id|>"
        return EOT_ID

    def decode(self, token_ids, **kwargs) -> str:
        return "".join(VOCAB[int(i)] for i in token_ids)


def span_of(ids: list[int]) -> list[int]:
    return user_prompt_span(FakeTokenizer(), torch.tensor(ids), USER_PROMPT)


def test_find_positions_locates_boundary_and_content_token():
    # bos, start_header, user, end_header, \n\n, content x3, eot_id,
    # start_header, assistant, end_header, \n\n
    ids = [1, 10, 11, 12, 13, 20, 21, 22, EOT_ID, 10, 14, 12, 13]
    input_ids = torch.tensor(ids)

    positions = find_positions(FakeTokenizer(), input_ids)

    assert positions[Position.ASSISTANT_BOUNDARY] == len(ids) - 1
    assert (
        positions[Position.LAST_CONTENT_TOKEN] == 7
    )  # index of the last content token (22)


def test_find_positions_raises_without_eot_id():
    input_ids = torch.tensor([1, 10, 11, 12, 13, 20, 21, 22])
    with pytest.raises(ValueError, match="eot_id"):
        find_positions(FakeTokenizer(), input_ids)


def test_find_positions_uses_last_eot_id_with_a_system_turn():
    # CONTROL/PROMPTED render a system turn before the user turn -- each
    # closed by its own <|eot_id|>. LAST_CONTENT_TOKEN must land on the last
    # *user* content token (30, 31), not the system turn's (which contains
    # the secret word), or the two arms aren't reading a comparable position.
    ids = [
        1,
        10,
        15,
        12,
        13,  # bos, start_header, system, end_header, \n\n
        40,
        41,  # system content
        EOT_ID,  # system turn's eot_id
        10,
        11,
        12,
        13,  # start_header, user, end_header, \n\n
        30,
        31,  # user content
        EOT_ID,  # user turn's eot_id
        10,
        14,
        12,
        13,  # start_header, assistant, end_header, \n\n
    ]
    input_ids = torch.tensor(ids)

    positions = find_positions(FakeTokenizer(), input_ids)

    assert positions[Position.ASSISTANT_BOUNDARY] == len(ids) - 1
    last_eot_index = len(ids) - 1 - 4  # the second EOT_ID, 4 tokens before the end
    assert ids[last_eot_index] == EOT_ID
    assert positions[Position.LAST_CONTENT_TOKEN] == last_eot_index - 1
    assert ids[positions[Position.LAST_CONTENT_TOKEN]] == 31  # last user content token


def test_user_prompt_span_basic():
    # Expected offsets come from the fixture's own id list, never a hardcoded
    # count: the count is a property of one template (plan S2), and pinning it
    # here would re-introduce the assumption the span search exists to remove.
    start = BARE_IDS.index(20)
    assert span_of(BARE_IDS) == list(range(start - len(BARE_IDS), 0))


def test_user_prompt_span_merged_leading_token():
    # First content token is '\n\nWhat' -- the case a standalone tokenization
    # of the prompt gets wrong, and the reason the match is lenient at the front.
    ids = [1, 10, 11, 12, 30, 21, 22, 23, 24, 25, EOT_ID] + GENERATION_PROMPT

    span = span_of(ids)

    assert ids[len(ids) + span[0]] == 30


def test_user_prompt_span_is_minimal():
    span = span_of(BARE_IDS)
    tokenizer = FakeTokenizer()

    assert tokenizer.decode(BARE_IDS[len(BARE_IDS) + span[0] :]).startswith(USER_PROMPT)
    # Dropping the first token leaves a slice that no longer contains the whole
    # prompt -- i.e. every token in the span is necessary.
    assert not tokenizer.decode(BARE_IDS[len(BARE_IDS) + span[1] :]).startswith(
        USER_PROMPT
    )


def test_user_prompt_span_identical_across_arms():
    # The cross-arm comparability invariant: a system turn shifts every
    # absolute index, so only end-relative offsets align the arms.
    system_turn = [10, 15, 12, 13, 50, 51, EOT_ID]
    with_system = [1] + system_turn + USER_TURN + GENERATION_PROMPT

    assert span_of(with_system) == span_of(BARE_IDS)
    assert len(with_system) != len(BARE_IDS)


def test_user_prompt_span_ends_at_the_assistant_boundary():
    # The span's defining property: everything the model sees before it starts
    # speaking, minus the system turn. Its last offset must be the boundary.
    ids = torch.tensor(BARE_IDS)
    span = span_of(BARE_IDS)

    assert span[-1] == -1
    assert (
        len(BARE_IDS) + span[-1]
        == find_positions(FakeTokenizer(), ids)[Position.ASSISTANT_BOUNDARY]
    )


def test_user_prompt_span_raises_when_absent():
    with pytest.raises(ValueError):
        span_of([1] + GENERATION_PROMPT)


def test_user_prompt_span_survives_trailing_user_turn_tokens():
    # The user turn continues after the question. A LAST_CONTENT_TOKEN-anchored
    # implementation would silently take the wrong span here; this one does not.
    ids = [1, 10, 11, 12, 13] + PROMPT_IDS + [40, 41, EOT_ID] + GENERATION_PROMPT

    span = span_of(ids)

    assert ids[len(ids) + span[0]] == 20


def test_user_prompt_span_takes_last_occurrence():
    # A system prompt quoting the question verbatim must not capture the span,
    # for the same reason find_positions() takes the *last* <|eot_id|>.
    quoting_system_turn = [10, 15, 12, 13, 50, 51, 52, 53, 54] + PROMPT_IDS + [EOT_ID]
    ids = [1] + quoting_system_turn + USER_TURN + GENERATION_PROMPT

    span = span_of(ids)

    assert span == span_of(BARE_IDS)
    assert len(ids) + span[0] > len(quoting_system_turn)


def test_expand_positions():
    tokenizer = FakeTokenizer()
    input_ids = torch.tensor(BARE_IDS)
    span = span_of(BARE_IDS)

    expanded = expand_positions(
        tokenizer,
        input_ids,
        USER_PROMPT,
        [Position.LAST_CONTENT_TOKEN, Position.USER_PROMPT_SPAN],
    )

    # LAST_CONTENT_TOKEN keeps its place at the front, and the offset it
    # already covers is not swept a second time by the span.
    last_content_offset = find_positions(tokenizer, input_ids)[
        Position.LAST_CONTENT_TOKEN
    ] - len(BARE_IDS)
    assert expanded[0] is Position.LAST_CONTENT_TOKEN
    assert expanded[1:] == [o for o in span if o != last_content_offset]


def test_expand_positions_does_not_duplicate_the_assistant_boundary():
    expanded = expand_positions(
        FakeTokenizer(),
        torch.tensor(BARE_IDS),
        USER_PROMPT,
        [Position.USER_PROMPT_SPAN, Position.ASSISTANT_BOUNDARY],
    )

    assert expanded == span_of(BARE_IDS)


def test_position_key():
    assert position_key(Position.ASSISTANT_BOUNDARY) == "assistant_boundary"
    assert position_key(-11) == "pos-11"
    assert position_key(-1) == "pos-1"


def test_hidden_states_roundtrip_negative_positions(tmp_path):
    hidden_states = {
        (0, -11): torch.arange(4, dtype=torch.float32),
        (3, Position.ASSISTANT_BOUNDARY): torch.ones(4),
    }
    path = tmp_path / "hidden.safetensors"

    save_hidden_states(path, hidden_states)
    loaded = load_hidden_states(path, [0], [-11])

    assert torch.equal(loaded[(0, -11)], hidden_states[(0, -11)])
    assert not list(tmp_path.glob(".*tmp"))  # the temp file was replaced, not left
