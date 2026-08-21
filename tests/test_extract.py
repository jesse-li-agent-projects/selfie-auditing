import pytest
import torch

from selfie_taboo.config import Position
from selfie_taboo.extract import find_positions

EOT_ID = 99


class FakeTokenizer:
    """Implements just enough of TokenizerLike for find_positions()."""

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|eot_id|>"
        return EOT_ID


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
