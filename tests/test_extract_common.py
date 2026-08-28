"""Tests for the batching utilities shared by the pangram and baseline
extraction scripts."""

from adapter_training.extract_common import left_pad, position_ids_from_mask


def test_left_pad_puts_content_at_the_end():
    input_ids, mask = left_pad([[5, 6], [7, 8, 9]], pad_id=0)

    assert input_ids.tolist() == [[0, 5, 6], [7, 8, 9]]
    assert mask.tolist() == [[0, 1, 1], [1, 1, 1]]


def test_position_ids_ignore_left_padding():
    _, mask = left_pad([[5, 6], [7, 8, 9]], pad_id=0)

    # The short row's real tokens must still be positions 0 and 1, not 1 and 2.
    assert position_ids_from_mask(mask).tolist() == [[0, 0, 1], [0, 1, 2]]
