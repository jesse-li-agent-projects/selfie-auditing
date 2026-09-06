"""Bucket one topic's labels by detail level, using length as a proxy.

The upstream Wikipedia dataset carries no complexity metadata: it merged
several generation runs and deduplicated with ``dict.fromkeys``, so labels
arrive in no recoverable order (see
``plans/bg_think_many/notes/step1_complexity.md`` for the measurements behind
this choice). Within-topic length terciles (rule A there) transfer well enough
across topics to use directly, with no global calibration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def label_buckets(
    labels: Sequence[str],
    n_buckets: int = 3,
    length_of: Callable[[str], int] = len,
) -> list[list[str]]:
    """Partition one topic's labels into complexity buckets, least detailed
    first.

    Labels are sorted by ``length_of`` (ties broken by the label text itself,
    so reordering the input cannot change the result) and split into
    ``n_buckets`` contiguous, near-equal groups.

    :param labels: one topic's labels, in any order
    :param n_buckets: number of complexity buckets to produce
    :param length_of: a label's length proxy; defaults to character count,
        but callers should inject a tokenizer-based length for training use
    :return: ``n_buckets`` lists of labels, ordered least to most detailed;
        every input label appears in exactly one output list
    """
    ordered = sorted(labels, key=lambda label: (length_of(label), label))
    n = len(ordered)
    base, remainder = divmod(n, n_buckets)
    buckets: list[list[str]] = []
    start = 0
    for i in range(n_buckets):
        size = base + (1 if i < remainder else 0)
        buckets.append(ordered[start : start + size])
        start += size
    return buckets
