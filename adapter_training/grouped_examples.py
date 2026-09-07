"""Composed labels and the 1:2:3 mixture sampler (bg_think_many step 3).

`Example(vector_index, label)` (`adapter_training.dataset`) is unchanged: a
group's composed label is still just a string, joining one label per topic
with `"; "`. Everything below produces the *right list* of them, in the right
k=1:2:3 mixture, from step 2's `groups.json` (and, for k=1, the reused
`outputs/bg_think_l19` topic records). Nothing here needs step 2's vectors to
be tested -- only `GroupRecord`/`TopicRecord` shapes and a `buckets_of`
callable.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path

import torch

from adapter_training.dataset import (
    Example,
    GroupRecord,
    TopicRecord,
    VectorStore,
    load_group_records,
    load_topic_records,
    load_vector_store,
    pooled_position_means,
)
from adapter_training.extract_grouped_vectors import drop_semicolon_topics
from adapter_training.label_complexity import label_buckets


def compose_label(labels: Sequence[str]) -> str:
    """Join one label per topic into the adapter's target string (D2).

    Plain `"; "` join, no trailing separator or punctuation -- `loss.py`'s
    `target_text` already appends the closing quote and eos token, and any
    punctuation added here would change the target relative to `bg_think`'s
    k=1 labels, which must stay comparable (parent plan D1).

    :param labels: one label per topic, already in the order to emit them
    :return: the composed target string
    """
    return "; ".join(labels)


def group_record_from_topic(record: TopicRecord) -> GroupRecord:
    """Adapt a single-topic record into a one-topic group.

    This is the bridge that lets `outputs/bg_think_l19` (written by the
    single-topic extractor, before `GroupRecord` existed) serve as the k=1
    population directly, per the parent plan's D1.

    :param record: a `topics.json` entry
    :return: the equivalent one-title `GroupRecord`
    """
    return GroupRecord(
        titles=(record.title,),
        labels_per_topic=(record.labels,),
        split=record.split,
        start=record.start,
        count=record.count,
        variant=record.variant,
    )


def _nearest_nonempty_bucket(
    buckets: Sequence[Sequence[str]], index: int
) -> tuple[Sequence[str], bool]:
    """The bucket at `index`, or the nearest non-empty one if it is empty.

    Ties (equal distance on either side) prefer the lower index, which is
    arbitrary but deterministic given a fixed `rng`.

    :param buckets: one topic's buckets, least to most detailed
    :param index: the drawn bucket index
    :return: the bucket to draw from, and whether a fallback was needed
    :raises ValueError: if every bucket is empty
    """
    if buckets[index]:
        return buckets[index], False
    for distance in range(1, len(buckets)):
        for candidate in (index - distance, index + distance):
            if 0 <= candidate < len(buckets) and buckets[candidate]:
                return buckets[candidate], True
    raise ValueError("_nearest_nonempty_bucket: every bucket is empty")


def sample_examples(
    records: Sequence[GroupRecord],
    n_examples: int,
    *,
    buckets_of: Callable[[Sequence[str]], list[list[str]]] = label_buckets,
    seed: int | str,
    stats: MutableMapping[str, int] | None = None,
) -> list[Example]:
    """Draw `n_examples` composed-label examples from `records` (D4, D6).

    One draw: pick a `(record, position)` uniformly from every valid pair,
    pick a complexity bucket index uniformly, pick one label per topic from
    that bucket (falling back to the topic's nearest non-empty bucket), and
    compose a uniformly random permutation of the chosen labels. A draw that
    repeats an already-drawn `(vector_index, composed label)` pair is
    rejected and redrawn.

    :param records: the group pool to draw from -- callers pass one split's
        records, already offset into a shared `VectorStore` if applicable
    :param n_examples: exact number of examples to return
    :param buckets_of: buckets one topic's labels by complexity, least
        detailed first (`label_complexity.label_buckets`, by default)
    :param seed: seeds a private `random.Random`; never mutates global state
    :param stats: if given, filled with `duplicate_rejections`,
        `empty_bucket_fallbacks` and `attempts` for a findings note
    :return: exactly `n_examples` examples, in draw order (unshuffled --
        that is the trainer's job)
    :raises ValueError: if `records` offers no vectors, or if `n_examples`
        exceeds what the pool can supply without repeats
    """
    pairs = [
        (record_index, position)
        for record_index, record in enumerate(records)
        for position in range(record.count)
    ]
    if not pairs:
        raise ValueError("sample_examples: no records have any vectors")

    rng = random.Random(seed)
    seen: set[tuple[int, str]] = set()
    examples: list[Example] = []
    duplicate_rejections = 0
    empty_bucket_fallbacks = 0
    max_attempts = max(n_examples * 50, 1000)
    attempts = 0

    while len(examples) < n_examples:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"sample_examples: could not draw {n_examples} unique examples "
                f"from a pool of {len(pairs)} (record, position) pairs after "
                f"{attempts - 1} attempts ({len(examples)} drawn so far) -- the "
                "requested count likely exceeds what this pool can supply "
                "without repeated (vector, label) pairs"
            )
        record_index, position = pairs[rng.randrange(len(pairs))]
        record = records[record_index]
        vector_index = record.start + position

        first_buckets = buckets_of(record.labels_per_topic[0])
        bucket_index = rng.randrange(len(first_buckets))

        chosen = []
        for topic_labels in record.labels_per_topic:
            buckets = buckets_of(topic_labels)
            bucket, used_fallback = _nearest_nonempty_bucket(buckets, bucket_index)
            if used_fallback:
                empty_bucket_fallbacks += 1
            chosen.append(bucket[rng.randrange(len(bucket))])

        order = list(range(len(chosen)))
        rng.shuffle(order)
        composed = compose_label([chosen[i] for i in order])

        key = (vector_index, composed)
        if key in seen:
            duplicate_rejections += 1
            continue
        seen.add(key)
        examples.append(Example(vector_index=vector_index, label=composed))

    if stats is not None:
        stats["duplicate_rejections"] = duplicate_rejections
        stats["empty_bucket_fallbacks"] = empty_bucket_fallbacks
        stats["attempts"] = attempts
    return examples


def _split_by_ratio(total: int, ratio: Mapping[int, int]) -> dict[int, int]:
    """Split `total` by `ratio`, handling the remainder explicitly.

    Each key's share is `total * ratio[k] // sum(ratio.values())`, floored;
    the remainder (at most `len(ratio) - 1`) goes to the keys with the
    largest fractional remainder, largest first, ties broken by key order.

    :param total: the exact sum the shares must add up to
    :param ratio: relative weight per key
    :return: one non-negative share per key in `ratio`, summing to `total`
    """
    keys = sorted(ratio)
    weight_sum = sum(ratio[k] for k in keys)
    exact = {k: total * ratio[k] / weight_sum for k in keys}
    counts = {k: int(exact[k]) for k in keys}
    remainder = total - sum(counts.values())
    order = sorted(keys, key=lambda k: (exact[k] - counts[k], -k), reverse=True)
    for k in order[:remainder]:
        counts[k] += 1
    return counts


def _load_mixture_store(
    directories: Mapping[int, Path],
) -> tuple[VectorStore, dict[int, list[GroupRecord]]]:
    """Load, centre against one pooled reference (D13) and concatenate every
    k's vectors into one `VectorStore`.

    k=1's directory is read as `TopicRecord`s (`outputs/bg_think_l19`
    predates `GroupRecord`) and adapted with `group_record_from_topic`; the
    `;`-label filter (parent plan §8) is applied to it here too, since that
    directory predates step 2's own filter.

    :param directories: k -> extraction output directory
    :return: the concatenated store, and each k's records with `start`
        offset into the concatenated store (so `record.start + position`
        addresses the right row directly)
    """
    ks = sorted(directories)
    per_k_records: dict[int, list[GroupRecord]] = {}
    sources: list[tuple[Path, list[GroupRecord]]] = []
    for k in ks:
        directory = directories[k]
        if k == 1:
            topic_records, _dropped = drop_semicolon_topics(
                load_topic_records(directory)
            )
            records = [group_record_from_topic(r) for r in topic_records]
        else:
            records = load_group_records(directory)
        per_k_records[k] = records
        sources.append((directory, records))

    pooled = pooled_position_means(sources)

    vector_chunks = []
    offsets: dict[int, int] = {}
    row_count = 0
    for k in ks:
        store = load_vector_store(
            directories[k], center=True, records=per_k_records[k], means=pooled
        )
        offsets[k] = row_count
        vector_chunks.append(store.vectors)
        row_count += store.vectors.shape[0]

    combined = torch.cat(vector_chunks, dim=0)
    offset_records = {
        k: [
            dataclasses.replace(record, start=record.start + offsets[k])
            for record in per_k_records[k]
        ]
        for k in ks
    }
    return VectorStore(vectors=combined, hidden_size=combined.shape[1]), offset_records


def build_mixture(
    directories: Mapping[int, Path],
    split: str,
    total_examples: int,
    *,
    ratio: Mapping[int, int] = {1: 1, 2: 2, 3: 3},
    seed: int | str,
) -> tuple[VectorStore, list[Example], dict[int, tuple[int, int]]]:
    """One split's 1:2:3 mixture, across the three k extraction directories.

    Each directory has its own `vectors.pt` and index space; this
    concatenates the three (centred) tables and offsets each directory's
    records so `Example.vector_index` addresses the combined table directly.
    See `_load_mixture_store` for the loading/centring; this adds the split
    filter and the per-k sampling.

    :param directories: k -> extraction output directory (k=1, 2, 3)
    :param split: which split's records to sample examples from
    :param total_examples: exact total across all k, split by `ratio`
    :param ratio: relative example count per k (D6's 1:2:3 by default)
    :param seed: seeds each k's sampler independently (`f"{seed}-k{k}-{split}"`)
    :return: the combined store, the examples (grouped by k, in k order),
        and each k's `(start, end)` index range into the returned examples --
        for per-k validation loss (parent plan D12)
    """
    store, offset_records = _load_mixture_store(directories)
    ks = sorted(directories)
    counts = _split_by_ratio(total_examples, ratio)

    examples: list[Example] = []
    k_ranges: dict[int, tuple[int, int]] = {}
    for k in ks:
        split_records = [r for r in offset_records[k] if r.split == split]
        start_idx = len(examples)
        examples.extend(
            sample_examples(
                split_records,
                counts[k],
                buckets_of=label_buckets,
                seed=f"{seed}-k{k}-{split}",
            )
        )
        k_ranges[k] = (start_idx, len(examples))

    return store, examples, k_ranges
