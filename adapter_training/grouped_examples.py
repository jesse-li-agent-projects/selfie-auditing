"""Composed labels and the multi-source mixture sampler.

`Example(vector_index, label)` (`adapter_training.dataset`) is unchanged: a
group's composed label is still just a string, joining one label per topic
with `"; "`. Everything below produces the *right list* of them, in the right
mixture, from each source's `groups.json` or `topics.json`.

A **source** is one extraction output directory, named by the caller. Sources
are keyed by name rather than by k (how many topics one extraction prompt
named) because two sources can share a k -- `tell_and_think` mixes two k=1
populations that differ in extraction prompt. k is a property of a source,
read off its records where anything needs it, never an identity.

Nothing here needs real vectors to be tested -- only `GroupRecord`/
`TopicRecord` shapes and a `buckets_of` callable.
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
    VectorRecord,
    VectorStore,
    load_group_records,
    load_topic_records,
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


def _split_by_ratio(total: int, ratio: Mapping[str, int]) -> dict[str, int]:
    """Split `total` by `ratio`, handling the remainder explicitly.

    Each source's share is `total * ratio[name] // sum(ratio.values())`,
    floored; the remainder (at most `len(ratio) - 1`) goes to the sources with
    the largest fractional remainder, largest first, ties broken by `ratio`'s
    own iteration order. Source names have no meaningful sort order, so the
    caller's order is the only stable one.

    :param total: the exact sum the shares must add up to
    :param ratio: relative weight per source, in the caller's order
    :return: one non-negative share per source in `ratio`, summing to `total`
    """
    names = list(ratio)
    weight_sum = sum(ratio.values())
    exact = {name: total * ratio[name] / weight_sum for name in names}
    counts = {name: int(exact[name]) for name in names}
    remainder = total - sum(counts.values())
    order = sorted(
        range(len(names)),
        key=lambda i: (exact[names[i]] - counts[names[i]], -i),
        reverse=True,
    )
    for i in order[:remainder]:
        counts[names[i]] += 1
    return counts


@dataclasses.dataclass
class _MixtureVectors:
    """`VectorStore.vectors` for a multi-source mixture, without ever
    concatenating the underlying tables (the memory-bounded path).

    Each source directory's `vectors.pt` is memory-mapped and kept bf16; the
    fp32 cast and the pooled centring happen at gather time, in
    `__getitem__`, on only the rows actually requested. This keeps resident
    anonymous memory near the batch size rather than the ~26 GiB a
    concatenated fp32 table would cost, since `torch.load(mmap=True)` touches
    page cache, not a private copy, and casting/centring only the gathered
    rows is bit-identical to casting the whole (bf16-backed) table up front.

    Drop-in for a plain vectors tensor: every caller in this codebase indexes
    with a list of global row indices (`store.vectors[[i, j, ...]]`) or a
    single `(index, column)` pair, both of which `__getitem__` below
    supports directly.

    :ivar tensors: one mmap'd, bf16, raw (uncentred) tensor per source, in the
        order `source_of` indexes into
    :ivar source_of: `[n_total]`, which `tensors` entry a global row maps to
    :ivar local_index_of: `[n_total]`, the row within that tensor
    :ivar position_of: `[n_total]`, the position (`row - record.start`) to
        centre that row with, or -1 for a row no surviving record addresses
    :ivar means: `[n_positions, hidden]` fp32, the pooled reference
    """

    tensors: list[torch.Tensor]
    source_of: torch.Tensor
    local_index_of: torch.Tensor
    position_of: torch.Tensor
    means: torch.Tensor

    @property
    def shape(self) -> tuple[int, int]:
        return (self.source_of.shape[0], self.means.shape[1])

    def _gather(self, indices: Sequence[int]) -> torch.Tensor:
        idx = torch.as_tensor(list(indices), dtype=torch.long)
        sources = self.source_of[idx]
        locals_ = self.local_index_of[idx]
        positions = self.position_of[idx]
        if bool((positions < 0).any()):
            raise IndexError(
                "_MixtureVectors: asked for a row no record addresses (it was "
                "dropped by a filter), which has no position to centre against"
            )
        hidden = self.means.shape[1]
        out = torch.empty(len(idx), hidden, dtype=torch.float32)
        for tensor_index in sources.unique().tolist():
            mask = sources == tensor_index
            out[mask] = self.tensors[tensor_index][locals_[mask]].to(torch.float32)
        out -= self.means[positions]
        return out

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)
        single = isinstance(row_key, int)
        rows = self._gather([row_key] if single else row_key)
        result = rows[:, col_key]
        return result[0] if single else result


@dataclasses.dataclass(frozen=True)
class Mixture:
    """One split's sampled mixture, plus the bookkeeping its checks need.

    `source_ranges` slices `examples`; `source_rows` slices the store's global
    row space. The two together are what lets a caller assert that a source's
    examples only ever address that source's own vectors -- a stronger check
    than counting `"; "` separators in the composed labels, which cannot tell
    two k=1 sources apart at all.

    :ivar store: the combined, centred store `examples`' indices address
    :ivar examples: every source's examples, concatenated in source order
    :ivar source_ranges: source -> `(start, end)` into `examples`
    :ivar source_rows: source -> `(start, end)` into the store's rows
    :ivar semicolon_drops: source -> how many topics the `;`-label filter
        dropped (0 for a source whose extractor already applied it)
    """

    store: VectorStore
    examples: list[Example]
    source_ranges: dict[str, tuple[int, int]]
    source_rows: dict[str, tuple[int, int]]
    semicolon_drops: dict[str, int]


def _load_source_records(directory: Path) -> tuple[list[GroupRecord], int]:
    """One source's records, as groups, however the extractor wrote them.

    Dispatches on which file the directory actually has rather than on the
    source's k: a single-topic directory predates `GroupRecord` and writes
    `topics.json`, and more than one source in a mixture can be single-topic.

    The `;`-label filter is applied to `topics.json` sources here, since the
    single-topic extractors predate it. Grouped extractors apply it
    themselves, so a `groups.json` source drops nothing here.

    :param directory: an extraction output directory
    :return: the records as `GroupRecord`s, and the `;`-filter drop count
    """
    if (directory / "groups.json").exists():
        return load_group_records(directory), 0
    topic_records, dropped = drop_semicolon_topics(load_topic_records(directory))
    return [group_record_from_topic(r) for r in topic_records], len(dropped)


def _load_mixture_store(
    directories: Mapping[str, Path],
) -> tuple[
    VectorStore,
    dict[str, list[GroupRecord]],
    dict[str, tuple[int, int]],
    dict[str, int],
]:
    """Load every source's vectors behind one mmap'd, pooled-centred
    `_MixtureVectors`, without concatenating any vector table.

    :param directories: source name -> extraction output directory, in the
        caller's order (which fixes the store's row layout)
    :return: the mixture store; each source's records with `start` offset into
        the store's global row space (so `record.start + position` addresses
        the right row directly); each source's `(start, end)` row range; and
        each source's `;`-filter drop count
    """
    names = list(directories)
    per_source_records: dict[str, list[GroupRecord]] = {}
    semicolon_drops: dict[str, int] = {}
    sources: list[tuple[Path, list[VectorRecord]]] = []
    for name in names:
        records, dropped = _load_source_records(directories[name])
        per_source_records[name] = records
        semicolon_drops[name] = dropped
        sources.append((directories[name], records))

    pooled = pooled_position_means(sources)

    tensors: list[torch.Tensor] = []
    source_chunks: list[torch.Tensor] = []
    local_chunks: list[torch.Tensor] = []
    position_chunks: list[torch.Tensor] = []
    source_rows: dict[str, tuple[int, int]] = {}
    offsets: dict[str, int] = {}
    row_count = 0
    for tensor_index, name in enumerate(names):
        vectors = torch.load(
            directories[name] / "vectors.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        tensors.append(vectors)
        n = vectors.shape[0]

        position_of = torch.full((n,), -1, dtype=torch.long)
        for record in per_source_records[name]:
            position_of[record.start : record.start + record.count] = torch.arange(
                record.count, dtype=torch.long
            )

        source_chunks.append(torch.full((n,), tensor_index, dtype=torch.long))
        local_chunks.append(torch.arange(n, dtype=torch.long))
        position_chunks.append(position_of)

        offsets[name] = row_count
        source_rows[name] = (row_count, row_count + n)
        row_count += n

    mixture_vectors = _MixtureVectors(
        tensors=tensors,
        source_of=torch.cat(source_chunks),
        local_index_of=torch.cat(local_chunks),
        position_of=torch.cat(position_chunks),
        means=pooled,
    )
    offset_records = {
        name: [
            dataclasses.replace(record, start=record.start + offsets[name])
            for record in per_source_records[name]
        ]
        for name in names
    }
    store = VectorStore(vectors=mixture_vectors, hidden_size=pooled.shape[1])
    return store, offset_records, source_rows, semicolon_drops


def build_mixture(
    directories: Mapping[str, Path],
    split: str,
    total_examples: int,
    *,
    ratio: Mapping[str, int],
    seed: int | str,
) -> Mixture:
    """One split's mixture, across several named extraction sources.

    Each source has its own `vectors.pt` and index space; this offsets each
    source's records so `Example.vector_index` addresses the combined table
    directly. See `_load_mixture_store` for the loading/centring; this adds
    the split filter and the per-source sampling.

    `directories` and `ratio` are read in their own iteration order, which
    must agree -- source names have no meaningful sort order, so there is no
    canonical order to fall back on.

    :param directories: source name -> extraction output directory
    :param split: which split's records to sample examples from
    :param total_examples: exact total across all sources, split by `ratio`
    :param ratio: relative example count per source
    :param seed: seeds each source's sampler independently
        (`f"{seed}-{name}-{split}"`)
    :return: the sampled `Mixture`
    :raises ValueError: if `ratio` does not name exactly `directories`' sources
    """
    if set(ratio) != set(directories):
        raise ValueError(
            f"build_mixture: ratio names {sorted(ratio)} but the sources are "
            f"{sorted(directories)}"
        )
    store, offset_records, source_rows, semicolon_drops = _load_mixture_store(
        directories
    )
    counts = _split_by_ratio(total_examples, ratio)

    examples: list[Example] = []
    source_ranges: dict[str, tuple[int, int]] = {}
    for name in directories:
        split_records = [r for r in offset_records[name] if r.split == split]
        start_idx = len(examples)
        examples.extend(
            sample_examples(
                split_records,
                counts[name],
                buckets_of=label_buckets,
                seed=f"{seed}-{name}-{split}",
            )
        )
        source_ranges[name] = (start_idx, len(examples))

    return Mixture(
        store=store,
        examples=examples,
        source_ranges=source_ranges,
        source_rows=source_rows,
        semicolon_drops=semicolon_drops,
    )
