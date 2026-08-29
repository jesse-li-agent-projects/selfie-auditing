"""Turns an extraction output directory (`extract_pangram_vectors.py` or
`extract_baseline_vectors.py`) into training/eval examples (plan step 2a S1).

Two things every caller must get right, because nothing on disk enforces
them (parent plan S9.2, "Means are written, not applied"):

- A topic's vectors are `vectors[start : start + count]`; its position index
  is `i - start`. `count` is not constant across topics in the pangram style.
- Vectors on disk are raw. Centering happens here, once, so no other code
  path can silently train on uncentred vectors -- `load_vector_store` is the
  only supported way to read `vectors.pt`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass(frozen=True)
class Example:
    """One (vector, label) training item -- the parent plan's 'example'."""

    vector_index: int
    label: str


@dataclass
class VectorStore:
    """One extraction run's vectors, ready to index by `Example.vector_index`."""

    vectors: Float[Tensor, "n_vectors hidden"]  # fp32
    hidden_size: int


@dataclass(frozen=True)
class TopicRecord:
    """One `topics.json` entry, read back for training.

    Mirrors the extractors' own `TopicRecord` dataclasses but is decoupled
    from either of them -- this module only reads the fields both styles
    write (`prompt` and pangram's `variant` are ignored).
    """

    title: str
    labels: tuple[str, ...]
    split: str
    start: int
    count: int


def load_topic_records(directory: Path) -> list[TopicRecord]:
    """Read `topics.json`, in the order the extractor wrote it.

    :param directory: an extraction output directory
    :return: one record per surviving topic
    """
    with open(directory / "topics.json") as handle:
        raw = json.load(handle)
    return [
        TopicRecord(
            title=entry["title"],
            labels=tuple(entry["labels"]),
            split=entry["split"],
            start=entry["start"],
            count=entry["count"],
        )
        for entry in raw
    ]


def load_vector_store(directory: Path, *, center: bool = True) -> VectorStore:
    """Read `vectors.pt`, cast bf16 -> fp32, and optionally centre.

    Centering subtracts each vector's own position mean: a vector at index
    `i` belonging to a topic with `start` gets `position_means[i - start]`
    subtracted (parent plan S5.3). This is what the trainer and the 1.3662
    reproduction check need (`center=True`, the default); `center=False`
    returns raw vectors, which is what downstream interpretation-time
    evaluation uses instead (parent plan S5.3: train centred, interpret raw).

    :param directory: an extraction output directory
    :param center: subtract per-position means (see above)
    :return: the vectors, fp32, indexed exactly as `vectors.pt` is
    """
    vectors = torch.load(
        directory / "vectors.pt", map_location="cpu", weights_only=True
    ).to(torch.float32)
    if center:
        means = torch.load(
            directory / "position_means.pt", map_location="cpu", weights_only=True
        ).to(torch.float32)
        for record in load_topic_records(directory):
            n = record.count
            vectors[record.start : record.start + n] -= means[:n]
    return VectorStore(vectors=vectors, hidden_size=vectors.shape[1])


def examples_from_records(records: Iterable[TopicRecord], split: str) -> list[Example]:
    """Flatten topic records of one split into (vector, label) examples.

    A `count`-vector topic with `len(labels)` labels yields `count *
    len(labels)` examples: every label against every one of the topic's
    vectors. Order is deterministic (record order, then label order, then
    position) -- the shuffling belongs to the sampler, not here.

    :param records: topic records, e.g. from `load_topic_records`
    :param split: keep only topics with this split
    :return: examples in deterministic order
    """
    examples = []
    for record in records:
        if record.split != split:
            continue
        for label in record.labels:
            for offset in range(record.count):
                examples.append(
                    Example(vector_index=record.start + offset, label=label)
                )
    return examples


def load_examples(directory: Path, split: str) -> list[Example]:
    """`examples_from_records(load_topic_records(directory), split)`."""
    return examples_from_records(load_topic_records(directory), split)


def restrict_to_titles(
    records: list[TopicRecord], titles: set[str]
) -> list[TopicRecord]:
    """Keep only records whose title is in `titles`.

    The baseline style filters no topics (49,637) while the pangram style
    keeps only compliant ones; comparing arms without this would risk an arm
    difference that is really a topic-population difference (parent plan
    S9.2). `start`/`count` are untouched, so the result still addresses the
    right vectors in its own directory's `vectors.pt`.

    :param records: topic records to filter
    :param titles: titles to keep (typically another directory's own titles)
    :return: the intersection, in `records`' original order
    """
    return [record for record in records if record.title in titles]


def pooled_vector_store(
    directory: Path,
    *,
    records: list[TopicRecord] | None = None,
    split: str | None = None,
) -> tuple[VectorStore, list[Example]]:
    """Arm C: one vector per topic, the mean of that topic's centred vectors.

    Always centres (pooling raw vectors before centering would average across
    positions before the per-position mean is subtracted, which is not the
    same operation). One example per (topic, label), addressing the pooled
    vector's own index -- unrelated to the raw `vectors.pt` indices.

    :param directory: an extraction output directory
    :param records: pool these records instead of `directory`'s own
        `topics.json` -- how this composes with `restrict_to_titles`
    :param split: keep only this split's topics; both splits if None
    :return: the pooled store, and one example per (topic, label)
    """
    store = load_vector_store(directory, center=True)
    all_records = records if records is not None else load_topic_records(directory)
    if split is not None:
        all_records = [record for record in all_records if record.split == split]

    pooled_vectors = []
    examples = []
    for new_index, record in enumerate(all_records):
        pooled_vectors.append(
            store.vectors[record.start : record.start + record.count].mean(dim=0)
        )
        examples.extend(
            Example(vector_index=new_index, label=label) for label in record.labels
        )
    pooled = (
        torch.stack(pooled_vectors)
        if pooled_vectors
        else torch.empty(0, store.hidden_size)
    )
    return VectorStore(vectors=pooled, hidden_size=store.hidden_size), examples
