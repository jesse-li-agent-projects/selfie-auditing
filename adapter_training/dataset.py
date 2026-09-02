"""Topic and vector data: reading the upstream topic dataset, reading an
extraction run's output directory, and turning either into training/eval
examples.

Two things every caller must get right, because nothing on disk enforces
them:

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
from typing import Any, cast

import torch
from jaxtyping import Float
from torch import Tensor

DEFAULT_DATASET = "keenanpepper/fifty-thousand-things"
DEFAULT_DATASET_FILE = "wikipedia_vital_articles_level5_dataset.jsonl"


@dataclass(frozen=True)
class Topic:
    """One upstream dataset entry.

    `labels` are the natural-language descriptions the adapter is trained to
    emit; `split` is the dataset's own topic-level train/val assignment, which
    every vector of the topic inherits.
    """

    title: str
    prompt: str
    labels: tuple[str, ...]
    split: str


def load_topics(
    dataset: str, dataset_file: Path | None = None, limit: int | None = None
) -> list[Topic]:
    """Read the upstream topic dataset, from the Hub or a local JSONL copy.

    :param dataset: Hugging Face dataset id, used when `dataset_file` is None
    :param dataset_file: local JSONL to read instead, for a machine with no egress
    :param limit: keep only the first N entries
    :return: topics in dataset order
    """
    rows: Iterable[dict[str, Any]]
    if dataset_file is not None:
        with open(dataset_file) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        from datasets import load_dataset

        # `Dataset.__iter__` is typed as a union wide enough to include lists,
        # which it never yields for a JSONL-backed dataset.
        rows = cast(Iterable[dict[str, Any]], load_dataset(dataset, split="train"))

    topics = []
    for row in rows:
        topics.append(
            Topic(
                title=row["original_title"],
                prompt=row["prompt"],
                labels=tuple(row["labels"]),
                split=row["split"],
            )
        )
        if limit is not None and len(topics) == limit:
            break
    return topics


@dataclass(frozen=True)
class Example:
    """One (vector, label) training item."""

    vector_index: int
    label: str


@dataclass
class VectorStore:
    """One extraction run's vectors, ready to index by `Example.vector_index`."""

    vectors: Float[Tensor, "n_vectors hidden"]  # fp32
    hidden_size: int


@dataclass(frozen=True)
class TopicRecord:
    """One `topics.json` entry -- both extractors write this shape and every
    reader (training, evaluation) reads it back.

    `prompt` and `variant` are extractor-specific (the baseline style's own
    prompt; the pangram style's matched response variant) and unset when read
    back here, where they're unused.
    """

    title: str
    labels: tuple[str, ...]
    split: str
    start: int
    count: int
    prompt: str | None = None
    variant: str | None = None


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
            prompt=entry.get("prompt"),
            variant=entry.get("variant"),
        )
        for entry in raw
    ]


def load_records(
    vectors_dir: Path, restrict_to: Path | None = None
) -> list[TopicRecord]:
    """`load_topic_records(vectors_dir)`, optionally intersected with another
    directory's own titles -- so a downstream comparison (e.g. between the
    baseline and pangram styles) isn't secretly also a topic-population
    difference.

    :param vectors_dir: an extraction output directory
    :param restrict_to: another extraction output directory whose titles to
        intersect with
    """
    records = load_topic_records(vectors_dir)
    if restrict_to is not None:
        other_titles = {record.title for record in load_topic_records(restrict_to)}
        records = restrict_to_titles(records, other_titles)
    return records


def load_vector_store(directory: Path, *, center: bool = True) -> VectorStore:
    """Read `vectors.pt`, cast bf16 -> fp32, and optionally centre.

    Centering subtracts each vector's own position mean: a vector at index
    `i` belonging to a topic with `start` gets `position_means[i - start]`
    subtracted. This is what the trainer and the 1.3662 reproduction check
    need (`center=True`, the default); `center=False` returns raw vectors,
    which is what downstream interpretation-time evaluation uses instead.

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
        # A `[hidden]` means file would slice to a scalar here and broadcast
        # over every dimension, leaving the vectors effectively uncentred --
        # silently, and worth 0.4 nats of val loss. Rejecting it means an
        # extraction directory written before PR #51 fails loudly here and
        # has to be re-extracted.
        if means.ndim != 2 or means.shape[1] != vectors.shape[1]:
            raise ValueError(
                f"{directory / 'position_means.pt'} has shape "
                f"{tuple(means.shape)}; expected [n_positions, "
                f"{vectors.shape[1]}]"
            )
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
    """Keep only records whose title is in `titles`, in their original order.

    `start`/`count` are untouched, so the result still addresses the right
    vectors in its own directory's `vectors.pt`.
    """
    return [record for record in records if record.title in titles]


def pooled_vector_store(
    directory: Path,
    *,
    records: list[TopicRecord] | None = None,
    split: str | None = None,
) -> tuple[VectorStore, list[Example]]:
    """One vector per topic: the mean of that topic's centred vectors.

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
