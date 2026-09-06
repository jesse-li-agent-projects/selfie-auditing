"""Grouped topic-vector extraction: asks the model to write one fixed
sentence while thinking about k topics at once (k = 1, 2 or 3), and keeps one
vector per sentence token of the group.

    python -m adapter_training.extract_grouped_vectors \
        --k 3 --rounds 2 --layer 19 --output-dir bg_think_many_l19_k3 \
        --source-topics outputs/bg_think_l19

The topic pool is another extraction run's `topics.json` (`--source-topics`),
not the upstream dataset: those topics already passed the single-topic
compliance filter, so every k shares one topic population and a difference
between the k slices is not secretly a difference in coverage. That directory
is also the k=1 population itself -- the one-topic prompt is byte-identical to
the single-topic one -- so k=1 is never re-extracted.

Records go to `groups.json` (`dataset.GroupRecord`), never `topics.json`.
Everything else -- the compliance filter, the forced-response variants, the
per-position means, the output format -- is shared with
`extract_pangram_vectors` through `pangram_extraction`.
"""

import argparse
from pathlib import Path

# Light import: config.py pulls in no heavy dependencies, so --help stays fast.
from config import BASE_MODEL_8B
from prompts import MAX_BACKGROUND_TOPICS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract multi-topic pangram-prompt Wikipedia topic vectors."
    )
    parser.add_argument(
        "--k", type=int, required=True, help="topics named per prompt (1-3)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="independent disjoint partitions of each split; each round puts "
        "every topic in exactly one group",
    )
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument(
        "--output-dir",
        type=lambda value: Path("outputs") / value,
        required=True,
        help="written under outputs/, which is implicitly prepended",
    )
    parser.add_argument(
        "--source-topics",
        type=Path,
        required=True,
        help="extraction directory whose topics.json supplies the topic pool",
    )
    parser.add_argument("--seed", type=int, default=0, help="seeds the grouping")
    parser.add_argument("--model", default=BASE_MODEL_8B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N groups -- what the plan's Gate 1 probe uses",
    )
    parsed = parser.parse_args()
    if not 1 <= parsed.k <= MAX_BACKGROUND_TOPICS:
        parser.error(f"--k must be 1 to {MAX_BACKGROUND_TOPICS}, got {parsed.k}")
    return parsed


# Parsed before the heavy imports below, so `--help` costs no torch import.
args = parse_args() if __name__ == "__main__" else None

import random
from collections.abc import Sequence

from adapter_training.dataset import GroupRecord, TopicRecord, load_topic_records
from adapter_training.extract_common import ExtractionResult
from adapter_training.pangram_extraction import (
    PANGRAM,
    extract_forced_vectors,
    write_forced_outputs,
)
from prompts import background_topics_prompt


def drop_semicolon_topics(
    topics: Sequence[TopicRecord],
) -> tuple[list[TopicRecord], list[str]]:
    """Drop topics with a literal `;` in any label.

    Composed targets join one label per topic with `"; "`, and set-level
    recall scores a generation by splitting it on `;`, so a label containing
    one cannot be split back into its parts and would be scored as extra
    segments. Dropping the ~10 affected topics costs 0.02% of the corpus and
    is simpler than escaping or changing the separator; see the parent plan's
    open-TODO section.

    :param topics: the candidate topic pool
    :return: the surviving topics, and the dropped titles (for the report)
    """
    kept, dropped = [], []
    for record in topics:
        if any(";" in label for label in record.labels):
            dropped.append(record.title)
        else:
            kept.append(record)
    return kept, dropped


def build_groups(
    topics: Sequence[TopicRecord], k: int, rounds: int, seed: int
) -> list[tuple[TopicRecord, ...]]:
    """Partition each split's topics into groups of k, `rounds` times over.

    One round is a disjoint partition: shuffle the split's topics and cut into
    consecutive groups of k, dropping the fewer-than-k leftovers. So each round
    puts every topic in exactly one group, and `rounds` rounds cover more topic
    *combinations* without repeating any (topic, position) pair more often than
    that.

    Groups never cross the train/val split -- a group mixing the two would leak
    val labels into training -- so each split is partitioned separately, and the
    result is deterministic given `(topics, k, rounds, seed)`.

    The topic order inside a group is the shuffled order and is not sorted: the
    label order is permuted independently when examples are composed, so a fixed
    prompt order here is not a bias the adapter can exploit.

    :param topics: topics carrying a `split` attribute
    :param k: topics per group
    :param rounds: independent partitions per split
    :return: groups, split-major then round-major
    """
    if k < 1:
        raise ValueError(f"build_groups: k must be at least 1, got {k}")
    if rounds < 1:
        raise ValueError(f"build_groups: rounds must be at least 1, got {rounds}")

    groups: list[tuple[TopicRecord, ...]] = []
    for split in sorted({topic.split for topic in topics}):
        pool = [topic for topic in topics if topic.split == split]
        for round_index in range(rounds):
            order = list(pool)
            random.Random(f"{seed}-round-{round_index}").shuffle(order)
            groups.extend(
                tuple(order[start : start + k])
                for start in range(0, len(order) - k + 1, k)
            )
    return groups


def extract_grouped_vectors(
    model,
    tokenizer,
    groups: Sequence[tuple[TopicRecord, ...]],
    layer: int,
    batch_size: int = 32,
    device: str = "cuda",
    pangram: str = PANGRAM,
    progress: bool = False,
) -> ExtractionResult:
    """`pangram_extraction.extract_forced_vectors` over groups of topics.

    :param groups: groups to extract, in the order they will be written
    :param pangram: the sentence named in the prompt
    :return: vectors, the surviving groups, the per-position means and the filter failures
    """
    return extract_forced_vectors(
        model,
        tokenizer,
        groups,
        layer=layer,
        instruction_of=lambda group: background_topics_prompt(
            pangram, [topic.title for topic in group]
        ),
        record_of=lambda group, variant, start, count: GroupRecord(
            titles=tuple(topic.title for topic in group),
            labels_per_topic=tuple(tuple(topic.labels) for topic in group),
            split=group[0].split,
            start=start,
            count=count,
            variant=variant,
        ),
        identify=lambda group: {"titles": [topic.title for topic in group]},
        batch_size=batch_size,
        device=device,
        progress=progress,
        description="extracting (grouped)",
    )


def write_outputs(
    output_dir: Path,
    result: ExtractionResult,
    layer: int,
    model_name: str,
    k: int,
    rounds: int,
    provenance: dict | None = None,
) -> None:
    """The shared forced-response artefacts, in the grouped shape.

    :param provenance: extra `positions.json` fields recording where the topic
        pool came from and what was filtered out of it
    """
    provenance = provenance or {}
    write_forced_outputs(
        output_dir,
        result,
        layer=layer,
        model_name=model_name,
        prompt_style="grouped",
        n_labels=sum(
            len(labels)
            for record in result.records
            for labels in record.labels_per_topic
        ),
        unit="groups",
        records_file="groups.json",
        extra_positions={"k": k, "rounds": rounds, **provenance},
    )


def main(args) -> Path:
    from model_loading import load_base_model, load_tokenizer, resolve_device

    source = load_topic_records(args.source_topics)
    topics, dropped = drop_semicolon_topics(source)
    print(
        f"Loaded {len(source)} topics from {args.source_topics}, "
        f"dropped {len(dropped)} with a ';' in a label"
    )

    groups = build_groups(topics, args.k, args.rounds, args.seed)
    if args.limit is not None:
        groups = groups[: args.limit]
    print(f"Built {len(groups)} groups of {args.k} ({args.rounds} rounds)")

    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, device=args.device, dtype=args.dtype)

    result = extract_grouped_vectors(
        model,
        tokenizer,
        groups,
        layer=args.layer,
        batch_size=args.batch_size,
        device=resolve_device(model),
        progress=True,
    )
    write_outputs(
        args.output_dir,
        result,
        args.layer,
        args.model,
        args.k,
        args.rounds,
        provenance={
            "source_topics": str(args.source_topics),
            "seed": args.seed,
            "dropped_semicolon_topics": dropped,
        },
    )
    print(
        f"Kept {len(result.records)}/{result.n_seen} groups, "
        f"{result.vectors.shape[0]} vectors"
    )
    return args.output_dir


if __name__ == "__main__":
    output_dir = main(args)
    print(
        f"Wrote {output_dir}/{{vectors,position_means}}.pt, groups.json, "
        f"positions.json, filter_report.json"
    )
