"""Pangram topic-vector extraction: asks the model to write one fixed
sentence while thinking about one topic, and keeps one vector per sentence
token.

    python -m adapter_training.extract_pangram_vectors \
        --layer 19 --output-dir vectors/pangram_l19

The compliance filter, the forced-response variants and the extraction loop
live in `pangram_extraction`, which `extract_grouped_vectors` shares; this
module is the single-topic prompt, record and CLI around them.

Vectors are written raw; the per-position means are written beside them and
subtracted at load time (`dataset.load_vector_store`).

Compare with `extract_baseline_vectors`, which uses each topic's own dataset
prompt and keeps one vector per topic instead.
"""

import argparse
from pathlib import Path

# Light import: config.py pulls in no heavy dependencies, so --help stays fast.
from config import BASE_MODEL_8B

from adapter_training.dataset import DEFAULT_DATASET


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract pangram-prompt Wikipedia topic vectors."
    )
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument(
        "--output-dir",
        type=lambda value: Path("outputs") / value,
        required=True,
        help="written under outputs/, which is implicitly prepended",
    )
    parser.add_argument("--model", default=BASE_MODEL_8B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N topics"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset id holding the topics",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="read topics from this local JSONL instead of the Hub -- for "
        "machines with no network egress",
    )
    return parser.parse_args()


# Parsed before the heavy imports below, so `--help` costs no torch import.
args = parse_args() if __name__ == "__main__" else None

from adapter_training.dataset import Topic, TopicRecord, load_topics
from adapter_training.extract_common import ExtractionResult
from adapter_training.pangram_extraction import (
    PANGRAM,
    extract_forced_vectors,
    write_forced_outputs,
)
from prompts import PANGRAM_PROMPT_TEMPLATE


def extract_pangram_vectors(
    model,
    tokenizer,
    topics: list[Topic],
    layer: int,
    batch_size: int = 32,
    device: str = "cuda",
    pangram: str = PANGRAM,
    progress: bool = False,
) -> ExtractionResult:
    """`pangram_extraction.extract_forced_vectors` over single topics.

    :param topics: topics to extract, in the order they will be written
    :param pangram: the sentence named in the pangram prompt
    :return: vectors, the surviving topics, the per-position means and the filter failures
    """
    return extract_forced_vectors(
        model,
        tokenizer,
        topics,
        layer=layer,
        instruction_of=lambda topic: PANGRAM_PROMPT_TEMPLATE.format(
            pangram=pangram, topic=topic.title
        ),
        record_of=lambda topic, variant, start, count: TopicRecord(
            title=topic.title,
            prompt=topic.prompt,
            labels=tuple(topic.labels),
            split=topic.split,
            start=start,
            count=count,
            variant=variant,
        ),
        identify=lambda topic: {"title": topic.title},
        batch_size=batch_size,
        device=device,
        progress=progress,
        description="extracting (pangram)",
    )


def write_outputs(
    output_dir: Path,
    result: ExtractionResult,
    layer: int,
    model_name: str,
) -> None:
    """The shared forced-response artefacts, in the single-topic shape."""
    write_forced_outputs(
        output_dir,
        result,
        layer=layer,
        model_name=model_name,
        prompt_style="pangram",
        n_labels=sum(len(record.labels) for record in result.records),
    )


def main(args) -> Path:
    from model_loading import load_base_model, load_tokenizer, resolve_device

    topics = load_topics(args.dataset, args.dataset_file, args.limit)
    print(f"Loaded {len(topics)} topics")

    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, device=args.device, dtype=args.dtype)

    result = extract_pangram_vectors(
        model,
        tokenizer,
        topics,
        layer=args.layer,
        batch_size=args.batch_size,
        device=resolve_device(model),
        progress=True,
    )
    write_outputs(args.output_dir, result, args.layer, args.model)
    print(
        f"Kept {len(result.records)}/{result.n_seen} topics, "
        f"{result.vectors.shape[0]} vectors"
    )
    return args.output_dir


if __name__ == "__main__":
    output_dir = main(args)
    print(
        f"Wrote {output_dir}/{{vectors,position_means}}.pt, topics.json, "
        f"positions.json, filter_report.json"
    )
