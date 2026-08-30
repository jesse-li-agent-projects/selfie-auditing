"""Baseline topic-vector extraction: upstream's own extraction, reproduced --
each topic's own conversational prompt, one vector read at the last prompt
token.

    python -m adapter_training.extract_baseline_vectors \
        --layer 19 --output-dir vectors/baseline_l19

Compare with `extract_pangram_vectors`, which asks the model to write a fixed
sentence instead and keeps one vector per response token.
"""

import argparse
import json
from pathlib import Path

# Light import: config.py pulls in no heavy dependencies, so --help stays fast.
from config import BASE_MODEL_8B

from adapter_training.dataset import DEFAULT_DATASET


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract baseline (upstream-style) Wikipedia topic vectors."
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

import torch

from adapter_training.dataset import Topic, TopicRecord, load_topics
from adapter_training.extract_common import (
    ExtractionResult,
    run_forward,
    write_extraction_outputs,
)
from extract import build_prompt


@torch.no_grad()
def extract_baseline_vectors(
    model,
    tokenizer,
    topics: list[Topic],
    layer: int,
    batch_size: int = 32,
    device: str = "cuda",
    progress: bool = False,
) -> ExtractionResult:
    """Run the forward passes and harvest one vector per topic.

    Each topic uses its own hand-written prompt from the dataset
    (`Tell me about bits (binary digits).`, not a mechanical
    `Tell me about {title}.`) -- that is what upstream trained on. The kept
    vector is `hidden_states[L + 1]` (the output of transformer layer L) at
    the last prompt token.

    :param model: the base model to read activations from
    :param tokenizer: its tokenizer, configured for left padding
    :param topics: topics to extract, in the order they will be written
    :param layer: transformer layer to read the residual stream at
    :param batch_size: topics per forward pass
    :param device: device to run on
    :param progress: show a tqdm bar
    :return: one record and one vector per topic, and their mean
    """
    hidden_size = model.config.hidden_size
    vectors = torch.empty(len(topics), hidden_size, dtype=torch.bfloat16, device="cpu")
    # Accumulated in float64 as the run goes rather than by casting the whole
    # vector table at the end.
    total = torch.zeros(hidden_size, dtype=torch.float64)
    records: list[TopicRecord] = []

    batches = range(0, len(topics), batch_size)
    if progress:
        from tqdm import tqdm

        batches = tqdm(batches, desc="extracting (baseline)")

    for start in batches:
        batch = topics[start : start + batch_size]
        prompts = [build_prompt(tokenizer, topic.prompt, None) for topic in batch]
        _, hidden = run_forward(model, tokenizer, prompts, [], layer, device)

        kept = hidden[:, -1, :].to(dtype=torch.bfloat16, device="cpu")
        vectors[start : start + len(batch)] = kept
        total += kept.double().sum(dim=0)

        for row, topic in enumerate(batch):
            records.append(
                TopicRecord(
                    title=topic.title,
                    prompt=topic.prompt,
                    labels=tuple(topic.labels),
                    split=topic.split,
                    start=start + row,
                    count=1,
                )
            )

    mean = (total / max(len(topics), 1)).float()
    return ExtractionResult(
        vectors=vectors, records=records, means=mean, n_seen=len(topics)
    )


def write_outputs(
    output_dir: Path,
    result: ExtractionResult,
    layer: int,
    model_name: str,
) -> None:
    """`write_extraction_outputs` plus `positions.json`. There is no
    `filter_report.json` -- the baseline style never filters.
    """
    write_extraction_outputs(output_dir, result)
    with open(output_dir / "positions.json", "w") as handle:
        json.dump(
            {
                "prompt_style": "baseline",
                "layer": layer,
                "model": model_name,
                "n_positions": 1,
                "position_tokens": ["last_prompt_token"],
                "n_topics": len(result.records),
                "n_vectors": int(result.vectors.shape[0]),
                "hidden_size": int(result.vectors.shape[1]),
                "means_file": "position_means.pt",
            },
            handle,
            indent=2,
        )


def main(args) -> Path:
    from model_loading import load_base_model, load_tokenizer, resolve_device

    topics = load_topics(args.dataset, args.dataset_file, args.limit)
    print(f"Loaded {len(topics)} topics")

    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, device=args.device, dtype=args.dtype)

    result = extract_baseline_vectors(
        model,
        tokenizer,
        topics,
        layer=args.layer,
        batch_size=args.batch_size,
        device=resolve_device(model),
        progress=True,
    )
    write_outputs(args.output_dir, result, args.layer, args.model)
    print(f"Wrote {len(result.records)} vectors")
    return args.output_dir


if __name__ == "__main__":
    output_dir = main(args)
    print(
        f"Wrote {output_dir}/{{vectors,position_means}}.pt, topics.json, positions.json"
    )
