"""Shared dataset loading and batched-forward-pass utilities for topic-vector
extraction (plan S5.1, S5.2, step 1). Used by both `extract_pangram_vectors`
and `extract_baseline_vectors`.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from jaxtyping import Int
from torch import Tensor

from extract import build_prompt

DEFAULT_DATASET = "keenanpepper/fifty-thousand-things"
DEFAULT_DATASET_FILE = "wikipedia_vital_articles_level5_dataset.jsonl"


@dataclass(frozen=True)
class Topic:
    """One upstream dataset entry.

    `labels` are the natural-language descriptions the adapter is trained to
    emit; `split` is the dataset's own topic-level train/val assignment, which
    every vector of the topic inherits (plan S5.2).
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


def left_pad(
    sequences: list[list[int]], pad_id: int
) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
    """Stack ragged token sequences with padding on the left.

    Left padding is what makes the response tokens land at fixed negative
    offsets for every example in the batch, so the caller can slice them
    without per-example bookkeeping.

    :param sequences: one token id list per example
    :param pad_id: the pad token id
    :return: the padded ids and their attention mask
    """
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, width - len(sequence) :] = torch.tensor(
            sequence, dtype=torch.long
        )
        attention_mask[row, width - len(sequence) :] = 1
    return input_ids, attention_mask


def position_ids_from_mask(
    attention_mask: Int[Tensor, "batch seq"],
) -> Int[Tensor, "batch seq"]:
    """RoPE positions that ignore left padding.

    A plain forward pass defaults to `arange(seq_len)`, which under left
    padding shifts every real token's rotary position by that row's pad count
    -- so an example's activations would depend on which batch it landed in.
    Deriving positions from the mask keeps batched extraction bit-comparable
    with unbatched.

    :param attention_mask: 1 for real tokens, 0 for padding
    :return: position ids, zero where padded
    """
    return (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)


def run_forward(
    model,
    tokenizer,
    prompts: list[str],
    forced_ids: list[int],
    layer: int,
    device: str,
):
    """One forward pass over a batch of already-built prompts, teacher-forcing
    `forced_ids` after each. Returns the logits (for a compliance check, if
    the caller wants one) and the layer's hidden states, both still batched.

    :param model: the model to run
    :param tokenizer: its tokenizer, configured for left padding
    :param prompts: one formatted (chat-templated) prompt string per example
    :param forced_ids: token ids to force after each prompt; empty for none
    :param layer: transformer layer to read the residual stream at
    :param device: device to run on
    :return: `(logits, hidden_states)`, where `hidden_states` is `layer`'s output
    """
    sequences = [
        tokenizer(prompt, add_special_tokens=False).input_ids + forced_ids
        for prompt in prompts
    ]
    input_ids, attention_mask = left_pad(sequences, tokenizer.pad_token_id)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids_from_mask(attention_mask),
        output_hidden_states=True,
        # The forced tokens sit at the last len(forced_ids) positions, and
        # each is predicted by the logits one position earlier -- so one
        # extra kept position covers the whole check.
        logits_to_keep=len(forced_ids) + 1,
    )
    return outputs.logits, outputs.hidden_states[layer + 1]


def formatted_prompt(tokenizer, user_prompt: str) -> str:
    """Render a user turn with the chat template, no system prompt."""
    return build_prompt(tokenizer, user_prompt, None)
