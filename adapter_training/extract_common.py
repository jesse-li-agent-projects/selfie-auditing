"""Forward-pass primitives shared by `extract_baseline_vectors` and
`extract_pangram_vectors`, plus the output format both write.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor

from adapter_training.dataset import TopicRecord


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
    """RoPE positions that ignore left padding -- otherwise every real
    token's rotary position shifts by that row's pad count, and an example's
    activations would depend on which batch it landed in.
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


@dataclass
class ExtractionResult:
    """Everything one extraction run writes, held in memory until
    `write_extraction_outputs` runs.

    `means` is one mean vector for the baseline style (`[hidden]`) and one
    per response position for the pangram style (`[n_positions, hidden]`);
    `position_tokens`/`failures` are pangram-only (the baseline style never
    filters, so there's nothing to report).
    """

    vectors: Float[Tensor, "n_vectors hidden"]
    records: list[TopicRecord]
    means: Tensor
    n_seen: int = 0
    position_tokens: list[str] | None = None
    failures: list[dict] = field(default_factory=list)


def write_extraction_outputs(output_dir: Path, result: ExtractionResult) -> None:
    """Write the `vectors.pt`/`position_means.pt`/`topics.json` triple every
    extraction style produces. The caller writes `positions.json` (and, for
    the pangram style, `filter_report.json`) itself -- their content differs
    per style.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.vectors, output_dir / "vectors.pt")
    torch.save(result.means, output_dir / "position_means.pt")
    with open(output_dir / "topics.json", "w") as handle:
        json.dump([asdict(record) for record in result.records], handle)
