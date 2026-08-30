"""Forward-pass primitives shared by `extract_baseline_vectors` and
`extract_pangram_vectors`, plus the output format both write.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from adapter_training.dataset import TopicRecord


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
    batch = tokenizer.pad({"input_ids": sequences}, return_tensors="pt").to(device)

    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
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

    `means` is always `[n_positions, hidden]` -- `n_positions` is 1 for the
    baseline style and one per response token for the pangram style;
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
