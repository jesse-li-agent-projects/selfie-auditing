"""Adapter loading, injection, generation (plan S3, S4.3).

Mechanism (verbatim from the adapter's own reference script,
research_notes_selfie_mechanism.md S1.4): injection happens at the embedding
layer via `inputs_embeds`, never a forward hook, and is decoupled from
whatever layer the hidden state was extracted from.

Raw hidden states are injected directly, with no mean-subtraction
preprocessing -- see plan S4.4: the reference repo's own bridge-entity layer
sweep (evals/bridge_entity/run_selfie_bridge_extraction.py) does the same,
never subtracting a mean at any layer.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor

# Verbatim from the adapter repo's reference script (research_notes S1.4). Do
# not reformat -- the adapter was trained against this exact string.
SELFIE_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    'What is the meaning of "<|reserved_special_token_0|>"?'
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    'The meaning of "<|reserved_special_token_0|>" is "'
)

# experimental alternate
# SELFIE_TEMPLATE = (
#     "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
#     'What is the hidden meaning of "<|reserved_special_token_0|>"?'
#     "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
#     'The hidden meaning of "<|reserved_special_token_0|>" is "'
# )
RESERVED_TOKEN = "<|reserved_special_token_0|>"


class Adapter(Protocol):
    """Interface generate_interpretations() needs -- satisfied by
    `selfie_adapters.SelfIEAdapter`."""

    def transform(self, vector: Float[Tensor, "hidden"]) -> Float[Tensor, "hidden"]: ...


CellKey = TypeVar("CellKey")


@torch.no_grad()
def generate_interpretations_batch(
    model,
    tokenizer,
    adapter: Adapter,
    hidden_vectors: dict[CellKey, Float[Tensor, "hidden"]],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    batch_size: int = 200,
) -> dict[CellKey, list[str]]:
    """Sample n_samples interpretations for each of several hidden states at once.

    A forward pass only depends on which soft token each row injects -- so
    rows from different cells (any mix of `hidden_vectors`' keys) can share
    one `generate()` call so long as they'd otherwise use the same template,
    LoRA state, and generation settings. That's what lets `batch_size` grow
    past a single cell's `n_samples`: the pool to draw a batch from is every
    requested cell's samples, not one cell's.

    Rows are chunked in a fixed order (`hidden_vectors`' own iteration order,
    each key repeated `n_samples` times), so a given `(hidden_vectors,
    batch_size)` pair always produces the same batch boundaries -- required
    for the seeded RNG stream a caller sets up before this call to reproduce.

    :param hidden_vectors: cells to interpret, keyed by anything hashable
    :param n_samples: generations per cell
    :param batch_size: rows per forward pass, pooled across cells
    :return: each key's `n_samples` generations, in the order they were drawn
    """
    template_tokens = tokenizer(
        SELFIE_TEMPLATE, return_tensors="pt", add_special_tokens=False
    ).to(device)
    reserved_token_id = tokenizer.convert_tokens_to_ids(RESERVED_TOKEN)
    inject_positions = [
        i
        for i, tid in enumerate(template_tokens.input_ids[0])
        if tid.item() == reserved_token_id
    ]
    if not inject_positions:
        raise ValueError(f"{RESERVED_TOKEN!r} not found in tokenized SELFIE_TEMPLATE")

    embed_layer = model.get_input_embeddings()
    template_embeds = embed_layer(template_tokens.input_ids)
    soft_tokens = {
        key: adapter.transform(vector.to(device)).to(
            dtype=template_embeds.dtype, device=template_embeds.device
        )
        for key, vector in hidden_vectors.items()
    }

    row_keys = [key for key in hidden_vectors for _ in range(n_samples)]
    descriptions: dict[CellKey, list[str]] = {key: [] for key in hidden_vectors}
    for start in range(0, len(row_keys), batch_size):
        chunk = row_keys[start : start + batch_size]
        embeddings = template_embeds.repeat(len(chunk), 1, 1)
        for row, key in enumerate(chunk):
            for pos in inject_positions:
                embeddings[row, pos, :] = soft_tokens[key]

        outputs = model.generate(
            inputs_embeds=embeddings,
            # Every row is an identical copy of the same template (no padding
            # ever exists within a batch here), but pad_token_id ==
            # eos_token_id makes that ambiguous to generate() without an
            # explicit mask -- so state it rather than let it guess.
            attention_mask=torch.ones(
                embeddings.shape[:2], dtype=torch.long, device=device
            ),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        for key, output in zip(chunk, outputs):
            text = tokenizer.decode(output, skip_special_tokens=True).strip()
            descriptions[key].append(text.rsplit('"', 1)[0] if '"' in text else text)

    return descriptions


def generate_interpretations(
    model,
    tokenizer,
    adapter: Adapter,
    hidden_vector: Float[Tensor, "hidden"],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    batch_size: int = 25,
) -> list[str]:
    """Single-cell convenience wrapper over `generate_interpretations_batch`."""
    return generate_interpretations_batch(
        model,
        tokenizer,
        adapter,
        {0: hidden_vector},
        n_samples,
        max_new_tokens,
        temperature,
        device,
        batch_size,
    )[0]
