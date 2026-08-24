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

from typing import Protocol

import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Float
from selfie_adapters import load_adapter
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
    """Interface generate_interpretations() needs -- satisfied by both the real
    `selfie_adapters.SelfIEAdapter` and the smoke-test stub adapters."""

    def transform(self, vector: Float[Tensor, "hidden"]) -> Float[Tensor, "hidden"]: ...


def load_wikipedia_adapter(
    adapter_repo: str, adapter_filename: str, device: str
) -> Adapter:
    adapter_path = hf_hub_download(repo_id=adapter_repo, filename=adapter_filename)
    return load_adapter(adapter_path, device=device)


@torch.no_grad()
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
    """Inject the adapter's soft token into SELFIE_TEMPLATE and sample n_samples
    generations, batched to bound peak memory."""
    soft_token = adapter.transform(hidden_vector.to(device))

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
    soft_token_cast = soft_token.to(
        dtype=template_embeds.dtype, device=template_embeds.device
    )

    descriptions: list[str] = []
    remaining = n_samples
    while remaining > 0:
        n = min(batch_size, remaining)
        embeddings = template_embeds.repeat(n, 1, 1)
        for pos in inject_positions:
            embeddings[:, pos, :] = soft_token_cast

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
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True).strip()
            descriptions.append(text.rsplit('"', 1)[0] if '"' in text else text)
        remaining -= n

    return descriptions
