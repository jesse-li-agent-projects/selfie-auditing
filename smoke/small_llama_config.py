"""Smoke-test config: Llama-3.2-1B-Instruct in place of the real 8B pipeline (plan S6).

Exercises every piece of machinery that doesn't depend on the real adapter or
real taboo weights: shapes, file formats, caching, scoring/aggregation, the
config-sweep machinery, chat-template rendering, and reserved-token position
finding. It does NOT exercise the FINETUNED arm's real LoRA-loading path --
there is no taboo LoRA trained for the 1B model, so that path is only ever
smoke-tested for real on the 8B run. A green smoke pass says nothing about
whether the adapter can actually find anything (plan S6, final paragraph).
"""

from __future__ import annotations

from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from selfie_taboo.config import Arm, PipelineConfig, Position, layers_smoke

SMOKE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
SMOKE_WORD = "banana"  # fake secret, only exercises prompt/scoring code paths


class IdentityAdapter:
    """Stub replacing SelfIEAdapter: passes the (contrastive) vector through
    unchanged. Dimension-agnostic, so it works at the 1B model's hidden size
    without needing a real adapter checkpoint (which is 4096-dim, for the 8B
    model only)."""

    def transform(self, vector: Float[Tensor, "hidden"]) -> Float[Tensor, "hidden"]:
        return vector


class RandomAffineAdapter:
    """Stub replacing SelfIEAdapter with a fixed random affine map, so the
    smoke test also exercises a nontrivial (non-identity) transform shape-wise."""

    def __init__(self, hidden_dim: int, device: str, seed: int = 0):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.weight = (
            torch.randn(hidden_dim, hidden_dim, generator=generator).to(device) * 0.01
        )
        self.bias = torch.zeros(hidden_dim, device=device)

    def transform(self, vector: Float[Tensor, "hidden"]) -> Float[Tensor, "hidden"]:
        return self.weight @ vector.to(self.weight.device) + self.bias


def smoke_config(output_dir: Path, num_hidden_layers: int = 16) -> PipelineConfig:
    """Config for the local smoke pass. `num_hidden_layers=16` is
    Llama-3.2-1B-Instruct's published layer count; overridden at runtime once
    the real config.json is loaded (plan S2 preflight pattern applies here too
    -- don't trust a hardcoded count over the model's own config)."""
    return PipelineConfig(
        base_model=SMOKE_MODEL,
        adapter_repo="",  # unused: smoke test injects a stub adapter directly
        adapter_filename="",
        mean_vector_layer=-1,  # unused: smoke test skips contrastive subtraction
        taboo_lora_repo_template="",  # unused: no taboo LoRA exists at 1B scale
        words=[SMOKE_WORD],
        arms=[
            Arm.CONTROL,
            Arm.PROMPTED,
        ],  # FINETUNED needs a real LoRA -- see module docstring
        layers=layers_smoke(num_hidden_layers),
        positions=[Position.ASSISTANT_BOUNDARY, Position.LAST_CONTENT_TOKEN],
        n_samples=3,
        temperature=0.7,
        max_new_tokens=20,
        output_dir=output_dir,
    )
