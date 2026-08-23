"""Smoke-test config: Llama-3.2-1B-Instruct in place of the real 8B pipeline (plan S6).

Exercises every piece of machinery that doesn't depend on the real adapter
weights: shapes, file formats, caching, scoring/aggregation, the config-sweep
machinery, chat-template rendering, reserved-token position finding, and --
via a freshly generated random-init LoRA standing in for a real taboo LoRA --
the FINETUNED arm's LoRA-loading and hot-swap path itself
(`PeftModel.from_pretrained` + `disable_adapter()` + `PeftModel.generate`).
What it can NOT exercise is whether any of this finds anything real: the
random LoRA has no secret to hide, and the stub adapter is not a trained
SelfIE adapter. A green smoke pass says nothing about whether the adapter can
actually find anything (plan S6, final paragraph).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from jaxtyping import Float
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file as safetensors_save_file
from torch import Tensor
from transformers import PreTrainedModel

from config import Arm, PipelineConfig, Position, layers_smoke

SMOKE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
SMOKE_WORD = "banana"  # fake secret, only exercises prompt/scoring code paths
SMOKE_ADAPTER_FILENAME = "selfie-random-scalar-affine.safetensors"

# Matches the real bcywinski taboo LoRAs' hyperparameters
# (research_notes_selfie_mechanism.md S3), so the random adapter this
# generates has the same shape (rank, target modules) as the real thing --
# only the weights differ (random init vs. trained).
#
# init_lora_weights=False is load-bearing, not decorative: PEFT's default
# (True) follows the LoRA paper and zero-inits lora_B, so DeltaW = B @ A = 0
# at init regardless of lora_A -- the adapter would be an exact no-op and
# this arm would silently stop testing anything. False gives fully random
# (nonzero) weights on both A and B.
RANDOM_LORA_HYPERPARAMS = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    init_lora_weights=False,
    target_modules=[
        "down_proj",
        "gate_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
        "k_proj",
    ],
    task_type="CAUSAL_LM",
)


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


def create_random_lora(
    model: PreTrainedModel, save_dir_template: str, word: str, seed: int = 0
) -> PreTrainedModel:
    """Build a random-init LoRA (real taboo LoRA hyperparams, random weights),
    save it where `save_dir_template.format(word=word)` points, and return the
    clean, unwrapped base model.

    Saves under the default adapter name and reloads via `attach_taboo_loras`
    rather than handing back the wrapped model directly, so the smoke test
    exercises the exact same `PeftModel.from_pretrained` load path the real 8B
    taboo LoRAs use.
    """
    torch.manual_seed(seed)
    peft_model = get_peft_model(model, LoraConfig(**RANDOM_LORA_HYPERPARAMS))
    save_dir = Path(save_dir_template.format(word=word))
    # Default adapter name only: a named adapter nests save_pretrained()'s
    # output one directory deeper, which doesn't match the real taboo repos'
    # single-adapter layout that attach_taboo_loras() expects. The adapter
    # gets its real name (`word`) only when attach_taboo_loras() reloads it.
    peft_model.save_pretrained(save_dir)
    # save_pretrained's adapter_model.safetensors comes out at mode 0600
    # regardless of umask (same issue fixed for the hidden-state cache in
    # extract.py) -- loosen it back to the output directory's ACL intent.
    for f in save_dir.iterdir():
        f.chmod(0o664)
    return peft_model.unload()


def create_random_selfie_adapter(
    hidden_dim: int, path: Path, init_scale: float, seed: int = 0
) -> Path:
    """Write a random-weight SelfIE adapter checkpoint of width `hidden_dim`.

    The real adapter is 4096-wide (8B only), so nothing at 1B scale can load
    it. This writes a checkpoint in the same `selfie_adapters` safetensors
    format at whatever width the small model uses, which lets the smoke path
    go through the ordinary `load_adapter()` -> `SelfIEAdapter.transform()`
    code instead of a stub object -- the loader, the metadata header, the
    dimension check and the projection math all get exercised for real.

    Architecture matches the real checkpoint (`scalar_affine`); only the
    weights are random, so the interpretations it produces are meaningless.

    `init_scale` sets the norm of the soft token (the projection L2-normalizes
    its input first). Pass the target model's own typical embedding norm --
    see `embedding_norm`. A soft token far outside embedding scale makes
    generation degenerate for reasons that have nothing to do with the shapes
    this checkpoint exists to test.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = {
        "log_scale": torch.log(torch.tensor([init_scale])),
        # Small relative to the scaled input, so the bias perturbs the soft
        # token rather than dominating it, at any hidden_dim.
        "bias": torch.randn(hidden_dim, generator=generator)
        * (0.1 * init_scale / hidden_dim**0.5),
    }
    # SelfIEAdapter reads its architecture out of the safetensors metadata
    # header; without it the loader rejects the file outright.
    metadata = {
        "model_dim": str(hidden_dim),
        "config_json": json.dumps(
            {
                "projection": {
                    "type": "scalar_affine",
                    "normalize_input": True,
                    "init_scale": init_scale,
                }
            }
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors_save_file(tensors, str(path), metadata=metadata)
    path.chmod(0o664)  # safetensors writes 0600 regardless of umask
    return path


def embedding_norm(model: PreTrainedModel) -> float:
    """Median L2 norm of the model's input embedding rows -- the scale a soft
    token has to land near to be a plausible embedding for that model."""
    weight = model.get_input_embeddings().weight
    return weight.detach().float().norm(dim=-1).median().item()


def smoke_config(output_dir: Path, num_hidden_layers: int = 16) -> PipelineConfig:
    """Config for the local smoke pass. `num_hidden_layers=16` is
    Llama-3.2-1B-Instruct's published layer count; overridden at runtime once
    the real config.json is loaded (plan S2 preflight pattern applies here too
    -- don't trust a hardcoded count over the model's own config)."""
    return PipelineConfig(
        base_model=SMOKE_MODEL,
        adapter_repo="",  # unused: smoke test injects a stub adapter directly
        adapter_filename="",
        taboo_lora_repo_template=str(output_dir / "random_lora" / "{word}"),
        words=[SMOKE_WORD],
        arms=[Arm.CONTROL, Arm.PROMPTED, Arm.FINETUNED],
        layers=layers_smoke(num_hidden_layers),
        positions=[Position.ASSISTANT_BOUNDARY, Position.LAST_CONTENT_TOKEN],
        n_samples=3,
        temperature=0.7,
        max_new_tokens=20,
        output_dir=output_dir,
    )
