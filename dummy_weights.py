"""Random-weight fabrication for the dummy SelfIE adapter and taboo LoRA.

Neither real weight set exists below 8B: the SelfIE adapter checkpoint is
4096-wide, and the bcywinski taboo LoRAs are published for the 8B base only.
This module writes random-weight stand-ins of the right shape, in the real
on-disk formats -- run through the ordinary loaders, never a stub object.

Nothing in a run path calls this module any more: it is the provenance record
for the published dummy repos (config.py's DUMMY_* constants; see
make_dummy_weights.py), and tests/test_dummy_weights.py is what keeps that
record honest.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file as safetensors_save_file
from transformers import PreTrainedModel

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


def create_random_lora(
    model: PreTrainedModel, save_dir_template: str, word: str, seed: int = 0
) -> PreTrainedModel:
    """Build a random-init LoRA (real taboo LoRA hyperparams, random weights),
    save it where `save_dir_template.format(word=word)` points, and return the
    clean, unwrapped base model.

    Saves under the default adapter name and reloads via `attach_taboo_loras`
    rather than handing back the wrapped model directly, so a fabrication run
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
    format at whatever width the small model uses, which lets it go through
    the ordinary `load_adapter()` -> `SelfIEAdapter.transform()` code instead
    of a stub object -- the loader, the metadata header, the dimension check
    and the projection math all get exercised for real.

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
