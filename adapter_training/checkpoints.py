"""Checkpoint IO for projection modules: loading any projection -- a fresh
untrained one, one this repo trained, or upstream's own Hub checkpoint -- and
writing checkpoints in the exact format `adapter_training.load_adapter` reads,
so `interpret.py` and every other downstream consumer stay unchanged
regardless of who trained the file.

Also the home of the trainer's own resume state, which is deliberately a
separate file: it carries optimizer state no evaluator wants, and keeping it
out of the checkpoints leaves their format exactly what `load_adapter` reads.
"""

from __future__ import annotations

from pathlib import Path

import torch

from adapter_training.inference import load_adapter
from adapter_training.projection import create_projection_module


def _resolve_checkpoint_path(source: str | Path) -> Path:
    """A local file, or a `repo_id:filename` pair downloaded from the Hub."""
    path = Path(source)
    if path.exists():
        return path
    text = str(source)
    if ":" in text:
        repo_id, _, filename = text.rpartition(":")
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo_id, filename=filename))
    raise FileNotFoundError(
        f"{source!r} is not a local checkpoint and not a 'repo_id:filename' pair"
    )


def load_projection(source: str | Path, *, device, dim: int | None = None):
    """Load a projection module and its recorded metadata, via
    `adapter_training.load_adapter`.

    :param source: a local `.pt`/`.safetensors` path, or a `repo_id:filename`
        pair naming a file on the Hub
    :param device: device to load the projection onto
    :param dim: if given, checked against the checkpoint's own `model_dim`
        (catches a vectors/checkpoint width mismatch before training loss
        makes no sense of it)
    :return: `(projection, metadata)`, `metadata` from `SelfIEAdapter.get_metadata()`
    :raises ValueError: if the checkpoint's `normalize_input` is not `True` --
        no known training config produces `False`, so it is worth
        investigating rather than silently accepting
    """
    path = _resolve_checkpoint_path(source)
    adapter = load_adapter(str(path), device=str(device))
    metadata = adapter.get_metadata()
    if dim is not None and metadata["model_dim"] != dim:
        raise ValueError(
            f"checkpoint {source!r} has model_dim={metadata['model_dim']}, "
            f"expected {dim}"
        )
    if not metadata["normalize_input"]:
        raise ValueError(
            f"checkpoint {source!r} has normalize_input=False, which no known "
            "training config produces -- investigate before trusting this checkpoint"
        )
    return adapter.projection, metadata


def untrained_projection(dim: int, *, device, init_scale: float = 1.0):
    """The floor comparator: upstream's own `identity_baseline.yaml`
    (`resources/selfie-adapters/training/configs/identity_baseline.yaml`) --
    `scale_only`, `init_scale=1.0`, never trained.

    :param dim: model hidden size
    :param device: device to place the projection on
    :param init_scale: the untrained scale. Only worth changing where an
        experiment's own reference picked a different one -- untrained SelfIE
        is sensitive to it, which is the paper's motivation for learning the
        mapping instead
    """
    return create_projection_module(
        projection_type="scale_only",
        dim=dim,
        normalize_input=True,
        device=device,
        init_scale=init_scale,
    )


def save_checkpoint(
    path: Path,
    projection,
    config: dict,
    *,
    global_step: int,
    best_val_loss: float | None,
) -> None:
    """Write the checkpoint dict `adapter_training.load_adapter` reads --
    matching upstream's own trainer (`training/trainer.py::_save_checkpoint`)
    minus training-only state (optimizer/scheduler) that no evaluator needs.

    :param path: file to write (`.pt`)
    :param projection: the projection module to save (trained or untrained)
    :param config: full run config as a dict; must have a `projection`
        section with `type`, `normalize_input`, `init_scale`, `low_rank_rank`
    :param global_step: training step this checkpoint was written at
    :param best_val_loss: best validation loss seen so far, or None
    """
    checkpoint = {
        "projection_state": projection.state_dict(),
        "model_dim": projection.dim,
        "checkpoint_format_version": 1,
        "config": config,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "projection_num_params": projection.num_parameters(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def save_resume_state(
    path: Path,
    projection,
    optimizer,
    *,
    train_config: dict,
    global_step: int,
    best_val_loss: float,
) -> None:
    """Write everything a `--resume` needs to carry on from `global_step`.

    Written to a temporary sibling and renamed, since the crash this file
    exists for can just as easily land mid-write: a half-written resume state
    would strand the run more thoroughly than having none at all.

    :param path: file to write (`.pt`)
    :param projection: the projection being trained
    :param optimizer: its optimizer, whose moments are the state a plain
        checkpoint cannot reconstruct
    :param train_config: the run's own config, checked on load
    :param global_step: training step this state was written at
    :param best_val_loss: best validation loss seen so far
    """
    state = {
        "projection_state": projection.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_config": train_config,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, staged)
    staged.replace(path)


def load_resume_state(path: Path) -> dict:
    """Read a `save_resume_state` file.

    Loaded onto the CPU: the tensors are placed by
    `Optimizer.load_state_dict`/`Module.load_state_dict` from there, so a run
    can resume on a different device than it crashed on.
    """
    return torch.load(path, map_location="cpu", weights_only=False)
