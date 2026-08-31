"""Tests for adapter_training.checkpoints (plan step 2a, test 8)."""

import pytest

from adapter_training.checkpoints import (
    load_projection,
    save_checkpoint,
    untrained_projection,
)
from adapter_training.inference import load_adapter

DIM = 6


def config_for(
    projection_type,
    *,
    normalize_input=True,
    init_scale=3.0,
    low_rank_rank=None,
    low_rank_init_factor=None,
):
    return {
        "projection": {
            "type": projection_type,
            "normalize_input": normalize_input,
            "init_scale": init_scale,
            "low_rank_rank": low_rank_rank,
            "low_rank_init_factor": low_rank_init_factor,
        }
    }


@pytest.mark.parametrize(
    "projection_type,low_rank_rank",
    [("scalar_affine", None), ("scalar_affine_plus_low_rank", 4)],
)
def test_save_checkpoint_round_trips_through_load_adapter(
    tmp_path, projection_type, low_rank_rank
):
    from adapter_training.projection import create_projection_module

    projection = create_projection_module(
        projection_type=projection_type,
        dim=DIM,
        normalize_input=True,
        device="cpu",
        init_scale=3.0,
        low_rank_rank=low_rank_rank,
        low_rank_init_factor=0.01 if low_rank_rank is not None else None,
    )
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        projection,
        config_for(
            projection_type,
            low_rank_rank=low_rank_rank,
            low_rank_init_factor=0.01 if low_rank_rank is not None else None,
        ),
        global_step=123,
        best_val_loss=1.5,
    )
    adapter = load_adapter(str(path), device="cpu")

    assert adapter.config["type"] == projection_type
    assert adapter.config["normalize_input"] is True
    assert adapter.model_dim == DIM
    assert adapter.global_step == 123
    assert adapter.best_val_loss == 1.5
    for name, tensor in projection.state_dict().items():
        assert (adapter.projection.state_dict()[name] == tensor).all()


def test_load_projection_round_trips_via_evaluate_adapter_helper(tmp_path):
    """`load_projection` is `checkpoints.py`'s own doorway back in, same file."""
    from adapter_training.projection import create_projection_module

    projection = create_projection_module(
        projection_type="scalar_affine",
        dim=DIM,
        normalize_input=False,
        device="cpu",
        init_scale=1.0,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, projection, config_for("scalar_affine"), global_step=1, best_val_loss=None
    )

    loaded, metadata = load_projection(path, device="cpu", dim=DIM)

    assert metadata["model_dim"] == DIM
    assert metadata["projection_type"] == "scalar_affine"
    assert metadata.get("best_val_loss") is None


def test_load_projection_rejects_a_dim_mismatch(tmp_path):
    from adapter_training.projection import create_projection_module

    projection = create_projection_module(
        projection_type="scalar_affine",
        dim=DIM,
        normalize_input=True,
        device="cpu",
        init_scale=1.0,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, projection, config_for("scalar_affine"), global_step=0, best_val_loss=None
    )

    with pytest.raises(ValueError):
        load_projection(path, device="cpu", dim=DIM + 1)


def test_untrained_projection_matches_identity_baseline_yaml():
    projection = untrained_projection(DIM, device="cpu")

    assert projection.normalize_input is True
    assert projection.get_scale() == pytest.approx(1.0)
    assert type(projection).__name__ == "ScaleOnlyProjection"
