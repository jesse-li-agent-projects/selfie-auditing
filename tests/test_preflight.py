"""Tokenizer-free half of preflight: the config checks.

The tokenization pins need a real tokenizer, so they are exercised in
test_real_tokenizer.py under the hf_cache marker.
"""

import pytest

from config import full_sweep_config
from preflight import PreflightError, check_config, check_output_dir


def config_for(tmp_path, **overrides):
    config = full_sweep_config(["gold"], num_hidden_layers=16, output_dir=tmp_path)
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_check_config_accepts_a_normal_sweep(tmp_path):
    check_config(config_for(tmp_path), 16)


def test_check_config_rejects_a_layer_count_the_model_does_not_have(tmp_path):
    # The smoke config's layer list is derived from a default count, so this is
    # the check that catches a model whose real config.json disagrees.
    with pytest.raises(PreflightError, match="outside the model"):
        check_config(config_for(tmp_path), 8)


def test_check_config_rejects_duplicate_words(tmp_path):
    with pytest.raises(PreflightError, match="duplicate"):
        check_config(config_for(tmp_path, words=["gold", "gold"]), 16)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"words": []}, "no words"),
        ({"n_samples": 0}, "n_samples"),
        ({"sample_start": -1}, "sample_start"),
    ],
)
def test_check_config_rejects_an_empty_or_negative_range(tmp_path, overrides, match):
    with pytest.raises(PreflightError, match=match):
        check_config(config_for(tmp_path, **overrides), 16)


def test_check_output_dir_creates_and_leaves_nothing_behind(tmp_path):
    config = config_for(tmp_path, output_dir=tmp_path / "new" / "nested")

    check_output_dir(config)

    assert config.output_dir.is_dir()
    assert not list(config.output_dir.iterdir())


def test_check_output_dir_rejects_an_unwritable_path(tmp_path):
    read_only = tmp_path / "read_only"
    read_only.mkdir(mode=0o500)

    with pytest.raises(PreflightError, match="not writable"):
        check_output_dir(config_for(tmp_path, output_dir=read_only / "out"))
