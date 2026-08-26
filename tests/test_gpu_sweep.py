"""GPU tests entering through main() (plan S6.1, S6.5).

Real 1B dummy model on a real CUDA device, through the ordinary CLI/main()
load path -- the same one a real 8B run uses, just with four different
strings. These say nothing about whether the sweep *finds* anything -- the
dummy adapter and LoRA are untrained by construction -- only that negative
offsets address the tokens the design claims, that main() writes exactly the
requested cells, that two shards really produce two different halves of a
cell, and that the published dummy LoRA is a genuine (non-no-op) perturbation.
"""

import pytest
import torch

from config import (
    DUMMY_ADAPTER_FILE,
    DUMMY_ADAPTER_REPO,
    DUMMY_BASE_MODEL,
    DUMMY_LORA_REPO_TEMPLATE,
    DUMMY_WORD,
    SECRET_PROMPT,
    Arm,
    Position,
)
from extract import build_prompt, cache_path, extract_hidden_states, user_prompt_span
from merge_results import merge
from model_loading import attach_taboo_loras, load_base_model, load_tokenizer
from results_store import read_cells, read_metadata
from run_pipeline import main, parse_args

pytestmark = [pytest.mark.gpu, pytest.mark.hf_cache]

DEVICE = "cuda:0"
LAYER = 8


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(DUMMY_BASE_MODEL)


@pytest.fixture(scope="module")
def model():
    return load_base_model(DUMMY_BASE_MODEL, device=DEVICE, dtype="bfloat16")


def test_span_reads_the_intended_tokens(model, tokenizer):
    """Negative offsets address what plan S3 says they do, end to end through
    a real forward pass rather than through a fake tokenizer."""
    ids = tokenizer(
        build_prompt(tokenizer, SECRET_PROMPT, None),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    span = user_prompt_span(tokenizer, ids, SECRET_PROMPT)

    extraction = extract_hidden_states(
        model,
        tokenizer,
        SECRET_PROMPT,
        None,
        [LAYER],
        [Position.USER_PROMPT_SPAN],
        DEVICE,
    )
    absolute = [len(ids) + offset for offset in span]
    by_absolute = extract_hidden_states(
        model, tokenizer, SECRET_PROMPT, None, [LAYER], absolute, DEVICE
    )

    assert extraction.positions == span
    assert [extraction.tokens[f"pos{o}"] for o in span] == [
        tokenizer.decode([ids[o]]) for o in span
    ]
    for offset, index in zip(span, absolute):
        assert torch.equal(
            extraction.hidden_states[(LAYER, offset)],
            by_absolute.hidden_states[(LAYER, index)],
        )


def dummy_run_args(tmp_path, *extra):
    """The base flags every dummy run below shares: the four DUMMY_* weight
    identifiers and --output-dir/--device. `extra` appends the flags that make
    each test's run different (arms, layers, budget, sharding)."""
    return parse_args(
        [
            "--words",
            DUMMY_WORD,
            "--model",
            DUMMY_BASE_MODEL,
            "--adapter-repo",
            DUMMY_ADAPTER_REPO,
            "--adapter-filename",
            DUMMY_ADAPTER_FILE,
            "--output-dir",
            str(tmp_path),
            "--device",
            DEVICE,
            *extra,
        ]
    )


def test_main_sweeps_every_requested_cell_and_writes_them(tmp_path):
    """The primary end-to-end check. --arms control,prompted means no LoRA
    and no PEFT, so this test needs only the base model and the dummy
    adapter."""
    args = dummy_run_args(
        tmp_path,
        "--arms",
        "control,prompted",
        "--layers",
        "0,8",
        "--n-samples",
        "2",
        "--max-new-tokens",
        "8",
    )

    cells_path = main(args)

    assert cells_path.parent == tmp_path
    metadata = read_metadata(cells_path)
    cells = list(read_cells(cells_path))
    span = set(metadata["spans"]["control"])  # what USER_PROMPT_SPAN expanded to

    # Assert set equality, not membership, at every level: `==` catches a
    # sweep that runs an extra cell or drops one, `in` does not.
    arms_seen = {cell["arm"] for cell in cells}
    assert arms_seen == {"control", "prompted"}
    for arm in arms_seen:
        arm_cells = [cell for cell in cells if cell["arm"] == arm]
        assert {cell["word"] for cell in arm_cells} == {DUMMY_WORD}
        layers_seen = {cell["layer"] for cell in arm_cells}
        assert layers_seen == {0, 8}
        for layer in layers_seen:
            layer_cells = [cell for cell in arm_cells if cell["layer"] == layer]
            assert {cell["position"] for cell in layer_cells} == span
            for cell in layer_cells:
                assert len(cell["generations"]) == 2
                assert len(cell["hits"]) == 2

    # The hidden-state cache is output-directory data too, not just the JSONL.
    for arm in ("control", "prompted"):
        assert cache_path(tmp_path, Arm(arm), DUMMY_WORD).exists()


def test_finetuned_arm_loads_the_dummy_lora(tmp_path):
    """Kept small and separate from the primary test, so a problem fetching
    the LoRA repo cannot take out the primary test."""
    args = dummy_run_args(
        tmp_path,
        "--lora-template",
        DUMMY_LORA_REPO_TEMPLATE,
        "--arms",
        "finetuned",
        "--layers",
        "0",
        "--n-samples",
        "1",
        "--max-new-tokens",
        "8",
    )

    cells = list(read_cells(main(args)))

    assert cells
    assert {cell["arm"] for cell in cells} == {"finetuned"}
    assert all(len(cell["generations"]) == 1 for cell in cells)


def test_two_shards_produce_different_generations(tmp_path):
    """If cell_seed were ignored, or both shards seeded identically, a
    "2n-sample" cell would really be n samples counted twice -- and the merged
    output would look perfectly healthy. Nothing in tier 1 can catch that."""
    n = 4
    paths = [
        main(
            dummy_run_args(
                tmp_path,
                "--arms",
                "control",
                "--layers",
                "0",
                "--n-samples",
                str(n),
                "--sample-start",
                str(start),
                "--max-new-tokens",
                "8",
            )
        )
        for start in (0, n)
    ]
    assert len(set(paths)) == 2  # distinct shard filenames, no collision

    def generations(path):
        return {
            cell["position"]: cell["generations"]
            for cell in read_cells(path)
            if cell["word"] == DUMMY_WORD and cell["layer"] == 0
        }

    first, second = generations(paths[0]), generations(paths[1])
    assert first.keys() == second.keys()
    assert all(len(g) == n for g in first.values())
    assert first != second

    merged = merge(tmp_path, total=2 * n)
    assert all(len(cell["generations"]) == 2 * n for cell in read_cells(merged))


def test_dummy_lora_perturbs_forward_pass():
    """A property of the published dummy LoRA, not of any particular run --
    checked once here rather than on every --finetuned run. Asks whether the
    random LoRA is a genuine no-op (`init_lora_weights=False` is load-bearing,
    see dummy_weights.RANDOM_LORA_HYPERPARAMS) and whether disable_adapter()
    returns a clean base model. Without this, a bug where
    set_adapter()/disable_adapter() silently no-ops -- or a zero-initialized
    lora_B making the "random" adapter an exact no-op -- would leave every
    FINETUNED-arm run producing plausible output while testing nothing.

    Loads its own model rather than the module's `model` fixture: PEFT wraps
    a model's Linear layers in place, so attaching a LoRA here would leave
    every other test in this module running against a silently-wrapped model.
    """
    tokenizer = load_tokenizer(DUMMY_BASE_MODEL)
    base_model = load_base_model(DUMMY_BASE_MODEL, device=DEVICE, dtype="bfloat16")
    layer, position = 0, Position.ASSISTANT_BOUNDARY

    baseline = extract_hidden_states(
        base_model, tokenizer, SECRET_PROMPT, None, [layer], [position], DEVICE
    ).hidden_states[(layer, position)]

    peft_model = attach_taboo_loras(base_model, [DUMMY_WORD], DUMMY_LORA_REPO_TEMPLATE)
    peft_model.set_adapter(DUMMY_WORD)
    active = extract_hidden_states(
        peft_model, tokenizer, SECRET_PROMPT, None, [layer], [position], DEVICE
    ).hidden_states[(layer, position)]
    with peft_model.disable_adapter():
        disabled = extract_hidden_states(
            peft_model, tokenizer, SECRET_PROMPT, None, [layer], [position], DEVICE
        ).hidden_states[(layer, position)]

    active_vs_disabled = (active - disabled).abs().max().item()
    disabled_vs_baseline = (disabled - baseline).abs().max().item()
    assert active_vs_disabled > 1e-3, (
        "the dummy LoRA had no measurable effect on the forward pass "
        f"(max diff {active_vs_disabled}) -- likely a no-op adapter (check init_lora_weights)"
    )
    assert disabled_vs_baseline < 1e-3, (
        "disable_adapter() output differs from the pre-wrap base model "
        f"(max diff {disabled_vs_baseline}) -- unload()/disable_adapter() may not be "
        "giving back a clean base model"
    )
