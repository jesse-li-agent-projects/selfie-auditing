"""selfie_on_assistant.py end to end, entering through main().

Real 1B dummy model on a real CUDA device, as in test_gpu_sweep.py. The dummy
weights are untrained by construction, so nothing here says the sweep *finds*
anything -- only that the cells it writes are the reply's own tokens, and that
a rerun after a crash finishes the sweep instead of redoing or duplicating it.
"""

import gc

import pytest
import torch
from huggingface_hub import hf_hub_download

from config import (
    DUMMY_ADAPTER_FILE,
    DUMMY_ADAPTER_REPO,
    DUMMY_BASE_MODEL,
    DUMMY_WORD,
    Arm,
)
from extract import cache_path
from results_store import read_cells, read_metadata
from selfie_on_assistant import main, parse_args

pytestmark = [pytest.mark.gpu, pytest.mark.hf_cache]

DEVICE = "cuda:0"


@pytest.fixture(autouse=True)
def _free_cuda_memory_after_each_test():
    """See test_gpu_sweep.py: every test here loads its own model, and a
    reference kept alive past a test's scope is enough to OOM the next."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


def dummy_run_args(tmp_path, *extra):
    """The base flags every dummy run below shares. --arms control means no
    LoRA and no PEFT, so these tests need only the base model and adapter."""
    return parse_args(
        [
            "--words",
            DUMMY_WORD,
            "--model",
            DUMMY_BASE_MODEL,
            "--adapter-path",
            hf_hub_download(repo_id=DUMMY_ADAPTER_REPO, filename=DUMMY_ADAPTER_FILE),
            "--output-dir",
            str(tmp_path),
            "--device",
            DEVICE,
            "--arms",
            "control",
            "--layers",
            "0,8",
            "--n-samples",
            "2",
            "--max-new-tokens",
            "8",
            "--response-max-new-tokens",
            "3",
            *extra,
        ]
    )


def test_main_interprets_every_token_of_the_reply(tmp_path):
    cells_path = main(dummy_run_args(tmp_path))

    metadata = read_metadata(cells_path)
    reply = metadata["responses"]["control"][DUMMY_WORD]
    cells = list(read_cells(cells_path))

    assert 1 <= len(reply["response_ids"]) <= 3
    assert list(reply["tokens"]) == [f"resp{i}" for i in range(len(reply["tokens"]))]
    # Set equality, not membership: `==` catches an extra or dropped cell.
    for layer in (0, 8):
        layer_cells = [cell for cell in cells if cell["layer"] == layer]
        assert {cell["position"] for cell in layer_cells} == set(reply["tokens"])
        assert all(len(cell["generations"]) == 2 for cell in layer_cells)
    assert cache_path(tmp_path, Arm.CONTROL, DUMMY_WORD).exists()


def test_a_rerun_finishes_an_interrupted_shard(tmp_path):
    """The crash-recovery path: the surviving cells are kept as they are, the
    missing ones are generated, and no cell ends up in the file twice."""
    cells_path = main(dummy_run_args(tmp_path))
    complete = list(read_cells(cells_path))
    assert len(complete) > 1, "need at least two cells for a partial file"
    kept = complete[:1]
    first_line = cells_path.read_text().split("\n")[0]
    cells_path.write_text(first_line + "\n")
    gc.collect()
    torch.cuda.empty_cache()

    resumed = list(read_cells(main(dummy_run_args(tmp_path))))

    assert resumed[: len(kept)] == kept
    assert len(resumed) == len(complete)
    assert len({tuple(cell[f] for f in ("layer", "position")) for cell in resumed}) == (
        len(complete)
    )
