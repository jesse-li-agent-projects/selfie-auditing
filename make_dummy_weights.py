r"""Generate the random-weight dummy SelfIE adapter and taboo LoRA for a small model.

Neither real weight set exists below 8B: the SelfIE adapter checkpoint is
4096-wide, and the bcywinski taboo LoRAs are published for the 8B base only.
This writes random-weight stand-ins of the right shape, in the real on-disk
formats, so `explore_selfie.py` and `run_pipeline.py` can run end to end
against a 1B model through their ordinary load paths -- no stub objects, no
branching on "is this a dummy run".

This is the generator that produced the dummy weights already published on
the Hub (config.py's DUMMY_ADAPTER_REPO/DUMMY_LORA_REPO_TEMPLATE); it exists
as a provenance record and for regenerating a fixture at a different seed or
width, not as something a run calls. explore_selfie.py and run_pipeline.py
load the SelfIE adapter from a local path (--adapter-path), so a freshly
generated adapter is usable directly, with no Hub round trip. The taboo LoRA
still loads through a Hub-or-local repo template (--lora-template), so
publishing one for reuse across runs still means uploading it.

    python make_dummy_weights.py --output-dir outputs/dummy_weights

What this proves: shapes, dtypes, file formats, dimension checks, the LoRA
hot-swap path, position finding, scoring. What it does NOT prove: that any of
it finds anything. The adapter is untrained and the LoRA has no secret to hide,
so the generations are noise by construction -- read them for shape, never for
meaning.
"""

import argparse
import hashlib


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write both weight sets to"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to size the weights for (default: the dummy 1B model)",
    )
    parser.add_argument(
        "--word",
        nargs="+",
        default=None,
        help="Word(s) to generate a taboo LoRA for (default: the dummy word)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from pathlib import Path

    from config import DUMMY_ADAPTER_FILE, DUMMY_BASE_MODEL, DUMMY_WORD
    from dummy_weights import (
        create_random_lora,
        create_random_selfie_adapter,
        embedding_norm,
    )
    from model_loading import load_base_model

    model_name = args.model or DUMMY_BASE_MODEL
    words = args.word or [DUMMY_WORD]
    output_dir = Path(args.output_dir)
    lora_template = str(output_dir / "taboo_lora" / "{word}")

    model = load_base_model(model_name, device=args.device, dtype=args.dtype)

    # Both the width and the soft-token scale come from the model itself, never
    # a hardcoded guess: a wrong width writes an adapter that loads fine and
    # then fails the dimension check at the first generation.
    hidden_dim = model.config.hidden_size
    init_scale = embedding_norm(model)
    adapter_path = create_random_selfie_adapter(
        hidden_dim, output_dir / DUMMY_ADAPTER_FILE, init_scale, seed=args.seed
    )
    print(
        f"Wrote {hidden_dim}-dim random SelfIE adapter (scale {init_scale:.3f}) to {adapter_path}"
    )

    for word in words:
        # A seed per word, not one for all: identical LoRAs would make a
        # set_adapter() that selects the wrong word's adapter indistinguishable
        # from a correct one in any dummy run. Derived from the word itself, so
        # regenerating one word later reproduces the same weights.
        word_seed = args.seed + int.from_bytes(
            hashlib.blake2b(word.encode(), digest_size=4).digest(), "big"
        )
        # Returns the unwrapped base model, so each word starts from a clean one.
        model = create_random_lora(model, lora_template, word, seed=word_seed)
        print(
            f"Wrote random taboo LoRA for {word!r} to {lora_template.format(word=word)}"
        )

    print(
        f"\nUse {adapter_path} directly with --adapter-path. "
        f"Upload {output_dir}'s LoRA directory to the Hub to use it with --lora-template."
    )
