"""CLI entry point: wires model loading -> extraction -> interpretation -> scoring.

    python run_pipeline.py --words gold,moon --output-dir results/sweep/

    # A local run on a small model, no 80GB card required -- see config.py's
    # DUMMY_* constants for what they point at:
    python run_pipeline.py --words banana --output-dir results/dummy/ \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --adapter-path outputs/dummy_weights/selfie-random-scalar-affine.safetensors \
        --lora-template cooleytukey/dummy-taboo-lora-llama-3.2-1b-{word}

The sweep is sharded by sample: every shard runs every cell, but only
`--n-samples` of that cell's generations, starting at `--sample-start`. Launch
one process per GPU with its own `--device` and sample range, then combine
them with merge_results.py:

    python run_pipeline.py --words gold,moon --output-dir results/sweep/ \
        --device cuda:0 --sample-start 0   --n-samples 100
    python run_pipeline.py --words gold,moon --output-dir results/sweep/ \
        --device cuda:1 --sample-start 100 --n-samples 100
    python merge_results.py --results-dir results/sweep/ --total 200

Each shard writes its cells to a JSONL file named by its sample range, with a
JSON metadata sidecar beside it (see results_store.py). Every run starts with a
preflight (preflight.py) that checks the tokenization and the config before any
weights load.
"""

import argparse
import hashlib
from itertools import product
from pathlib import Path

# Light import: config.py carries no heavy dependencies, so parsing --arms and
# --positions doesn't cost --help its speed.
from config import Arm, Position


def parse_arms(spec: str) -> list[Arm]:
    """Parse `--arms`: a comma-separated list of arm names.

    :param spec: the flag's raw value
    :return: the named arms
    :raises argparse.ArgumentTypeError: if any name is not an arm
    """
    try:
        return [Arm(name) for name in spec.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{error} -- expected a comma-separated list of "
            f"{', '.join(arm.value for arm in Arm)}"
        )


def parse_positions(spec: str) -> list[Position | int]:
    """Parse `--positions`: a comma-separated list of position names and/or
    raw token offsets (see `extract.resolve_position`).

    :param spec: the flag's raw value
    :return: the named positions and raw offsets, in the order given
    :raises argparse.ArgumentTypeError: if an entry is neither a name nor an offset
    """
    names = {position.value: position for position in Position}
    positions: list[Position | int] = []
    for token in spec.split(","):
        if token in names:
            positions.append(names[token])
            continue
        try:
            positions.append(int(token))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{token!r} is not a token offset or one of {', '.join(names)}"
            )
    return positions


def parse_layers(spec: str) -> str:
    """Check `--layers` is `"all"` or a list of layer indices.

    Returns the spec unchanged rather than the indices: resolving `"all"` needs
    the model's own layer count, which nothing knows at parse time (see
    `config.resolve_layers`).

    :param spec: the flag's raw value
    :return: `spec`, unchanged
    :raises argparse.ArgumentTypeError: if the spec is neither form
    """
    try:
        if spec != "all":
            [int(layer) for layer in spec.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{spec!r} is not 'all' or a comma-separated list of layer indices"
        )
    return spec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--words", required=True, help="Comma-separated secret words to sweep"
    )
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--model", default=None, help="Base model repo (default: 8B)")
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Local path to the SelfIE adapter checkpoint (default: the 8B one, "
        "fetched once to config.SELFIE_ADAPTER_PATH)",
    )
    parser.add_argument(
        "--lora-template",
        default=None,
        help="Taboo LoRA repo/path template containing {word} (default: the 8B repos)",
    )
    parser.add_argument(
        "--arms",
        type=parse_arms,
        default=None,
        help="Comma-separated arms to sweep (default: control,prompted,finetuned)",
    )
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default="all",
        help="'all' or a comma-separated list of 0-indexed layers (default: all)",
    )
    parser.add_argument(
        "--positions",
        type=parse_positions,
        default=None,
        help="Comma-separated position names or raw token offsets "
        "(default: user_prompt_span)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Generations per cell for this shard (default: the config's own)",
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="Index of this shard's first generation, for seeding and merging",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Rows per forward pass, pooled across an (arm, word)'s cells "
        "(default: the config's own)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args(argv)


def cell_seed(arm: str, word: str, sample_start: int) -> int:
    """Deterministic per-(arm, word), per-shard seed.

    blake2b rather than hash(): Python's hash() is salted per process, so a
    hash()-derived seed would give a different generation stream on every run
    and silently break replay. Folding in `sample_start` is what keeps two
    shards from regenerating the same samples -- without it a "200-sample"
    cell would really be 100 samples counted twice.

    One seed per (arm, word) rather than per cell: `generate_interpretations_batch`
    pools every layer/position cell of an (arm, word) into shared forward
    passes (see run_pipeline.py's docstring on batching), so only one RNG
    stream is live per (arm, word) -- replaying a single cell means replaying
    its whole (arm, word) group, at the batch size it was produced with.

    :param arm: the experimental condition
    :param word: the secret word
    :param sample_start: index of this shard's first generation
    :return: a seed in [0, 2**31)
    """
    digest = hashlib.blake2b(
        f"{arm}|{word}|{sample_start}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**31)


def run(config, *, adapter, tokenizer, peft_model) -> Path:
    """Run extraction + interpretation + scoring for every cell in `config`.

    Cells are appended as they finish, so an interrupted shard keeps every cell
    it had already paid for. The metadata sidecar carries the prompt and the
    span's resolved tokens: an offset like -11 only names a token relative to
    one formatted prompt, so recording what it actually resolved to is what
    keeps a stored result interpretable after a prompt or template change, and
    lets merge_results.py check two shards' comparability instead of assuming it.

    :param config: this shard's pipeline config
    :param adapter: SelfIE adapter used to interpret each cell's hidden state
    :param tokenizer: tokenizer shared by extraction and generation
    :param peft_model: the (possibly LoRA-wrapped) model to extract from and generate with
    :return: path to this shard's cells file; its metadata sidecar sits beside it
    """
    import torch

    from extract import (
        cache_path,
        extract_hidden_states,
        position_key,
        save_hidden_states,
    )
    from interpret import generate_interpretations_batch
    from model_loading import arm_active, system_prompt_for
    from results_store import (
        KEY_FIELDS,
        append_cell,
        shard_cells_path,
        write_metadata,
    )
    from scoring import score_cell

    spans: dict[str, dict[str, str]] = {}
    sample_end = config.sample_start + config.n_samples
    cells_path = shard_cells_path(config.output_dir, config.sample_start, sample_end)

    def metadata() -> dict:
        return {
            "sample_range": [config.sample_start, sample_end],
            "batch_size": config.batch_size,
            "spans": spans,
            **config.comparable_settings(),
        }

    # Written before the first cell so an interrupted shard is still
    # identifiable, and rewritten as each arm's spans become known.
    write_metadata(cells_path, metadata())
    with open(cells_path, "w") as handle:
        # arm: control/prompt/fine-tuned; word: which word is taboo
        for arm, word in product(config.arms, config.words):
            with arm_active(peft_model, arm, word):
                system_prompt = system_prompt_for(arm, word)
                extraction = extract_hidden_states(
                    peft_model,
                    tokenizer,
                    config.secret_prompt,
                    system_prompt,
                    config.layers,
                    config.positions,
                    config.device,
                )
                save_hidden_states(
                    cache_path(config.output_dir, arm, word), extraction.hidden_states
                )
                # Recorded, not checked: preflight.py already proved every
                # (arm, word) resolves the same span, and against a pinned
                # measurement rather than merely against each other.
                spans.setdefault(arm.value, extraction.tokens)
                write_metadata(cells_path, metadata())

                # One seed, one pooled batch of forward passes for every cell
                # in this (arm, word): every cell here shares the same LoRA
                # state and generation settings, differing only in which
                # hidden state's soft token gets injected, so batch_size is no
                # longer bounded by a single cell's n_samples (see
                # generate_interpretations_batch and cell_seed's docstrings).
                # No contrastive (mean-subtracted) preprocessing on the hidden
                # states -- see plan S4.4: the reference repo's own
                # bridge-entity layer sweep
                # (evals/bridge_entity/run_selfie_bridge_extraction.py)
                # injects raw hidden states at every layer, including 19, so
                # this sweep does too.
                torch.manual_seed(cell_seed(arm.value, word, config.sample_start))
                # The keys are the extraction's own, not config.positions: only
                # the extraction knows what USER_PROMPT_SPAN expanded to. Each
                # arrives as its last generation is drawn, so a group's cells
                # reach disk during the group, not after it.
                for (layer, position), generations in generate_interpretations_batch(
                    peft_model,
                    tokenizer,
                    adapter,
                    extraction.hidden_states,
                    config.n_samples,
                    config.max_new_tokens,
                    config.temperature,
                    config.device,
                    config.batch_size,
                ):
                    key = (arm.value, word, layer, position_key(position))
                    cell = score_cell(generations, word)
                    append_cell(
                        handle,
                        dict(
                            zip(KEY_FIELDS, key),
                            generations=cell.generations,
                            hits=cell.hits,
                            hit_rate=cell.hit_rate,
                        ),
                    )
    return cells_path


def main(args) -> Path:
    """Load, sweep, write.

    :param args: parsed command-line arguments
    :return: path to this run's cells file; its metadata sidecar sits beside it
    """
    from adapter_training.inference import load_adapter
    from transformers import AutoConfig

    from config import BASE_MODEL_8B, Arm, resolve_layers, sweep_config
    from model_loading import (
        attach_taboo_loras,
        load_base_model,
        load_tokenizer,
    )
    from preflight import preflight

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Layer count comes from the model's own config, not an assumed 32 (plan
    # S2: "reported elsewhere as 32 ... but treat that as unverified until
    # the preflight check confirms it").
    model_name = args.model or BASE_MODEL_8B
    num_hidden_layers = AutoConfig.from_pretrained(model_name).num_hidden_layers
    kwargs = {
        name: value
        for name, value in (
            ("base_model", args.model),
            ("adapter_path", args.adapter_path),
            ("taboo_lora_repo_template", args.lora_template),
            ("n_samples", args.n_samples),
            ("batch_size", args.batch_size),
            ("max_new_tokens", args.max_new_tokens),
            ("temperature", args.temperature),
        )
        if value is not None
    }
    config = sweep_config(
        args.words.split(","),
        layers=resolve_layers(args.layers, num_hidden_layers),
        arms=args.arms,
        positions=args.positions,
        output_dir=output_dir,
        sample_start=args.sample_start,
        device=args.device,
        **kwargs,
    )
    tokenizer = load_tokenizer(config.base_model)
    preflight(config, tokenizer, num_hidden_layers)
    model = load_base_model(config.base_model, device=args.device, dtype=args.dtype)
    adapter = load_adapter(config.adapter_path, device=args.device)

    # Attach every word's taboo LoRA, downloaded from the Hub.
    peft_model = (
        attach_taboo_loras(model, config.words, config.taboo_lora_repo_template)
        if Arm.FINETUNED in config.arms
        else model
    )

    # One pair of files per shard, named by its sample range, so shards writing
    # into a shared output directory never collide. merge_results.py combines them.
    return run(
        config,
        adapter=adapter,
        tokenizer=tokenizer,
        peft_model=peft_model,
    )


if __name__ == "__main__":
    cells_path = main(parse_args())
    print(f"Wrote {cells_path}, metadata in {cells_path.with_suffix('.json')}")
