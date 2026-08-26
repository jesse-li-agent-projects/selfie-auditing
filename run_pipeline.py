"""CLI entry point: wires model loading -> extraction -> interpretation -> scoring.

    python run_pipeline.py --smoke --output-dir smoke_results/
    python run_pipeline.py --words gold,moon --output-dir results/sweep/

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

The --smoke path (plan S6) swaps in Llama-3.2-1B-Instruct and random-weight
stand-in weights so the whole pipeline can be exercised locally without the
real 8B model or adapter weights. It validates shapes/plumbing only -- see
smoke/small_llama_config.py for exactly what it does and doesn't cover.
"""

import argparse
import hashlib
from itertools import product
from pathlib import Path

# Light import: config.py carries no heavy dependencies, so parsing --arms and
# --positions doesn't cost --help its speed.
from config import Arm, Position


def parse_arms(spec: str) -> list[Arm]:
    """Parse `--arms`: a comma-separated list of arm names."""
    return [Arm(name) for name in spec.split(",")]


def parse_positions(spec: str) -> list[Position | int]:
    """Parse `--positions`: a comma-separated list of position names and/or
    raw token offsets (see `extract.resolve_position`)."""
    positions: list[Position | int] = []
    for token in spec.split(","):
        try:
            positions.append(Position(token))
        except ValueError:
            positions.append(int(token))
    return positions


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the local smoke pass (S6) instead of a real pass",
    )
    parser.add_argument(
        "--words", required=True, help="Comma-separated secret words to sweep"
    )
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--model", default=None, help="Base model repo (default: 8B)")
    parser.add_argument(
        "--adapter-repo",
        default=None,
        help="SelfIE adapter repo on the Hub (default: the 8B one)",
    )
    parser.add_argument(
        "--adapter-filename",
        default=None,
        help="SelfIE adapter filename within --adapter-repo (default: the 8B one)",
    )
    parser.add_argument(
        "--lora-template",
        default=None,
        help="Taboo LoRA repo/path template containing {word} (default: the 8B repos)",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help="Comma-separated arms to sweep (default: control,prompted,finetuned)",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="'all' or a comma-separated list of 0-indexed layers (default: all)",
    )
    parser.add_argument(
        "--positions",
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
        help="Generations per forward pass (default: the config's own)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def cell_seed(arm: str, word: str, layer: int, position: str, sample_start: int) -> int:
    """Deterministic per-cell, per-shard seed.

    blake2b rather than hash(): Python's hash() is salted per process, so a
    hash()-derived seed would give a different generation stream on every run
    and silently break replay. Folding in `sample_start` is what keeps two
    shards of one cell from regenerating the same samples -- without it a
    "200-sample" cell would really be 100 samples counted twice.

    :param arm: the experimental condition
    :param word: the secret word
    :param layer: the transformer layer index
    :param position: the position key (see `extract.position_key`)
    :param sample_start: index of this shard's first generation
    :return: a seed in [0, 2**31)
    """
    digest = hashlib.blake2b(
        f"{arm}|{word}|{layer}|{position}|{sample_start}".encode(), digest_size=8
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
    from interpret import generate_interpretations
    from model_loading import arm_active, system_prompt_for
    from results_store import (
        KEY_FIELDS,
        append_cell,
        shard_cells_path,
        write_metadata,
    )
    from scoring import score_cell

    def cell_result(hidden_state, word) -> dict:
        # No contrastive (mean-subtracted) preprocessing here -- see plan S4.4:
        # the reference repo's own bridge-entity layer sweep
        # (evals/bridge_entity/run_selfie_bridge_extraction.py) injects raw
        # hidden states at every layer, including 19, so this sweep does too.
        generations = generate_interpretations(
            peft_model,
            tokenizer,
            adapter,
            hidden_state,
            config.n_samples,
            config.max_new_tokens,
            config.temperature,
            config.device,
            config.batch_size,
        )
        cell = score_cell(generations, word)
        return {
            "generations": cell.generations,
            "hits": cell.hits,
            "hit_rate": cell.hit_rate,
        }

    spans: dict[str, dict[str, str]] = {}
    sample_end = config.sample_start + config.n_samples
    cells_path = shard_cells_path(config.output_dir, config.sample_start, sample_end)

    def metadata() -> dict:
        return {
            "sample_range": [config.sample_start, sample_end],
            "batch_size": config.batch_size,
            "secret_prompt": config.secret_prompt,
            "spans": spans,
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
                # Iterate the extraction's own keys, not config.positions: only
                # the extraction knows what USER_PROMPT_SPAN expanded to.
                for (layer, position), hidden in extraction.hidden_states.items():
                    key = (arm.value, word, layer, position_key(position))
                    torch.manual_seed(cell_seed(*key, config.sample_start))
                    append_cell(
                        handle,
                        dict(
                            zip(KEY_FIELDS, key),
                            **cell_result(hidden, word),
                        ),
                    )
    return cells_path


if __name__ == "__main__":
    args = parse_args()

    from transformers import AutoConfig

    from config import Arm
    from model_loading import (
        attach_taboo_loras,
        load_base_model,
        load_tokenizer,
    )
    from preflight import preflight

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        from selfie_adapters import load_adapter

        from smoke.small_llama_config import (
            SMOKE_ADAPTER_FILENAME,
            SMOKE_MODEL,
            create_random_lora,
            create_random_selfie_adapter,
            embedding_norm,
            smoke_config,
        )

        num_hidden_layers = AutoConfig.from_pretrained(SMOKE_MODEL).num_hidden_layers
        config = smoke_config(output_dir, num_hidden_layers=num_hidden_layers)
        if args.n_samples is not None:
            config.n_samples = args.n_samples
        config.sample_start = args.sample_start
        config.device = args.device
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        tokenizer = load_tokenizer(config.base_model)
        preflight(config, tokenizer, num_hidden_layers)
        model = load_base_model(config.base_model, device=args.device, dtype=args.dtype)
        smoke_lora_baseline = None
        if Arm.FINETUNED in config.arms:
            # Captured before create_random_lora() wraps the model, so the
            # self-check below can confirm unload() hands back a genuinely
            # clean base model, not just one that "looks" clean.
            from extract import extract_hidden_states as _extract_baseline

            smoke_lora_baseline = _extract_baseline(
                model,
                tokenizer,
                config.secret_prompt,
                None,
                [config.layers[0]],
                [config.positions[0]],
                args.device,
            ).hidden_states[(config.layers[0], config.positions[0])]

            # No real taboo LoRA exists at 1B scale -- generate a random-init
            # one (same hyperparams as the real ones) and save it where
            # attach_taboo_loras() below expects to find it, so the FINETUNED
            # arm exercises the exact same load path the real 8B run uses.
            model = create_random_lora(
                model, config.taboo_lora_repo_template, config.words[0]
            )
        # A random-weight checkpoint in the real on-disk format, loaded through
        # the ordinary load_adapter() path, rather than a stub object: this is
        # what makes the smoke run exercise the adapter loader, its dimension
        # check and its projection math. The weights are untrained either way,
        # so it still says nothing about whether the sweep finds anything.
        adapter = load_adapter(
            str(
                create_random_selfie_adapter(
                    model.config.hidden_size,
                    output_dir / SMOKE_ADAPTER_FILENAME,
                    embedding_norm(model),
                )
            ),
            device=args.device,
        )
    else:
        from huggingface_hub import hf_hub_download
        from selfie_adapters import load_adapter

        from config import BASE_MODEL_8B, resolve_layers, sweep_config

        # Layer count comes from the model's own config, not an assumed 32
        # (plan S2: "reported elsewhere as 32 ... but treat that as unverified
        # until the preflight check confirms it").
        model_name = args.model or BASE_MODEL_8B
        num_hidden_layers = AutoConfig.from_pretrained(model_name).num_hidden_layers
        kwargs = {
            name: value
            for name, value in (
                ("base_model", args.model),
                ("adapter_repo", args.adapter_repo),
                ("adapter_filename", args.adapter_filename),
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
            arms=parse_arms(args.arms) if args.arms else None,
            positions=parse_positions(args.positions) if args.positions else None,
            output_dir=output_dir,
            sample_start=args.sample_start,
            device=args.device,
            **kwargs,
        )
        tokenizer = load_tokenizer(config.base_model)
        preflight(config, tokenizer, num_hidden_layers)
        model = load_base_model(config.base_model, device=args.device, dtype=args.dtype)
        adapter = load_adapter(
            hf_hub_download(
                repo_id=config.adapter_repo, filename=config.adapter_filename
            ),
            device=args.device,
        )

    # Shared by both paths: attach every word's taboo LoRA (real, downloaded
    # from HF, or smoke's freshly generated random one -- either way saved to
    # disk at config.taboo_lora_repo_template by this point) via the same
    # PeftModel.from_pretrained() load path.
    peft_model = (
        attach_taboo_loras(model, config.words, config.taboo_lora_repo_template)
        if Arm.FINETUNED in config.arms
        else model
    )

    if args.smoke and smoke_lora_baseline is not None:
        # Self-check, not a demonstration: confirms the random LoRA actually
        # perturbs the forward pass when active, and that disable_adapter()
        # gives back the same result as the pre-wrap base model. Without
        # this, a bug where set_adapter()/disable_adapter() silently no-ops
        # (or, as happened once during development, a zero-initialized
        # lora_B making the "random" adapter an exact no-op) would leave
        # every arm producing plausible output while testing nothing.
        from extract import extract_hidden_states

        layer0, position0 = config.layers[0], config.positions[0]
        active = extract_hidden_states(
            peft_model,
            tokenizer,
            config.secret_prompt,
            None,
            [layer0],
            [position0],
            args.device,
        ).hidden_states[(layer0, position0)]
        with peft_model.disable_adapter():
            disabled = extract_hidden_states(
                peft_model,
                tokenizer,
                config.secret_prompt,
                None,
                [layer0],
                [position0],
                args.device,
            ).hidden_states[(layer0, position0)]

        active_vs_disabled = (active - disabled).abs().max().item()
        disabled_vs_baseline = (disabled - smoke_lora_baseline).abs().max().item()
        print(
            f"[smoke] LoRA self-check: active-vs-disabled diff={active_vs_disabled:.4f}, "
            f"disabled-vs-pre-wrap-baseline diff={disabled_vs_baseline:.6f}"
        )
        assert active_vs_disabled > 1e-3, (
            "random LoRA had no measurable effect on the forward pass "
            f"(max diff {active_vs_disabled}) -- likely a no-op adapter (check init_lora_weights)"
        )
        assert disabled_vs_baseline < 1e-3, (
            "disable_adapter() output differs from the pre-wrap base model "
            f"(max diff {disabled_vs_baseline}) -- unload()/disable_adapter() may not be "
            "giving back a clean base model"
        )

    # One pair of files per shard, named by its sample range, so shards writing
    # into a shared output directory never collide. merge_results.py combines them.
    cells_path = run(
        config,
        adapter=adapter,
        tokenizer=tokenizer,
        peft_model=peft_model,
    )
    print(f"Wrote {cells_path}, metadata in {cells_path.with_suffix('.json')}")
