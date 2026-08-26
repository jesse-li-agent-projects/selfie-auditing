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

The --smoke path (plan S6) swaps in Llama-3.2-1B-Instruct and random-weight
stand-in weights so the whole pipeline can be exercised locally without the
real 8B model or adapter weights. It validates shapes/plumbing only -- see
smoke/small_llama_config.py for exactly what it does and doesn't cover.
"""

import argparse
import hashlib
from itertools import product


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
        "--words",
        help="Comma-separated secret words to sweep (required unless --smoke)",
    )
    parser.add_argument("--output-dir", required=True, type=str)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if not args.smoke and not args.words:
        parser.error("--words is required unless --smoke is set")
    return args


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


def run(config, *, adapter, tokenizer, peft_model) -> dict:
    """Run extraction + interpretation + scoring for every cell in `config`.

    The prompt and span metadata make the returned document self-describing.
    An offset like -11 only names a token relative to one formatted prompt, so
    recording what it actually resolved to is what keeps a stored result
    interpretable after a prompt or template change, and lets
    merge_results.py check two shards' comparability instead of assuming it.

    :param config: this shard's pipeline config
    :param adapter: SelfIE adapter used to interpret each cell's hidden state
    :param tokenizer: tokenizer shared by extraction and generation
    :param peft_model: the (possibly LoRA-wrapped) model to extract from and generate with
    :return: this shard's sample range, prompt, resolved-position metadata, and
        the nested arm -> word -> layer -> position cells (each keeping every
        raw generation -- plan S4.6: never keep only the aggregate rate)
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
    from scoring import score_cell

    def cell_result(hidden_state, word, layer, position) -> dict:
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
        )
        cell = score_cell(generations, word)
        return {
            "generations": cell.generations,
            "hits": cell.hits,
            "hit_rate": cell.hit_rate,
        }

    flat: dict = {}
    spans: dict[str, dict[str, str]] = {}
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
            if spans.setdefault(arm.value, extraction.tokens) != extraction.tokens:
                raise ValueError(
                    f"arm {arm.value!r} resolved different tokens for word {word!r} "
                    "than for an earlier word -- the arm's cells are not comparable"
                )
            # Iterate the extraction's own keys, not config.positions: only the
            # extraction knows what FULL_USER_SPAN expanded to.
            for (layer, position), hidden_state in extraction.hidden_states.items():
                key = (arm.value, word, layer, position_key(position))
                torch.manual_seed(cell_seed(*key, config.sample_start))
                flat[key] = cell_result(hidden_state, word, layer, position)
    return {
        "sample_range": [config.sample_start, config.sample_start + config.n_samples],
        "secret_prompt": config.secret_prompt,
        "spans": spans,
        "cells": nest_results(flat),
    }


def nest_results(flat: dict) -> dict:
    """Reshape `run`'s flat (arm, word, layer, position) keys into the nested
    arm -> word -> layer -> position dict the JSON results file uses.
    """
    nested: dict = {}
    for (arm, word, layer, position), cell in flat.items():
        nested.setdefault(arm, {}).setdefault(word, {}).setdefault(layer, {})[
            position
        ] = cell
    return nested


if __name__ == "__main__":
    args = parse_args()

    import json
    from pathlib import Path

    from config import Arm, full_sweep_config
    from model_loading import (
        attach_taboo_loras,
        load_base_model,
        load_tokenizer,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        from selfie_adapters import load_adapter

        from smoke.small_llama_config import (
            SMOKE_ADAPTER_FILENAME,
            create_random_lora,
            create_random_selfie_adapter,
            embedding_norm,
            smoke_config,
        )

        config = smoke_config(output_dir)
        tokenizer = load_tokenizer(config.base_model)
        model = load_base_model(config.base_model, device=args.device, dtype=args.dtype)
        if args.n_samples is not None:
            config.n_samples = args.n_samples
        config.sample_start = args.sample_start
        config.device = args.device
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
        from transformers import AutoConfig

        from config import BASE_MODEL_8B
        from interpret import load_wikipedia_adapter

        # Layer count comes from the model's own config, not an assumed 32
        # (plan S2: "reported elsewhere as 32 ... but treat that as unverified
        # until the preflight check confirms it").
        num_hidden_layers = AutoConfig.from_pretrained(BASE_MODEL_8B).num_hidden_layers
        kwargs = {} if args.n_samples is None else {"n_samples": args.n_samples}
        config = full_sweep_config(
            args.words.split(","),
            num_hidden_layers=num_hidden_layers,
            output_dir=output_dir,
            sample_start=args.sample_start,
            device=args.device,
            **kwargs,
        )
        tokenizer = load_tokenizer(config.base_model)
        model = load_base_model(config.base_model, device=args.device, dtype=args.dtype)
        adapter = load_wikipedia_adapter(
            config.adapter_repo, config.adapter_filename, args.device
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

    results = run(
        config,
        adapter=adapter,
        tokenizer=tokenizer,
        peft_model=peft_model,
    )

    # One file per shard, named by its sample range, so shards writing into a
    # shared output directory never collide. merge_results.py combines them.
    start, end = results["sample_range"]
    results_path = output_dir / f"results_{start:06d}_{end:06d}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to {results_path}")
