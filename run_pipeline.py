"""CLI entry point: wires model loading -> extraction -> interpretation -> scoring.

    python run_pipeline.py --smoke --output-dir smoke_results/
    python run_pipeline.py --word gold --output-dir results/gold/

The --smoke path (plan S6) swaps in Llama-3.2-1B-Instruct and a stub adapter
so the whole pipeline can be exercised locally without the real 8B model or
adapter weights. It validates shapes/plumbing only -- see
smoke/small_llama_config.py for exactly what it does and doesn't cover.
"""

import argparse
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
        "--word", help="Secret word for a real first-pass run (required unless --smoke)"
    )
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if not args.smoke and not args.word:
        parser.error("--word is required unless --smoke is set")
    return args


def run(config, *, adapter, mean_vector, tokenizer, peft_model) -> dict:
    """Run extraction + interpretation + scoring for every cell in `config`.

    Returns a flat dict keyed by (arm, word, layer, position) -> a
    scoring.CellResult-shaped dict, including every raw generation (plan S4.6:
    never keep only the aggregate rate). See `nest_results` for the
    arm -> word -> layer -> position shape the JSON output uses.
    """
    from extract import (
        cache_path,
        extract_hidden_states,
        save_hidden_states,
    )
    from interpret import generate_interpretations, make_contrastive
    from model_loading import arm_active, system_prompt_for
    from scoring import score_cell

    def cell_result(hidden_states, word, layer, position) -> dict:
        hidden_state = hidden_states[(layer, position)]
        vector = (
            make_contrastive(hidden_state, mean_vector)
            if mean_vector is not None
            else hidden_state
        )
        generations = generate_interpretations(
            peft_model,
            tokenizer,
            adapter,
            vector,
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

    results: dict = {}
    # arm: control/prompt/fine-tuned; word: which word is taboo
    for arm, word in product(config.arms, config.words):
        with arm_active(peft_model, arm, word):
            system_prompt = system_prompt_for(arm, word)
            hidden_states = extract_hidden_states(
                peft_model,
                tokenizer,
                config.secret_prompt,
                system_prompt,
                config.layers,
                config.positions,
                config.device,
            )
            save_hidden_states(cache_path(config.output_dir, arm, word), hidden_states)
            for layer, position in product(config.layers, config.positions):
                key = (arm.value, word, layer, position.value)
                results[key] = cell_result(hidden_states, word, layer, position)
    return results


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

    from config import Arm, first_pass_config
    from model_loading import (
        attach_taboo_loras,
        load_base_model,
        load_tokenizer,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        from smoke.small_llama_config import (
            IdentityAdapter,
            create_random_lora,
            smoke_config,
        )

        config = smoke_config(output_dir)
        tokenizer = load_tokenizer(config.base_model)
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
            )[(config.layers[0], config.positions[0])]

            # No real taboo LoRA exists at 1B scale -- generate a random-init
            # one (same hyperparams as the real ones) and save it where
            # attach_taboo_loras() below expects to find it, so the FINETUNED
            # arm exercises the exact same load path the real 8B run uses.
            model = create_random_lora(
                model, config.taboo_lora_repo_template, config.words[0]
            )
        adapter = IdentityAdapter()
        mean_vector = (
            None  # smoke config has no real mean vectors; skip contrastive step
        )
    else:
        from transformers import AutoConfig

        from config import BASE_MODEL_8B
        from interpret import load_mean_vector, load_wikipedia_adapter

        # Layer count comes from the model's own config, not an assumed 32
        # (plan S2: "reported elsewhere as 32 ... but treat that as unverified
        # until the preflight check confirms it").
        num_hidden_layers = AutoConfig.from_pretrained(BASE_MODEL_8B).num_hidden_layers
        config = first_pass_config(
            args.word, num_hidden_layers=num_hidden_layers, output_dir=output_dir
        )
        tokenizer = load_tokenizer(config.base_model)
        model = load_base_model(config.base_model, device=args.device, dtype=args.dtype)
        adapter = load_wikipedia_adapter(
            config.adapter_repo, config.adapter_filename, args.device
        )
        mean_vector = load_mean_vector(config.adapter_repo, config.mean_vector_layer)

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
        )[(layer0, position0)]
        with peft_model.disable_adapter():
            disabled = extract_hidden_states(
                peft_model,
                tokenizer,
                config.secret_prompt,
                None,
                [layer0],
                [position0],
                args.device,
            )[(layer0, position0)]

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
        mean_vector=mean_vector,
        tokenizer=tokenizer,
        peft_model=peft_model,
    )

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(nest_results(results), f, indent=2)
    print(f"Wrote results to {results_path}")
