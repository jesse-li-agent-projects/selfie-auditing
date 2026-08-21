"""CLI entry point: wires model loading -> extraction -> interpretation -> scoring.

    python -m selfie_taboo.run_pipeline --smoke --output-dir smoke_results/
    python -m selfie_taboo.run_pipeline --word gold --output-dir results/gold/

The --smoke path (plan S6) swaps in Llama-3.2-1B-Instruct and a stub adapter
so the whole pipeline can be exercised locally without the real 8B model or
adapter weights. It validates shapes/plumbing only -- see
smoke/small_llama_config.py for exactly what it does and doesn't cover.
"""

import argparse


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

    Returns a nested dict: results[arm][word][layer][position] -> a
    scoring.CellResult-shaped dict, including every raw generation (plan S4.6:
    never keep only the aggregate rate).
    """
    from selfie_taboo.extract import (
        cache_path,
        extract_hidden_states,
        save_hidden_states,
    )
    from selfie_taboo.interpret import generate_interpretations, make_contrastive
    from selfie_taboo.model_loading import arm_active, system_prompt_for
    from selfie_taboo.scoring import score_cell

    results: dict = {}
    for arm in config.arms:
        results[arm.value] = {}
        for word in config.words:
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
                save_hidden_states(
                    cache_path(config.output_dir, arm, word), hidden_states
                )

                word_results: dict = {}
                for layer in config.layers:
                    word_results[layer] = {}
                    for position in config.positions:
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
                        word_results[layer][position.value] = {
                            "generations": cell.generations,
                            "hits": cell.hits,
                            "hit_rate": cell.hit_rate,
                        }
                results[arm.value][word] = word_results
    return results


if __name__ == "__main__":
    args = parse_args()

    import json
    from pathlib import Path

    from selfie_taboo.config import Arm, first_pass_config
    from selfie_taboo.model_loading import (
        attach_taboo_loras,
        load_base_model,
        load_tokenizer,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        from smoke.small_llama_config import IdentityAdapter, smoke_config

        config = smoke_config(output_dir)
        tokenizer = load_tokenizer(config.base_model)
        model = load_base_model(config.base_model, device=args.device, dtype=args.dtype)
        peft_model = model  # no LoRA arm in the smoke config -- see its docstring
        adapter = IdentityAdapter()
        mean_vector = (
            None  # smoke config has no real mean vectors; skip contrastive step
        )
    else:
        from transformers import AutoConfig

        from selfie_taboo.config import BASE_MODEL_8B
        from selfie_taboo.interpret import load_mean_vector, load_wikipedia_adapter

        # Layer count comes from the model's own config, not an assumed 32
        # (plan S2: "reported elsewhere as 32 ... but treat that as unverified
        # until the preflight check confirms it").
        num_hidden_layers = AutoConfig.from_pretrained(BASE_MODEL_8B).num_hidden_layers
        config = first_pass_config(
            args.word, num_hidden_layers=num_hidden_layers, output_dir=output_dir
        )
        tokenizer = load_tokenizer(config.base_model)
        model = load_base_model(config.base_model, device=args.device, dtype=args.dtype)
        peft_model = (
            attach_taboo_loras(model, config.words, config.taboo_lora_repo_template)
            if Arm.FINETUNED in config.arms
            else model
        )
        adapter = load_wikipedia_adapter(
            config.adapter_repo, config.adapter_filename, args.device
        )
        mean_vector = load_mean_vector(config.adapter_repo, config.mean_vector_layer)

    results = run(
        config,
        adapter=adapter,
        mean_vector=mean_vector,
        tokenizer=tokenizer,
        peft_model=peft_model,
    )

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to {results_path}")
