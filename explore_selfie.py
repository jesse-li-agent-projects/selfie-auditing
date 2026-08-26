"""Read the raw SelfIE interpretations for one cell, without running a sweep.

A sanity check on the interpretation half of the pipeline: pick a model, an
arm, a layer (or a few), and just print what SelfIE says about that hidden
state. Use it to answer "does this produce sensible English at all?" before
spending a full run's generation budget in run_pipeline.py.

Which token the hidden state comes from matters as much as the layer: the two
named --position values are both late-prompt tokens, and the secret is not
always still legible there. --token-index reads any token instead, addressed by
the token map printed on every run.

The extraction prompt defaults to config.SECRET_PROMPT; --prompt overrides it,
which is the easiest way to check that interpretations track the content of the
hidden state (a prompt about Paris should read back as something about Paris).

The published SelfIE adapter and taboo LoRAs exist for Llama-3.1-8B-Instruct
only, so a smaller --model needs weights of its own width. Point --adapter-repo
/ --adapter-filename / --lora-template at a dummy weight set (see
make_smoke_weights.py) for a shape-correctness run; its generations carry no
meaning.
"""

import argparse

# config and scoring are deliberately free of heavy imports, so --help stays fast.
from config import Arm, Position
from scoring import score_cell


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--word", required=True, help="Secret word for the arm's setup")
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        required=True,
        help="Layer(s) to interpret, 0-indexed transformer layers",
    )
    parser.add_argument(
        "--arm",
        default="finetuned",
        choices=[a.value for a in Arm],
        help="Experimental arm (default: finetuned)",
    )
    position_group = parser.add_mutually_exclusive_group()
    position_group.add_argument(
        "--position",
        default=None,
        choices=[p.value for p in Position],
        help="Named token position to read the hidden state from (default: assistant_boundary)",
    )
    position_group.add_argument(
        "--token-index",
        type=int,
        default=None,
        help="Raw token index in the formatted prompt, for positions no name covers. Negative values index from the end. Every run prints the token map to pick from.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Extraction prompt (default: config.SECRET_PROMPT)",
    )
    parser.add_argument(
        "-n", "--n-samples", type=int, default=10, help="Generations per layer"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=50)
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
        help="Taboo LoRA path/repo template containing {word} (default: the 8B repos)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def print_cell(layer: int, generations: list[str], word: str) -> None:
    """Print every generation for one layer, marking secret-word hits."""
    cell = score_cell(generations, word)
    print(
        f"\n=== layer {layer} -- hit rate {cell.hit_rate:.0%} "
        f"({sum(cell.hits)}/{cell.n}) ==="
    )
    for i, (text, hit) in enumerate(zip(cell.generations, cell.hits)):
        print(f"  [{i:>3}]{' *' if hit else '  '} {text!r}")


if __name__ == "__main__":
    args = parse_args()

    from huggingface_hub import hf_hub_download
    from selfie_adapters import load_adapter

    from config import (
        BASE_MODEL_8B,
        SECRET_PROMPT,
        SELFIE_ADAPTER_FILE,
        SELFIE_ADAPTER_REPO,
        TABOO_LORA_REPO_TEMPLATE,
    )
    from extract import build_prompt, extract_hidden_states
    from interpret import generate_interpretations
    from model_loading import (
        arm_active,
        attach_taboo_loras,
        load_base_model,
        load_tokenizer,
        system_prompt_for,
    )
    from token_map import print_token_map

    arm = Arm(args.arm)
    position: Position | int = (
        args.token_index
        if args.token_index is not None
        else Position(args.position or Position.ASSISTANT_BOUNDARY)
    )
    prompt = args.prompt if args.prompt is not None else SECRET_PROMPT

    model_name = args.model or BASE_MODEL_8B
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, device=args.device, dtype=args.dtype)
    # The taboo LoRA is only loaded for the arm that needs it -- the other arms
    # differ from the base model only by their system prompt.
    if arm is Arm.FINETUNED:
        lora_template = args.lora_template or TABOO_LORA_REPO_TEMPLATE
        model = attach_taboo_loras(model, [args.word], lora_template)
    adapter = load_adapter(
        hf_hub_download(
            repo_id=args.adapter_repo or SELFIE_ADAPTER_REPO,
            filename=args.adapter_filename or SELFIE_ADAPTER_FILE,
        ),
        device=args.device,
    )

    with arm_active(model, arm, args.word):
        system_prompt = system_prompt_for(arm, args.word)
        print(f"[validate] arm={arm.value} word={args.word!r} prompt={prompt!r}")
        print(f"[validate] system_prompt={system_prompt!r}")
        print_token_map(tokenizer, build_prompt(tokenizer, prompt, system_prompt))
        extraction = extract_hidden_states(
            model,
            tokenizer,
            prompt,
            system_prompt,
            args.layer,
            [position],
            args.device,
        )
        for layer in args.layer:
            generations = generate_interpretations(
                model,
                tokenizer,
                adapter,
                extraction.hidden_states[(layer, position)],
                args.n_samples,
                args.max_new_tokens,
                args.temperature,
                args.device,
            )
            print_cell(layer, generations, args.word)
