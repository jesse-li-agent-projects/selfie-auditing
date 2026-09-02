"""Print the token map for an elicitation prompt, without loading any model.

explore_selfie.py's --token-index addresses tokens by the index printed here.
This script only tokenizes the formatted prompt -- no model or adapter load --
so it's the fast way to pick a --token-index before spending time on a full
run. (run_pipeline.py has no --token-index: it sweeps whole position lists
declared in config.py, not one token at a time.)
"""

import argparse

# config is deliberately free of heavy imports, so --help stays fast.
from config import ModelOrganism


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--word", required=True, help="Secret word for the organism's setup")
    parser.add_argument(
        "--organism",
        default="finetuned",
        choices=[a.value for a in ModelOrganism],
        help="Experimental organism (default: finetuned)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Extraction prompt (default: config.SECRET_PROMPT)",
    )
    parser.add_argument("--model", default=None, help="Base model repo (default: 8B)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from config import BASE_MODEL_8B, SECRET_PROMPT
    from extract import build_prompt
    from model_loading import load_tokenizer, system_prompt_for
    from token_map import print_token_map

    organism = ModelOrganism(args.organism)
    prompt = args.prompt if args.prompt is not None else SECRET_PROMPT
    model_name = args.model or BASE_MODEL_8B

    tokenizer = load_tokenizer(model_name)
    system_prompt = system_prompt_for(organism, args.word)
    print(f"[validate] organism={organism.value} word={args.word!r} prompt={prompt!r}")
    print(f"[validate] system_prompt={system_prompt!r}")
    print_token_map(tokenizer, build_prompt(tokenizer, prompt, system_prompt))
