"""Shared helper: print a formatted prompt as index:token pairs.

Used by explore_token_map.py (which prints nothing else) and explore_selfie.py
(which prints it as a preamble before running the SelfIE sweep), so both agree
on what a given --token-index refers to.
"""


def print_token_map(tokenizer, formatted: str) -> None:
    """Print the formatted prompt as index:token pairs, to pick --token-index from."""
    ids = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).input_ids[
        0
    ]
    print(f"[validate] {len(ids)} prompt tokens:")
    # Packed by hand rather than with textwrap, which would break inside a token
    # repr that contains a space (e.g. 41:' say').
    line = ""
    for i, token_id in enumerate(ids):
        pair = f"{i}:{tokenizer.decode([token_id.item()])!r}"
        if line and len(line) + 1 + len(pair) > 84:
            print(f"    {line}")
            line = pair
        else:
            line = f"{line} {pair}" if line else pair
    if line:
        print(f"    {line}")
