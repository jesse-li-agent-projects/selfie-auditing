"""Pick the TwoHopFact questions the bridge-entity sweep runs on (paper S3.6).

    python filter_bridge_questions.py --output-dir bridge_entity --n-questions 500

A two-hop question only tests bridge-entity recovery if the model actually
routes through the bridge, so the paper keeps questions where the model (1)
names the bridge entity when asked the first hop alone and (2) answers the
two-hop question correctly with no chain of thought -- about 12% of the
dataset. This script asks the model both hops greedily and writes the
questions that pass to `questions.jsonl`, which every arm of the sweep then
shares.

Questions are drawn in a seeded shuffle and the walk stops as soon as
`--n-questions` have passed, so the cost scales with the sample kept rather
than with the 45k-row dataset. Verdicts are appended to `verdicts.jsonl` as
they are decided and reread on restart, so a crash costs at most one batch.
A crash mid-write can leave a truncated final line in that file; delete the
line and rerun.
"""

import argparse
from pathlib import Path

# Light import: config.py pulls in no heavy dependencies, so --help stays fast.
from config import BASE_MODEL_8B, outputs_relative

VERDICTS_FILE = "verdicts.jsonl"
QUESTIONS_FILE = "questions.jsonl"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=outputs_relative,
        default="bridge_entity",
        help="written under outputs/ (implicitly prepended)",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=500,
        help="how many questions to accept before stopping (default: the paper's 500)",
    )
    parser.add_argument("--model", default=BASE_MODEL_8B)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="local TwoHopFact CSV; omit to fetch it from the Hub",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seeds the order questions are tried in"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="questions per generate() call"
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from bridge_dataset import (  # noqa: E402
    BridgeQuestion,
    answer_hop_messages,
    answer_matches,
    bridge_hop_messages,
    read_questions,
    shuffled,
    twohopfact_csv,
    write_question_file,
)
from model_loading import load_base_model, load_tokenizer, resolve_device  # noqa: E402
from results_store import append_cell, read_cells  # noqa: E402


@torch.no_grad()
def answer_greedily(
    model, tokenizer, conversations: list[list[dict[str, str]]], max_new_tokens: int
) -> list[str]:
    """Complete each chat conversation greedily, in one batch.

    Greedy because the filter asks what the model knows, not what it might
    say: a sampled answer would make membership of the question set depend on
    a lucky draw.

    :param conversations: chat message lists, one per question to answer
    :param max_new_tokens: cap on each reply's length
    :return: the reply text for each conversation, in order
    """
    prompts = [
        tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        for conversation in conversations
    ]
    batch = tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(resolve_device(model))
    outputs = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    replies = outputs[:, batch.input_ids.shape[1] :]
    return [
        tokenizer.decode(reply, skip_special_tokens=True).strip() for reply in replies
    ]


def judge(question: BridgeQuestion, bridge_reply: str, answer_reply: str) -> dict:
    """Score one question's two replies into a verdict record."""
    bridge_correct = answer_matches(bridge_reply, question.bridge_aliases)
    answer_correct = answer_matches(answer_reply, question.answer_aliases)
    return {
        "id": question.id,
        "bridge_reply": bridge_reply,
        "answer_reply": answer_reply,
        "bridge_correct": bridge_correct,
        "answer_correct": answer_correct,
        "accepted": bridge_correct and answer_correct,
    }


def previous_verdicts(verdicts_path: Path) -> dict[str, dict]:
    """Verdicts an earlier run already decided, keyed by question id."""
    if not verdicts_path.exists():
        return {}
    return {verdict["id"]: verdict for verdict in read_cells(verdicts_path)}


def run(args, *, model, tokenizer, questions: list[BridgeQuestion]) -> Path:
    """Walk the shuffled dataset until `--n-questions` have been accepted.

    :param questions: the whole dataset, in file order
    :return: path to the written question set
    """
    verdicts_path = args.output_dir / VERDICTS_FILE
    decided = previous_verdicts(verdicts_path)
    accepted: list[BridgeQuestion] = []
    pending: list[BridgeQuestion] = []
    progress = tqdm(total=args.n_questions, desc="accepted questions")

    def flush() -> list[BridgeQuestion]:
        """Decide every pending question, then hand back the accepted ones.

        Questions an earlier run already decided are read from `decided`
        rather than asked again -- that, and the fixed shuffle order, is what
        makes a resumed run rebuild the same set an uninterrupted one would.
        """
        unasked = [question for question in pending if question.id not in decided]
        if unasked:
            conversations = [
                messages(question)
                for question in unasked
                for messages in (bridge_hop_messages, answer_hop_messages)
            ]
            replies = answer_greedily(
                model, tokenizer, conversations, args.max_new_tokens
            )
            with open(verdicts_path, "a") as handle:
                for index, question in enumerate(unasked):
                    verdict = judge(
                        question, replies[2 * index], replies[2 * index + 1]
                    )
                    append_cell(handle, verdict)
                    decided[question.id] = verdict
        passed = [q for q in pending if decided[q.id]["accepted"]]
        pending.clear()
        progress.update(len(passed))
        return passed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for question in shuffled(questions, args.seed):
        # Checked per batch rather than per question, so the walk overshoots
        # by at most one batch -- whose verdicts are still recorded, and so
        # still count towards a later run asking for more questions.
        if len(accepted) >= args.n_questions:
            break
        pending.append(question)
        if len(pending) >= args.batch_size:
            accepted.extend(flush())
    accepted.extend(flush())
    progress.close()

    questions_path = args.output_dir / QUESTIONS_FILE
    write_question_file(questions_path, accepted[: args.n_questions])
    return questions_path


def main(args) -> Path:
    questions = read_questions(twohopfact_csv(args.dataset))
    print(f"Read {len(questions)} TwoHopFact questions")
    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, device=args.device, dtype=args.dtype)
    return run(args, model=model, tokenizer=tokenizer, questions=questions)


if __name__ == "__main__":
    path = main(args)
    kept = sum(1 for _ in open(path))
    print(f"Wrote {kept} accepted questions to {path}")
