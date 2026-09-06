"""Grouped extraction logic tests (bg_think_many step 2).

The prompt tests pin the wording the whole k=1/k=2/k=3 comparison rests on;
the rest drive the extractor with the same fake model and tokenizer the
single-topic tests use, which is what lets them assert on the grouping, the
index arithmetic and the records without weights.
"""

import json
import sys
from unittest import mock

import pytest
import torch

from adapter_training.dataset import (
    GroupRecord,
    load_group_records,
    load_vector_store,
)
from adapter_training.extract_grouped_vectors import (
    build_groups,
    parse_args,
    drop_semicolon_topics,
    extract_grouped_vectors,
    write_outputs,
)
from adapter_training.pangram_extraction import PANGRAM, response_token_ids
from conftest import FakeModel, FakeTokenizer
from prompts import PANGRAM_PROMPT_TEMPLATE, background_topics_prompt


def record(title, split="train", labels=None):
    from adapter_training.dataset import TopicRecord

    return TopicRecord(
        title=title,
        labels=tuple(labels or (f"{title} label 0", f"{title} label 1")),
        split=split,
        start=0,
        count=0,
    )


# --- the prompt ------------------------------------------------------------


def test_one_topic_prompt_is_byte_identical_to_the_single_topic_template():
    # What makes the existing single-topic extraction reusable as the k=1
    # population (parent plan D1). If an edit breaks this, the k=1 slice
    # quietly becomes a different experiment.
    assert background_topics_prompt(
        PANGRAM, ["Alpha"]
    ) == PANGRAM_PROMPT_TEMPLATE.format(pangram=PANGRAM, topic="Alpha")


def test_two_topics_are_joined_with_and():
    assert background_topics_prompt(PANGRAM, ["Alpha", "Bravo"]) == (
        'Write "The quick brown fox jumps over the lazy dog." Think about the '
        'topics "Alpha" and "Bravo" while writing the sentence. Do not write '
        "anything else or change the words."
    )


def test_three_topics_take_an_oxford_comma():
    assert background_topics_prompt(PANGRAM, ["Alpha", "Bravo", "Charlie"]) == (
        'Write "The quick brown fox jumps over the lazy dog." Think about the '
        'topics "Alpha", "Bravo", and "Charlie" while writing the sentence. Do '
        "not write anything else or change the words."
    )


@pytest.mark.parametrize("titles", [[], ["A", "B", "C", "D"]])
def test_prompt_refuses_a_list_it_has_no_wording_for(titles):
    with pytest.raises(ValueError):
        background_topics_prompt(PANGRAM, titles)


# --- the topic pool --------------------------------------------------------


def test_semicolon_labels_drop_the_whole_topic():
    topics = [
        record("Alpha"),
        record("Frankenstein", labels=("Frankenstein; or, The Modern Prometheus",)),
        record("Charlie"),
    ]

    kept, dropped = drop_semicolon_topics(topics)

    assert [topic.title for topic in kept] == ["Alpha", "Charlie"]
    assert dropped == ["Frankenstein"]


# --- grouping --------------------------------------------------------------


def pool(n, split="train"):
    return [record(f"T{i:03d}", split=split) for i in range(n)]


def test_groups_never_mix_splits():
    topics = pool(9, "train") + pool(6, "val")

    groups = build_groups(topics, k=3, rounds=2, seed=0)

    assert all(len({topic.split for topic in group}) == 1 for group in groups)
    assert sum(group[0].split == "train" for group in groups) == 6
    assert sum(group[0].split == "val" for group in groups) == 4


def test_one_round_is_a_disjoint_partition():
    topics = pool(12)

    groups = build_groups(topics, k=3, rounds=1, seed=0)

    titles = [topic.title for group in groups for topic in group]
    assert sorted(titles) == sorted(topic.title for topic in topics)


def test_rounds_put_every_topic_in_exactly_that_many_groups():
    topics = pool(12)

    groups = build_groups(topics, k=3, rounds=2, seed=0)

    counts: dict[str, int] = {}
    for group in groups:
        for topic in group:
            counts[topic.title] = counts.get(topic.title, 0) + 1
    assert set(counts.values()) == {2}


def test_leftovers_shorter_than_k_are_dropped_per_round():
    topics = pool(11)  # 3 groups of 3, 2 topics left over

    groups = build_groups(topics, k=3, rounds=2, seed=0)

    assert len(groups) == 6
    assert all(len(group) == 3 for group in groups)


def test_k_one_rounds_are_duplicates_and_the_cli_rejects_them():
    # At k=1 every round is the same partition into singletons, so extra
    # rounds buy no new combinations -- they just extract each prompt again.
    groups = build_groups(pool(10), k=1, rounds=2, seed=0)
    assert len(groups) == 20
    assert len({group[0].title for group in groups}) == 10

    argv = [
        "prog",
        "--k",
        "1",
        "--rounds",
        "2",
        "--output-dir",
        "d",
        "--source-topics",
        "s",
    ]
    with mock.patch.object(sys, "argv", argv), pytest.raises(SystemExit):
        parse_args()


def test_grouping_is_deterministic_and_seed_dependent():
    topics = pool(30)

    def titles(seed):
        return [
            tuple(topic.title for topic in group)
            for group in build_groups(topics, k=3, rounds=2, seed=seed)
        ]

    assert titles(0) == titles(0)
    assert titles(0) != titles(1)
    # Two rounds with the same seed must still be two different partitions,
    # or the second round buys nothing.
    rounds = titles(0)
    assert rounds[:10] != rounds[10:]


# --- extraction ------------------------------------------------------------


def test_grouped_extraction_records_address_their_own_vectors():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)
    groups = [(record("Alpha"), record("Bravo")), (record("Charlie"), record("Delta"))]
    sentence_ids, _ = response_token_ids(tokenizer, PANGRAM + ".")

    result = extract_grouped_vectors(model, tokenizer, groups, layer=1, device="cpu")

    assert [r.titles for r in result.records] == [
        ("Alpha", "Bravo"),
        ("Charlie", "Delta"),
    ]
    assert result.vectors.shape == (2 * len(sentence_ids), 8)
    first, second = result.records
    assert (first.start, first.count) == (0, len(sentence_ids))
    assert second.start == first.count
    # The fake's hidden_states[L + 1] is (token id + L + 1) in every channel,
    # so this asserts each record's range holds the sentence tokens of its own
    # forward pass, not a neighbour's.
    for record_ in result.records:
        rows = result.vectors[record_.start : record_.start + record_.count, 0]
        assert rows.float().tolist() == [float(i + 2) for i in sentence_ids]


def test_grouped_extraction_carries_labels_and_split_per_topic():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)
    groups = [(record("Alpha", split="val"), record("Bravo", split="val"))]

    result = extract_grouped_vectors(model, tokenizer, groups, layer=1, device="cpu")

    (group,) = result.records
    assert group.split == "val"
    assert group.labels_per_topic == (
        ("Alpha label 0", "Alpha label 1"),
        ("Bravo label 0", "Bravo label 1"),
    )


def test_a_rejected_group_is_reported_by_its_titles_and_leaves_no_gap():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, fail_titles={"Bravo"})
    groups = [(record("Alpha"), record("Bravo")), (record("Charlie"), record("Delta"))]

    result = extract_grouped_vectors(model, tokenizer, groups, layer=1, device="cpu")

    assert [r.titles for r in result.records] == [("Charlie", "Delta")]
    assert result.n_seen == 2
    assert [failure["titles"] for failure in result.failures] == [["Alpha", "Bravo"]]
    assert result.records[0].start == 0


# --- the output directory --------------------------------------------------


def extract_to(tmp_path, groups, k=2):
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)
    result = extract_grouped_vectors(model, tokenizer, groups, layer=1, device="cpu")
    write_outputs(tmp_path, result, layer=1, model_name="fake", k=k, rounds=2)
    return result


def test_records_round_trip_through_groups_json(tmp_path):
    groups = [(record("Alpha"), record("Bravo")), (record("Charlie"), record("Delta"))]

    result = extract_to(tmp_path, groups)

    assert not (tmp_path / "topics.json").exists()
    assert load_group_records(tmp_path) == list(result.records)
    assert all(isinstance(r, GroupRecord) for r in load_group_records(tmp_path))


def test_positions_json_records_the_grouped_style_and_k(tmp_path):
    extract_to(tmp_path, [(record("Alpha"), record("Bravo"))], k=2)

    with open(tmp_path / "positions.json") as handle:
        positions = json.load(handle)

    assert positions["prompt_style"] == "grouped"
    assert (positions["k"], positions["rounds"]) == (2, 2)
    assert positions["n_groups"] == 1


def test_filter_report_counts_groups(tmp_path):
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, fail_titles={"Bravo"})
    groups = [
        (record("Alpha"), record("Bravo")),
        (record("Charlie"), record("Delta", split="train")),
    ]
    result = extract_grouped_vectors(model, tokenizer, groups, layer=1, device="cpu")
    write_outputs(tmp_path, result, layer=1, model_name="fake", k=2, rounds=2)

    with open(tmp_path / "filter_report.json") as handle:
        report = json.load(handle)

    assert (report["groups_seen"], report["groups_kept"]) == (2, 1)
    assert report["keep_rate"] == 0.5
    assert report["labels_kept"] == 4  # two topics, two labels each
    assert report["train_groups"] == 1
    assert sum(report["variant_counts"].values()) == 1


def test_load_vector_store_centres_a_grouped_directory(tmp_path):
    groups = [(record("Alpha"), record("Bravo")), (record("Charlie"), record("Delta"))]
    result = extract_to(tmp_path, groups)
    means = torch.load(tmp_path / "position_means.pt", weights_only=True)

    store = load_vector_store(tmp_path, records=load_group_records(tmp_path))

    for group in load_group_records(tmp_path):
        rows = store.vectors[group.start : group.start + group.count]
        raw = result.vectors[group.start : group.start + group.count].float()
        assert torch.allclose(rows, raw - means[: group.count], atol=1e-2)


def test_a_directory_holding_both_records_files_is_refused(tmp_path):
    # A grouped run into a directory that already held a single-topic one
    # leaves the stale topics.json beside the new groups.json. Guessing
    # between them would centre with another run's start/count ranges.
    extract_to(tmp_path, [(record("Alpha"), record("Bravo"))])
    (tmp_path / "topics.json").write_text("[]")

    with pytest.raises(ValueError, match="incompatibly"):
        load_vector_store(tmp_path)


def test_a_grouped_directory_is_not_readable_as_a_single_topic_one(tmp_path):
    # The whole point of writing groups.json instead of topics.json: a reader
    # that does not know about groups must fail, not silently mis-centre.
    extract_to(tmp_path, [(record("Alpha"), record("Bravo"))])

    with pytest.raises(FileNotFoundError):
        load_vector_store(tmp_path)
