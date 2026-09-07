"""Tests for `adapter_training.retrieval_eval.score_sets` (bg_think_many
step 5, `plans/bg_think_many/step5_set_retrieval_eval.md`).

All against a fake index with hand-chosen embeddings, so ranks are known
exactly -- no `sentence_transformers` call, no GPU. `FakeScoringIndex`
duck-types the pieces `score_sets` touches (`.titles`, `.topic_embeddings`,
`._embed_texts`), the same pattern `test_retrieval_eval.py` uses for `score`.
"""

import torch

from adapter_training.retrieval_eval import score, score_sets, split_segments


class FakeScoringIndex:
    def __init__(self, titles, topic_embeddings, description_embeddings):
        self.titles = titles
        self.labels = [[] for _ in titles]
        self.topic_embeddings = topic_embeddings
        self.device = torch.device("cpu")
        self._description_embeddings = description_embeddings

    def _embed_texts(self, texts, show_progress=False):
        return torch.stack([self._description_embeddings[t] for t in texts])


# --- test 1: the k=1 reduction to evaluate_labels/score, exact equality ----


def test_k1_no_semicolon_reduces_exactly_to_score():
    titles = ["A", "B", "C", "D", "E"]
    topic_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.5, 0.5],
            [0.1, 0.9],
            [0.0, 1.0],
        ]
    )
    # q1: A > B > C > D > E, ground truth C sits at rank 3.
    # q2: exactly matches A's own embedding, ground truth A sits at rank 1.
    description_embeddings = {
        "q1": torch.tensor([0.6, 0.4]),
        "q2": torch.tensor([1.0, 0.0]),
    }
    index = FakeScoringIndex(titles, topic_embeddings, description_embeddings)
    descriptions = ["q1", "q2"]
    ground_truth = ["C", "A"]
    k_values = [1, 3, 5]

    reference = score(index, descriptions, ground_truth, k_values=k_values)
    grouped = score_sets(
        index, descriptions, [(t,) for t in ground_truth], k_values=k_values
    )

    assert grouped["recalls"] == reference["recalls"]
    assert grouped["mrr"] == reference["mrr"]
    assert grouped["n_true_topics"] == reference["n_labels"]


# --- test 2: best rank over segments ----------------------------------------


def test_best_rank_is_taken_over_segments():
    titles = ["A", "B", "F1", "F2", "F3"]
    topic_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    # Segment 1 ranks A 1st, B last; segment 2 ranks B 1st, A last.
    description_embeddings = {
        "seg-a-first": torch.tensor([1.0, 0.0, 0.4, 0.3, 0.2]),
        "seg-b-first": torch.tensor([0.0, 1.0, 0.4, 0.3, 0.2]),
    }
    index = FakeScoringIndex(titles, topic_embeddings, description_embeddings)

    result = score_sets(
        index,
        ["seg-a-first;seg-b-first"],
        [("A", "B")],
        k_values=[1],
    )

    assert result["recalls"][1] == 1.0  # both true topics recovered at N=1
    assert result["per_query"][0]["best_ranks"] == [1, 1]


# --- test 3: no ';' and k=3 -- all three true topics scored against one segment


def test_no_semicolon_with_k3_scores_all_three_true_topics_against_the_one_segment():
    titles = ["A", "B", "C", "D"]
    topic_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.0, 1.0],
        ]
    )
    description_embeddings = {"only segment": torch.tensor([0.6, 0.4])}
    index = FakeScoringIndex(titles, topic_embeddings, description_embeddings)

    result = score_sets(
        index, ["only segment"], [("A", "B", "C")], k_values=[1, 2, 3, 4]
    )

    assert result["per_query"][0]["n_segments"] == 1
    assert len(result["per_query"][0]["best_ranks"]) == 3
    assert result["n_true_topics"] == 3
    # A(0.6) > B(0.5) > C(0.44) > D(0.4): ranks 1, 2, 3.
    assert result["per_query"][0]["best_ranks"] == [1, 2, 3]


# --- test 4: seven segments, all scored, histogram records it --------------


def test_seven_segments_are_all_scored_and_counted_in_the_histogram():
    titles = ["A", "B", "C"]
    topic_embeddings = torch.eye(3)
    segments = [f"seg{i}" for i in range(7)]
    description_embeddings = {seg: torch.tensor([1.0, 0.0, 0.0]) for seg in segments}
    index = FakeScoringIndex(titles, topic_embeddings, description_embeddings)

    result = score_sets(index, [";".join(segments)], [("A", "B", "C")], k_values=[1])

    assert result["per_query"][0]["n_segments"] == 7
    assert result["segments"]["histogram"] == {"4+": 1}
    assert result["segments"]["fraction_not_equal_k"] == 1.0  # 7 != k=3


# --- test 5: empty/whitespace-only segments dropped; ";;;" scores 0 --------


def test_split_segments_drops_empty_and_whitespace_only():
    assert split_segments("a; b ;  ; c") == ["a", "b", "c"]
    assert split_segments(";;;") == []
    assert split_segments("  ;  ") == []


def test_all_semicolons_yields_no_segments_and_scores_zero_without_raising():
    titles = ["A", "B"]
    topic_embeddings = torch.eye(2)
    index = FakeScoringIndex(titles, topic_embeddings, {})

    result = score_sets(index, [";;;"], [("A",)], k_values=[1, 2])

    assert result["per_query"][0]["n_segments"] == 0
    assert result["per_query"][0]["best_ranks"] == [None]
    assert result["recalls"] == {1: 0.0, 2: 0.0}
    assert result["mrr"] == 0.0


# --- test 6: rank convention matches evaluate_labels on a tie --------------


def test_rank_convention_matches_score_on_a_tie():
    titles = ["A", "B", "C"]
    topic_embeddings = torch.eye(3)
    # A and B tie exactly; ground truth C is strictly lower than both.
    description_embeddings = {"q": torch.tensor([0.7, 0.7, 0.3])}
    index = FakeScoringIndex(titles, topic_embeddings, description_embeddings)

    reference = score(index, ["q"], ["C"], k_values=[1, 2, 3])
    grouped = score_sets(index, ["q"], [("C",)], k_values=[1, 2, 3])

    assert grouped["recalls"] == reference["recalls"]
    assert grouped["per_query"][0]["best_ranks"] == [
        reference["per_label_results"][0]["rank"]
    ]


# --- test 7: per-k reporting keeps k slices separate ------------------------


def test_per_k_reporting_keeps_k_slices_separate():
    titles = ["A", "B", "C"]
    topic_embeddings = torch.eye(3)
    description_embeddings = {
        "q1": torch.tensor([1.0, 0.0, 0.0]),
        "q2": torch.tensor([0.0, 1.0, 0.0]),
    }
    index = FakeScoringIndex(titles, topic_embeddings, description_embeddings)

    result_k1 = score_sets(index, ["q1"], [("A",)], k_values=[1])
    result_k2 = score_sets(index, ["q2"], [("A", "B")], k_values=[1])

    assert result_k1["k"] == 1
    assert result_k2["k"] == 2


def test_score_sets_refuses_mixed_k_across_queries():
    titles = ["A", "B", "C"]
    topic_embeddings = torch.eye(3)
    description_embeddings = {"q1": torch.tensor([1.0, 0.0, 0.0])}
    index = FakeScoringIndex(titles, topic_embeddings, description_embeddings)

    import pytest

    with pytest.raises(ValueError, match="mixed k"):
        score_sets(index, ["q1", "q1"], [("A",), ("A", "B")], k_values=[1])
