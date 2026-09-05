from adapter_training.label_complexity import label_buckets


def test_order_is_least_detailed_first():
    labels = ["a very long and detailed description of the topic", "short", "medium length one"]
    buckets = label_buckets(labels, n_buckets=3)
    means = [sum(len(l) for l in b) / len(b) for b in buckets]
    assert means == sorted(means)


def test_partition_property():
    labels = [f"label {i} " + "x" * i for i in range(9)]
    buckets = label_buckets(labels, n_buckets=3)
    recombined = sorted(l for b in buckets for l in b)
    assert recombined == sorted(labels)
    assert len(recombined) == len(labels)


def test_deterministic_under_reordering():
    labels = ["ccc", "a", "bb", "dddd", "ee", "f"]
    shuffled = ["f", "dddd", "a", "ee", "bb", "ccc"]
    assert label_buckets(labels, n_buckets=3) == label_buckets(shuffled, n_buckets=3)


def test_six_label_topic_minimum():
    labels = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]
    buckets = label_buckets(labels, n_buckets=3)
    assert [len(b) for b in buckets] == [2, 2, 2]
    assert buckets[0] == ["a", "bb"]
    assert buckets[2] == ["eeeee", "ffffff"]


def test_twenty_label_topic_maximum():
    labels = [f"{'x' * i}" for i in range(1, 21)]
    buckets = label_buckets(labels, n_buckets=3)
    assert sum(len(b) for b in buckets) == 20
    assert all(len(b) > 0 for b in buckets)
    all_labels = [l for b in buckets for l in b]
    assert sorted(all_labels, key=len) == all_labels


def test_tie_breaking_by_label_text_not_input_order():
    labels = ["bb", "aa", "cc", "dd", "ee", "ff"]  # all same length, ties broken lexically
    reordered = ["ff", "ee", "dd", "cc", "bb", "aa"]
    result = label_buckets(labels, n_buckets=3)
    assert result == label_buckets(reordered, n_buckets=3)
    assert result == [["aa", "bb"], ["cc", "dd"], ["ee", "ff"]]


def test_custom_length_of_is_used():
    labels = ["a", "bb", "ccc"]
    # invert the natural order: length_of maps "a" -> 10, "bb" -> 5, "ccc" -> 1
    length_of = {"a": 10, "bb": 5, "ccc": 1}.get
    buckets = label_buckets(labels, n_buckets=3, length_of=length_of)
    assert buckets == [["ccc"], ["bb"], ["a"]]


def test_every_label_lands_in_exactly_one_bucket():
    labels = ["dup", "unique1", "unique2", "unique3"]
    buckets = label_buckets(labels, n_buckets=2)
    flat = [l for b in buckets for l in b]
    assert sorted(flat) == sorted(labels)
