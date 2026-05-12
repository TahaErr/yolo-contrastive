"""Tests for duplicate detection and cross-set leakage."""

from __future__ import annotations

from yolo_contrastive.data.dedup.leakage import (
    cross_set_leakage,
    find_exact_duplicates,
    summarize_duplicates,
)


# ----------------------- find_exact_duplicates ---------------------------


class TestFindExactDuplicates:
    def test_empty(self):
        assert find_exact_duplicates({}) == []

    def test_no_duplicates(self):
        phashes = {"a": "0000", "b": "1111", "c": "2222"}
        assert find_exact_duplicates(phashes) == []

    def test_single_pair(self):
        phashes = {
            "bdd100k/train/a": "abcd1234",
            "bdd100k/train/b": "abcd1234",
            "bdd100k/train/c": "ffff0000",
        }
        groups = find_exact_duplicates(phashes)
        assert len(groups) == 1
        assert set(groups[0]) == {"bdd100k/train/a", "bdd100k/train/b"}

    def test_multi_item_group(self):
        phashes = {
            "a": "ff", "b": "ff", "c": "ff",  # 3-item group
            "d": "00",                          # singleton, excluded
        }
        groups = find_exact_duplicates(phashes)
        assert len(groups) == 1
        assert set(groups[0]) == {"a", "b", "c"}

    def test_multiple_groups(self):
        phashes = {
            "a1": "AA", "a2": "AA",
            "b1": "BB", "b2": "BB", "b3": "BB",
            "c":  "CC",  # singleton
        }
        groups = find_exact_duplicates(phashes)
        assert len(groups) == 2
        sizes = sorted(len(g) for g in groups)
        assert sizes == [2, 3]


# -------------------------- cross_set_leakage ----------------------------


class TestCrossSetLeakage:
    def test_empty_sets(self):
        assert cross_set_leakage({}, {}) == []
        assert cross_set_leakage({"a": "ff"}, {}) == []
        assert cross_set_leakage({}, {"a": "ff"}) == []

    def test_no_overlap(self):
        a = {"a1": "1111", "a2": "2222"}
        b = {"b1": "3333", "b2": "4444"}
        assert cross_set_leakage(a, b) == []

    def test_one_to_one_match(self):
        a = {"pool/x": "abcd"}
        b = {"eval/y": "abcd"}
        pairs = cross_set_leakage(a, b)
        assert pairs == [("pool/x", "eval/y", "abcd")]

    def test_cartesian_when_many_to_many(self):
        # Same pHash in 2 pool items and 2 eval items → 4 pairs enumerated
        a = {"pool/x1": "ff", "pool/x2": "ff"}
        b = {"eval/y1": "ff", "eval/y2": "ff"}
        pairs = cross_set_leakage(a, b)
        assert len(pairs) == 4

    def test_does_not_match_within_set(self):
        # Two items in set_a share a hash, but if set_b has no match they
        # should NOT show up as cross-set pairs.
        a = {"a1": "ff", "a2": "ff"}
        b = {"b1": "00"}
        assert cross_set_leakage(a, b) == []

    def test_returns_full_triple(self):
        a = {"pool/x": "abcd"}
        b = {"eval/y": "abcd"}
        ((a_id, b_id, shared),) = cross_set_leakage(a, b)
        assert a_id == "pool/x"
        assert b_id == "eval/y"
        assert shared == "abcd"


# ------------------------- summarize_duplicates --------------------------


class TestSummarizeDuplicates:
    def test_empty(self):
        assert summarize_duplicates([]) == {}

    def test_intra_dataset_group(self):
        groups = [["bdd100k/train/a", "bdd100k/train/b"]]
        assert summarize_duplicates(groups) == {"bdd100k": 1}

    def test_cross_dataset_group(self):
        groups = [["bdd100k/train/a", "cityscapes/train/b"]]
        # Sorted by dataset name → "bdd100k|cityscapes"
        assert summarize_duplicates(groups) == {"bdd100k|cityscapes": 1}

    def test_multiple_groups_mixed(self):
        groups = [
            ["bdd100k/train/a", "bdd100k/train/b"],            # intra
            ["bdd100k/train/c", "mapillary/training/d"],       # cross
            ["a2d2/x/e", "cityscapes/train/f", "bdd100k/g/h"], # 3-way cross
            ["mapillary/training/i", "mapillary/training/j"],  # intra mapillary
        ]
        result = summarize_duplicates(groups)
        assert result == {
            "bdd100k": 1,
            "bdd100k|mapillary": 1,
            "a2d2|bdd100k|cityscapes": 1,
            "mapillary": 1,
        }

    def test_keys_are_sorted(self):
        # Ensure key order is deterministic regardless of input order
        groups_1 = [["cityscapes/x/a", "bdd100k/y/b"]]
        groups_2 = [["bdd100k/y/b", "cityscapes/x/a"]]
        assert summarize_duplicates(groups_1) == summarize_duplicates(groups_2)
