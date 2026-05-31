"""Tests for the water-filling allocation and target resolution.

Pure logic, no filesystem. The headline cases are the user's worked example
(two sources at 450 + eight at 800 -> exactly 5000) and the cascade case (a
borderline source at 505 cannot reach the redistributed share).
"""

from __future__ import annotations

import pytest

from yolo_contrastive.data.downstream.allocate import resolve_target, water_fill_allocation


# --------------------------------------------------------------------------- resolve_target
def test_resolve_total_fixed():
    assert resolve_target(15, total=5000) == 5000


def test_resolve_per_dataset_scales_with_n():
    assert resolve_target(15, per_dataset=500) == 7500
    assert resolve_target(10, per_dataset=500) == 5000


def test_resolve_requires_exactly_one():
    with pytest.raises(ValueError):
        resolve_target(10)
    with pytest.raises(ValueError):
        resolve_target(10, total=5000, per_dataset=500)


# --------------------------------------------------------------------------- water-filling
def test_simple_example():
    # two small (450) + eight large (800) -> exactly 5000
    counts = {**{f"s{i}": 450 for i in range(2)}, **{f"L{i}": 800 for i in range(8)}}
    a = water_fill_allocation(counts, 5000)
    assert sum(a.values()) == 5000
    assert a["s0"] == 450 and a["s1"] == 450
    assert sorted(a[f"L{i}"] for i in range(8)) == [512, 512, 512, 512, 513, 513, 513, 513]


def test_cascade():
    # borderline source (505) cannot reach 512.5 -> taken whole, deficit re-spreads
    counts = {"s0": 450, "s1": 450, "c": 505, **{f"L{i}": 2000 for i in range(7)}}
    a = water_fill_allocation(counts, 5000)
    assert sum(a.values()) == 5000
    assert a["c"] == 505
    assert sorted(a[f"L{i}"] for i in range(7)) == [513, 513, 513, 514, 514, 514, 514]


def test_all_large_equal():
    a = water_fill_allocation({f"L{i}": 2000 for i in range(10)}, 5000)
    assert all(v == 500 for v in a.values()) and sum(a.values()) == 5000


def test_exactly_at_cap():
    a = water_fill_allocation({f"L{i}": 500 for i in range(10)}, 5000)
    assert all(v == 500 for v in a.values())


def test_undersupply_takes_all():
    counts = {f"s{i}": 400 for i in range(10)}  # 4000 < 5000
    a = water_fill_allocation(counts, 5000)
    assert a == counts and sum(a.values()) == 4000


def test_never_exceeds_capacity():
    counts = {"a": 10, "b": 50, "c": 1000, "d": 3, "e": 700}
    a = water_fill_allocation(counts, 900)
    assert sum(a.values()) == 900
    for name, v in a.items():
        assert 0 <= v <= counts[name]


def test_scales_to_fifteen_sources():
    # 15 sources, per_dataset=500 -> target 7500; mix of small and large
    counts = {**{f"s{i}": 300 for i in range(3)}, **{f"L{i}": 2000 for i in range(12)}}
    target = resolve_target(len(counts), per_dataset=500)
    a = water_fill_allocation(counts, target)
    assert target == 7500 and sum(a.values()) == 7500
    assert all(a[f"s{i}"] == 300 for i in range(3))
