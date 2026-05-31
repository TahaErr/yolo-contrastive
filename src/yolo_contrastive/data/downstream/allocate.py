"""Capacity-constrained equal allocation (water-filling) for source balancing.

WORK_PLAN_v9 §13.4 — downstream "Pothole-5K" assembly.

Why this exists:
    The downstream pool is assembled from N independently-collected Roboflow
    sources of uneven size (some <500 images, some >2000). We want each source
    to contribute as EQUALLY as possible toward a target total, but no source
    can contribute more images than it actually has. Sources below the fair
    share are taken whole; their shortfall is redistributed equally over the
    sources that still have spare capacity. This is the classic water-filling
    allocation.

Design choices:

1. Target is explicit, never implicit.
   The caller passes EITHER a fixed ``total`` OR a ``per_dataset`` level (from
   which ``total = per_dataset * N``). Exactly one — there is no hidden default,
   because for N != 10 a stale default would silently change every count.
   ``resolve_target`` enforces this.

2. Integer, exact.
   The real-valued fair share (e.g. 512.5) is integerized with the
   largest-remainder method, so the allocation sums to EXACTLY ``target``
   whenever supply allows. Remainders go to the highest-capacity free sources
   first (always safe — a free source has capacity >= ceil(share)).

3. Cascade.
   A source that clears the initial fair share but cannot reach the
   redistributed share (e.g. 505 vs 512.5) is itself saturated and taken whole;
   its shortfall re-redistributes. Iterated until the free set is stable.

4. Undersupply is not an error.
   If total available < target, every source is taken whole; the caller warns.
   Allocation never invents images.

N-agnostic: nothing here assumes a specific source count.
"""

from __future__ import annotations


def resolve_target(n_sources: int, total: int | None = None,
                   per_dataset: int | None = None) -> int:
    """Resolve the allocation target. Pass EXACTLY one of ``total`` / ``per_dataset``.

    ``total=5000``            -> 5000 regardless of N (fixed budget).
    ``per_dataset=500, N=15`` -> 7500 (per-source level fixed, total scales with N).
    """
    if (total is None) == (per_dataset is None):
        raise ValueError("Pass exactly one of `total` or `per_dataset`.")
    if total is not None:
        if total < 0:
            raise ValueError("total must be >= 0")
        return total
    if per_dataset < 0:
        raise ValueError("per_dataset must be >= 0")
    return per_dataset * n_sources


def water_fill_allocation(counts: dict[str, int], target: int) -> dict[str, int]:
    """Allocate ``target`` images across sources, as equally as capacity allows.

    Args:
        counts: source name -> available image count (>= 0).
        target: total images to keep across all sources.

    Returns:
        source name -> kept count, with ``0 <= kept <= available`` and summing
        to ``target`` when total supply >= target (otherwise to total supply).
    """
    if any(c < 0 for c in counts.values()):
        raise ValueError("counts must be non-negative")

    total_supply = sum(counts.values())
    if total_supply <= target:
        return dict(counts)  # cannot reach target (or exactly meets it): keep all

    fixed: dict[str, int] = {}
    free = set(counts)
    remaining = target
    while free:
        share = remaining / len(free)
        saturated = [n for n in free if counts[n] <= share]
        if not saturated:
            break
        for n in saturated:           # source cannot fill its share -> give all it has
            fixed[n] = counts[n]
            remaining -= counts[n]
            free.discard(n)

    alloc = dict(fixed)
    if free:                          # integerize the remainder over free sources
        free_sorted = sorted(free, key=lambda n: (-counts[n], n))  # extras to most capacity
        base, leftover = divmod(remaining, len(free))
        for i, n in enumerate(free_sorted):
            alloc[n] = base + (1 if i < leftover else 0)
    return alloc
