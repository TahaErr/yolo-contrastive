"""Duplicate detection and cross-set leakage check based on pHash.

Two distinct operations live here:

- :func:`find_exact_duplicates` — groups image_ids that share an identical
  64-bit pHash within a single set. After our resize+JPEG normalization,
  identical pHash is a strong signal of perceptual duplication.

- :func:`cross_set_leakage` — given two ``image_id -> phash`` maps (e.g.
  SSL pool and an evaluation set), returns pairs where their hashes
  collide. This is how we keep the SSL pretrain ↔ downstream-eval boundary
  clean.

Both are exact-match (Hamming = 0) in this version. A Hamming-thresholded
near-duplicate finder is a natural extension and the API doesn't preclude
it, but the v1 we ship here is what the plan asks for and what catches
the actual contamination patterns we care about (cross-dataset scene
re-uploads, accidental double-materialization).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


def find_exact_duplicates(phashes: Dict[str, str]) -> List[List[str]]:
    """Group ``image_id``s that share an identical pHash.

    Returns a list of groups, each a list of ≥ 2 image_ids that all hash to
    the same value. Singletons (image_ids whose hash appears once) are not
    returned — only actionable groups make it out.

    The order of groups and of image_ids within a group is insertion-order
    stable for determinism in tests and logs.
    """
    by_hash: Dict[str, List[str]] = defaultdict(list)
    for image_id, phash in phashes.items():
        by_hash[phash].append(image_id)
    return [group for group in by_hash.values() if len(group) >= 2]


def cross_set_leakage(
    set_a: Dict[str, str],
    set_b: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    """Find ``(id_a, id_b, shared_phash)`` triples where ``set_a`` ∩ ``set_b`` is non-empty by pHash.

    Used to detect contamination — e.g. an evaluation set has images that
    were also in the SSL pretrain pool. If many items in ``set_a`` share a
    pHash with many in ``set_b``, the cartesian product is enumerated
    (rare in practice; usually one-to-one or small clusters).

    Symmetry note: this returns A→B matches. Calling it with swapped args
    yields B→A matches with the same pairs but swapped positions.
    """
    b_by_hash: Dict[str, List[str]] = defaultdict(list)
    for image_id, phash in set_b.items():
        b_by_hash[phash].append(image_id)

    pairs: List[Tuple[str, str, str]] = []
    for a_id, a_phash in set_a.items():
        for b_id in b_by_hash.get(a_phash, []):
            pairs.append((a_id, b_id, a_phash))
    return pairs


def summarize_duplicates(groups: List[List[str]]) -> Dict[str, int]:
    """Count duplicate groups by which dataset(s) participate in them.

    Image_ids are assumed to start with ``<dataset>/...`` (our convention
    across all ssl_pool adapters). The first slash-segment is taken as the
    dataset name. A group spanning multiple datasets gets a key like
    ``"bdd100k|cityscapes"`` with components sorted; intra-dataset
    duplicates get a single-name key.

    The returned dict makes it easy to tell at a glance whether most
    duplicates are intra-dataset (lower-priority, typically near-identical
    consecutive frames) or cross-dataset (higher-priority, real overlap
    between sources).
    """
    counts: Dict[str, int] = defaultdict(int)
    for group in groups:
        datasets = sorted(set(image_id.split("/")[0] for image_id in group))
        counts["|".join(datasets)] += 1
    return dict(counts)
