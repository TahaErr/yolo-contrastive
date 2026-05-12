"""Image-set deduplication and cross-set leakage detection via pHash.

Public API:

- :func:`compute_phash`, :func:`compute_pool_phashes`, :func:`load_phashes`,
  :func:`hamming_distance` — pHash computation and persistence.
- :func:`find_exact_duplicates`, :func:`cross_set_leakage`,
  :func:`summarize_duplicates` — duplicate / leakage discovery on top of
  pHash sidecars.
"""

from .leakage import (
    cross_set_leakage,
    find_exact_duplicates,
    summarize_duplicates,
)
from .phash import (
    DEFAULT_HASH_SIZE,
    compute_phash,
    compute_pool_phashes,
    hamming_distance,
    load_phashes,
)

__all__ = [
    "DEFAULT_HASH_SIZE",
    "compute_phash",
    "compute_pool_phashes",
    "cross_set_leakage",
    "find_exact_duplicates",
    "hamming_distance",
    "load_phashes",
    "summarize_duplicates",
]
