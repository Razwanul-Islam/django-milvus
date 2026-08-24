"""
Eviction policies for the django-milvus cache.

Every policy is a pure data structure over keys - no payloads, no I/O -
selected by the ``ALGORITHM`` setting in ``MILVUS_CACHE``:

===========  ==========================================================
``lru``      Least Recently Used. Simple and fast; scans flush it.
``lfu``      Least Frequently Used, O(1) via frequency buckets.
``fifo``     Insertion order. Cheapest possible; a useful baseline.
``random``   Arbitrary victim. Immune to scan pollution by construction.
``ttl``      Nearest deadline first. For freshness-dominated data.
``slru``     Segmented LRU: a key must be hit twice to be protected.
``2q``       Admission queue plus ghost list; strongly scan-resistant.
``arc``      Self-balances between recency and frequency, no tuning.
``w-tinylfu`` Frequency-gated admission over SLRU. **The default.**
===========  ==========================================================

See :mod:`.base` for the contract and for registering your own.
"""

from .base import (  # noqa: F401
    EvictionPolicy,
    available_policies,
    create_policy,
    get_policy_class,
    register,
)
from .arc import ARCPolicy  # noqa: F401
from .fifo import FIFOPolicy  # noqa: F401
from .lfu import LFUPolicy  # noqa: F401
from .lru import LRUPolicy  # noqa: F401
from .random_policy import RandomPolicy  # noqa: F401
from .sketch import CountMinSketch  # noqa: F401
from .slru import SLRUPolicy  # noqa: F401
from .tinylfu import WTinyLFUPolicy  # noqa: F401
from .ttl import TTLPolicy  # noqa: F401
from .twoq import TwoQPolicy  # noqa: F401

__all__ = [
    "EvictionPolicy", "register", "create_policy", "get_policy_class",
    "available_policies", "CountMinSketch",
    "LRUPolicy", "LFUPolicy", "FIFOPolicy", "RandomPolicy", "TTLPolicy",
    "SLRUPolicy", "TwoQPolicy", "ARCPolicy", "WTinyLFUPolicy",
]
