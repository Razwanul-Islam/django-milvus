"""
Cache backends for django-milvus.

===================== =================================================
``LocalRAMBackend``   In-process RAM. Microsecond hits, per-process.
``RedisBackend``      Shared across workers and hosts, needs ``redis``.
``DjangoCacheBackend`` Wraps whatever ``django.core.cache`` you run.
``TieredCache``       Composes an L1 and an L2, with promotion.
``DummyBackend``      Stores nothing; disables caching structurally.
===================== =================================================

See :mod:`.base` for the contract a custom backend must implement.
"""

from .base import MISSING, BaseCacheBackend, CacheEntry  # noqa: F401
from .dummy import DummyBackend  # noqa: F401
from .local import LocalRAMBackend  # noqa: F401

__all__ = [
    "BaseCacheBackend", "CacheEntry", "MISSING",
    "LocalRAMBackend", "DummyBackend",
    "DjangoCacheBackend", "RedisBackend", "TieredCache",
]


def __getattr__(name):
    """Import the optional backends lazily.

    ``RedisBackend`` pulls in ``redis`` and ``DjangoCacheBackend`` touches
    Django's cache framework; neither should be imported by projects that
    do not use them.
    """
    if name == "RedisBackend":
        from .redis import RedisBackend
        return RedisBackend
    if name == "DjangoCacheBackend":
        from .djangocache import DjangoCacheBackend
        return DjangoCacheBackend
    if name == "TieredCache":
        from .tiered import TieredCache
        return TieredCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
