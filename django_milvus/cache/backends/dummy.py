"""No-op cache backend."""

from .base import MISSING, BaseCacheBackend


class DummyBackend(BaseCacheBackend):
    """Accepts everything, stores nothing, always misses.

    Useful for switching caching off in one environment without deleting
    the configuration, and as the null object in tests that assert on
    "cache disabled" behaviour. Every call is valid and cheap; nothing is
    ever returned.
    """

    name = "dummy"
    shared = False

    def get(self, key, default=MISSING):
        return default

    def set(self, key, value, ttl=None, size=None):
        return False

    def delete(self, key):
        return False

    def clear(self):
        return 0

    def __len__(self):
        return 0
