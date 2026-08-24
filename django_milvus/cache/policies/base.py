"""
Eviction policy contract and registry.

A policy decides *ordering* only: which key should leave next, and
whether an incoming key deserves a place at all. It never stores payloads
and never touches bytes - the backend owns storage and byte accounting.
That split keeps every policy a pure, deterministic data structure that
can be unit-tested against an access trace with no I/O.

Capacity is expressed in *entries*, not bytes, because the segmented
policies (SLRU, 2Q, ARC, W-TinyLFU) size their internal regions as a
fraction of it. The backend recomputes an entry capacity from its byte
budget and pushes it down via :meth:`EvictionPolicy.set_capacity`.

To write your own::

    from django_milvus.cache.policies.base import EvictionPolicy, register

    class MyPolicy(EvictionPolicy):
        name = "mine"
        def on_admit(self, key, size=0, expires_at=None): ...
        def on_hit(self, key): ...
        def on_remove(self, key): ...
        def select_victim(self): ...

    register(MyPolicy)

Then set ``ALGORITHM: "mine"`` in ``MILVUS_CACHE``.
"""

from ...exceptions import CacheConfigurationError


class EvictionPolicy:
    """Base class for eviction policies.

    Implementations must keep their bookkeeping consistent with the
    backend: every key handed to :meth:`on_admit` is eventually passed to
    :meth:`on_remove`, either because it was evicted, expired, or deleted.
    """

    #: Registry name used by the ``ALGORITHM`` setting.
    name = "base"

    def __init__(self, capacity=1024, **options):
        self.capacity = max(1, int(capacity))
        self.options = options

    # ── required hooks ───────────────────────────────────

    def on_admit(self, key, size=0, expires_at=None):
        """Record that ``key`` has just been stored."""
        raise NotImplementedError

    def on_hit(self, key):
        """Record a successful lookup of ``key``."""
        raise NotImplementedError

    def on_remove(self, key):
        """Forget ``key``; it is no longer stored."""
        raise NotImplementedError

    def select_victim(self):
        """Return the key that should be evicted next, or None if empty."""
        raise NotImplementedError

    # ── optional hooks ───────────────────────────────────

    def should_admit(self, key):
        """Whether an incoming key is worth the eviction it would cause.

        Only W-TinyLFU overrides this. Everything else admits
        unconditionally, which is the classic behaviour.
        """
        return True

    def on_reject(self, key):
        """Record that ``key`` was refused admission."""

    def age(self):
        """Periodic maintenance, called by the janitor thread."""

    def set_capacity(self, capacity):
        """Resize the policy, re-proportioning any internal segments."""
        self.capacity = max(1, int(capacity))

    def clear(self):
        """Drop all bookkeeping."""
        raise NotImplementedError

    def keys(self):
        """Iterate tracked keys, in no guaranteed order."""
        raise NotImplementedError

    def stats(self):
        """Policy-specific numbers for ``cache_stats()``."""
        return {"algorithm": self.name, "tracked": len(self)}

    def __len__(self):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__} entries={len(self)} cap={self.capacity}>"


# ─────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────

_REGISTRY = {}


def register(policy_class, name=None):
    """Register a policy class under its ``name``."""
    key = name or policy_class.name
    if not key or key == "base":
        raise CacheConfigurationError(
            f"{policy_class.__name__} must define a unique 'name' attribute"
        )
    _REGISTRY[key] = policy_class
    return policy_class


def get_policy_class(name):
    """Look up a registered policy class by name."""
    _autoload()
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise CacheConfigurationError(
            f"Unknown cache ALGORITHM {name!r}. "
            f"Available: {sorted(_REGISTRY)}."
        ) from exc


def create_policy(name, capacity, **options):
    """Instantiate a registered policy."""
    return get_policy_class(name)(capacity=capacity, **options)


def available_policies():
    """Names of every registered policy."""
    _autoload()
    return sorted(_REGISTRY)


_loaded = False


def _autoload():
    """Import the bundled policies once, populating the registry."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import (  # noqa: F401
        arc, fifo, lfu, lru, random_policy, slru, tinylfu, ttl, twoq,
    )
