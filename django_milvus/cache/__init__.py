"""
Caching layer for django-milvus.

Off by default. Configure ``MILVUS_CACHE`` in settings, then opt a model
in with ``MilvusMeta.cache`` or a single queryset with ``.cache()``::

    MILVUS_CACHE = {
        "default": {
            "TTL": 300,
            "L1": {"MAX_MEMORY": "256MB", "ALGORITHM": "w-tinylfu"},
        }
    }

    class Document(MilvusModel):
        class MilvusMeta:
            cache = {"ttl": 600, "semantic": {"threshold": 0.98}}

    Document.objects.search(vector, limit=5)            # cached
    Document.objects.search(vector, limit=5).no_cache()  # not

What it provides:

* a tiered store - an in-process RAM tier, optionally in front of Redis
  or ``django.core.cache``, with promotion between them;
* nine eviction algorithms, byte-accurate memory bounds and an adaptive
  admission window;
* **semantic caching**: a query whose embedding is close enough to one
  already cached is served from RAM, with results reranked against the
  caller's actual vector;
* version-stamped invalidation on every write, so a cached read never
  outlives the data behind it;
* stampede protection, negative caching, stale-while-revalidate and a
  circuit breaker on the shared tier.

Every operation fails open. A cache error costs a Milvus round trip, and
never a wrong answer or a failed request.

See the Caching section of the README for the full reference.
"""

from .backends.base import MISSING, BaseCacheBackend, CacheEntry
from .config import (
    DEFAULTS,
    CacheConfig,
    is_configured,
    load_config,
    parse_size,
    validate_all,
)
from .memory import estimate_size
from .policies import EvictionPolicy, available_policies, register
from .registry import (
    MilvusCache,
    cache_stats,
    caches,
    clear_all,
    get_cache,
    invalidate,
)
from .signals import (
    cache_evicted,
    cache_hit,
    cache_invalidated,
    cache_miss,
    cache_set,
)
from .stats import prometheus_metrics

__all__ = [
    # registry / public API
    "caches", "get_cache", "invalidate", "clear_all", "cache_stats",
    "MilvusCache",
    # configuration
    "CacheConfig", "DEFAULTS", "load_config", "validate_all",
    "is_configured", "parse_size",
    # extension points
    "BaseCacheBackend", "CacheEntry", "MISSING",
    "EvictionPolicy", "register", "available_policies",
    # observability
    "cache_hit", "cache_miss", "cache_set", "cache_evicted",
    "cache_invalidated", "prometheus_metrics",
    # utilities
    "estimate_size",
    # options helper
    "resolve_cache_options",
]


def resolve_cache_options(model, queryset_options=None):
    """Work out the effective cache settings for one query.

    Three levels, each overriding the last:

    1. ``MILVUS_CACHE`` - the alias defaults;
    2. ``MilvusMeta.cache`` - per model;
    3. ``.cache(...)`` / ``.no_cache()`` - per queryset.

    Returns ``None`` when this query must not be cached, and otherwise a
    dict of resolved options. ``None`` is the answer for every "cache is
    not available" case - unconfigured, disabled globally, not opted in,
    or explicitly turned off - because the caller does the same thing in
    all of them.
    """
    if queryset_options is not None and not queryset_options.get("enabled", True):
        return None

    model_option = getattr(getattr(model, "_options", None), "cache", None)

    resolved = {}
    if model_option is True:
        resolved = {}
    elif isinstance(model_option, dict):
        resolved = dict(model_option)
    elif model_option in (None, False):
        # Not opted in at the model level: only an explicit .cache() call
        # can turn caching on for this query.
        if queryset_options is None:
            return None
        resolved = {}

    if queryset_options:
        resolved.update(
            {k: v for k, v in queryset_options.items() if k != "enabled"}
        )

    resolved.setdefault("alias", "default")
    return resolved
