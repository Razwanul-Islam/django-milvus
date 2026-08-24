"""
Django signals emitted by the caching layer.

Connect these to feed your own metrics pipeline, log slow misses, or
audit invalidations::

    from django.dispatch import receiver
    from django_milvus.cache.signals import cache_hit

    @receiver(cache_hit)
    def record(sender, key, tier, **kwargs):
        statsd.increment(f"milvus.cache.hit.{tier}")

``sender`` is the model class the query targeted, or ``None`` for
operations that are not tied to a model (a manual ``invalidate()``).
Receivers run inline on the query path, so keep them cheap; anything slow
belongs on a queue.
"""

from django.dispatch import Signal


#: Fired when a lookup is served from cache.
#: kwargs: key, tier ("l1" | "l2" | "semantic"), collection, alias,
#:         similarity (semantic hits only), elapsed
cache_hit = Signal()

#: Fired when a lookup is not served from cache and Milvus was queried.
#: kwargs: key, collection, alias, reason, elapsed
cache_miss = Signal()

#: Fired when a payload is written into the cache.
#: kwargs: key, collection, alias, size, ttl, tier
cache_set = Signal()

#: Fired when entries are evicted to reclaim capacity.
#: kwargs: keys, collection, alias, reason ("capacity" | "entries" |
#:         "expired" | "pressure"), freed
cache_evicted = Signal()

#: Fired when a collection's cached entries are invalidated.
#: kwargs: collection, alias, version, reason ("write" | "manual" | "clear")
cache_invalidated = Signal()


def _send(signal, sender=None, **kwargs):
    """Dispatch a signal without letting a receiver break the query path."""
    try:
        signal.send(sender=sender, **kwargs)
    except Exception:  # pragma: no cover - receiver bugs are not ours
        import logging
        logging.getLogger("django_milvus.cache").warning(
            "django-milvus cache signal receiver raised", exc_info=True
        )
