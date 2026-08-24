"""
The cache orchestrator - where all the pieces meet the query path.

:class:`MilvusCache` owns one alias's worth of caching: its backend
stack, version registry, semantic index, vector cache, statistics and
stampede protection. :data:`caches` holds one instance per configured
alias, built lazily.

The two entry points mirror the two things a queryset can do::

    cache.query(...)     non-vector reads: .filter(), .count()
    cache.search(...)    vector reads: .search(), .hybrid_search()

Both take a ``producer`` callable that performs the real Milvus call, and
both guarantee that a cache failure can only cost performance: every
lookup and store is wrapped, and any exception falls through to the
producer.
"""

import logging
import threading
import time

from django.utils.module_loading import import_string

from ..exceptions import CacheConfigurationError
from .backends.base import MISSING
from .config import is_configured, load_config
from .keys import (
    OP_COUNT,
    OP_QUERY,
    OP_SEARCH,
    collection_prefix,
    query_key,
    search_bucket_id,
    search_key,
    vector_fingerprint,
)
from .semantic import SemanticCache
from .signals import _send, cache_hit, cache_miss, cache_set
from .stampede import DistributedFlight
from .stats import CacheStats, prometheus_metrics
from .vectors import VectorCacheRegistry
from .versions import VersionRegistry

logger = logging.getLogger("django_milvus.cache")


def build_backend(config, stats):
    """Construct the backend stack described by ``config``.

    Always at least an L1. When an L2 is configured the two are wrapped in
    a :class:`~.backends.tiered.TieredCache`; otherwise the L1 is used
    directly, so an L1-only deployment pays nothing for the tiering it is
    not using.
    """
    try:
        l1_class = import_string(config.l1.backend)
    except ImportError as exc:
        raise CacheConfigurationError(
            f"Could not import L1 BACKEND {config.l1.backend!r}: {exc}"
        ) from exc

    l1 = l1_class(config=config, stats=stats, l1_config=config.l1)

    if config.l2 is None:
        return l1

    try:
        l2_class = import_string(config.l2.backend)
    except ImportError as exc:
        raise CacheConfigurationError(
            f"Could not import L2 BACKEND {config.l2.backend!r}: {exc}"
        ) from exc

    try:
        l2 = l2_class(
            config=config,
            stats=None,          # the tier records its own hits
            location=config.l2.location,
            prefix=config.l2.prefix,
            socket_timeout=config.l2.socket_timeout,
            **config.l2.options,
        )
    except Exception:
        # A shared tier that will not start must not stop the application.
        # Log loudly and run L1-only; the cache still works, just without
        # cross-worker sharing.
        logger.warning(
            "django-milvus cache: could not initialise the shared tier %r; "
            "continuing with the local tier only",
            config.l2.backend, exc_info=True,
        )
        return l1

    from .backends.tiered import CircuitBreaker, TieredCache
    return TieredCache(
        l1=l1, l2=l2, config=config, stats=stats,
        breaker=CircuitBreaker(
            failures=config.l2.breaker_failures,
            reset_after=config.l2.breaker_reset_after,
        ),
    )


class CacheResult:
    """Outcome of a cached lookup, so callers can see *how* it was served."""

    __slots__ = ("value", "hit", "tier", "similarity", "key", "elapsed")

    def __init__(self, value, hit=False, tier=None, similarity=None,
                 key=None, elapsed=0.0):
        self.value = value
        self.hit = hit
        self.tier = tier
        self.similarity = similarity
        self.key = key
        self.elapsed = elapsed

    def __repr__(self):
        how = self.tier if self.hit else "miss"
        return f"<CacheResult {how} key={self.key}>"


class MilvusCache:
    """Everything caching for one alias."""

    def __init__(self, alias="default", config=None):
        self.alias = alias
        self.config = config if config is not None else load_config(alias)

        self.stats = CacheStats(
            window=self.config.stats_window, enabled=self.config.stats_enabled
        )
        self.backend = build_backend(self.config, self.stats)
        self.versions = VersionRegistry(
            alias=alias,
            backend=self.backend,
            shared=self.config.versioning_shared,
            refresh_interval=self.config.versioning_refresh,
            enabled=self.config.versioning_enabled,
        )
        self.vectors = VectorCacheRegistry(
            capacity=self.config.semantic.max_vectors
        )
        self.semantic = SemanticCache(self.config.semantic, self.vectors)
        self.flight = DistributedFlight(
            self.backend,
            timeout=self.config.stampede_timeout,
            stats=self.stats,
        )
        self._lock = threading.Lock()

    # ── enable / disable ─────────────────────────────────

    @property
    def enabled(self):
        return self.config.enabled

    def _ttl_for(self, value, options):
        """Pick a TTL, with jitter, for a payload about to be stored.

        Empty results get their own shorter ``NEGATIVE_TTL``: they are
        worth caching (a repeated query for nothing still costs a round
        trip) but they are the most likely thing an insert invalidates.

        Jitter spreads expiry. Without it, entries written together expire
        together, and a batch warm-up turns into a synchronised stampede
        one TTL later.
        """
        if options.get("ttl") is not None:
            ttl = options["ttl"]
        elif not value and self.config.negative_ttl is not None:
            ttl = self.config.negative_ttl
        else:
            ttl = self.config.ttl

        if not ttl:
            return None

        jitter = self.config.ttl_jitter
        if jitter:
            import random
            ttl = ttl * (1.0 + random.uniform(-jitter, jitter))
        return max(0.001, ttl)

    # ── versions ─────────────────────────────────────────

    def version(self, collection):
        return self.versions.get(collection)

    def invalidate(self, collection, reason="write", sender=None):
        """Invalidate every cached entry for ``collection``."""
        version = self.versions.bump(collection, reason=reason, sender=sender)
        # The semantic index maps query vectors to keys that just became
        # unreachable, so its contents are worthless too.
        self.semantic.drop_collection(collection)
        self.stats.record_invalidation()
        return version

    def clear(self, collection=None):
        """Drop cached entries: one collection's, or everything."""
        if collection is None:
            self.semantic.clear()
            self.vectors.clear()
            self.versions.reset()
            return self.backend.clear()

        self.semantic.drop_collection(collection)
        self.vectors.drop_collection(collection)
        removed = self.backend.delete_prefix(
            collection_prefix(self.alias, collection)
        )
        self.invalidate(collection, reason="clear")
        return removed

    # ── generic lookup ───────────────────────────────────

    def _lookup(self, key):
        """Read ``key``, honouring stale-while-revalidate.

        Returns ``(value, state)`` where state is ``"fresh"``, ``"stale"``
        or ``"miss"``.
        """
        try:
            value = self.backend.get(key, MISSING)
        except Exception:
            self.stats.record_error()
            logger.warning(
                "django-milvus cache: lookup failed for %s", key, exc_info=True
            )
            return MISSING, "miss"

        if value is not MISSING:
            return value, "fresh"

        grace = self.config.stale_while_revalidate
        if not grace:
            return MISSING, "miss"

        try:
            entry = self.backend.get_entry(key)
        except Exception:
            return MISSING, "miss"
        if entry is MISSING or entry is None:
            return MISSING, "miss"
        if entry.is_stale_usable(grace):
            return entry.value, "stale"
        return MISSING, "miss"

    def _store(self, key, value, options, sender=None, collection=None):
        """Write a payload, respecting the entry-size ceiling."""
        ttl = self._ttl_for(value, options)
        try:
            from .memory import estimate_size
            size = estimate_size(value)
            stored = self.backend.set(key, value, ttl=ttl, size=size)
        except Exception:
            self.stats.record_error()
            logger.warning(
                "django-milvus cache: store failed for %s", key, exc_info=True
            )
            return False
        if stored:
            _send(cache_set, sender=sender, key=key, collection=collection,
                  alias=self.alias, size=size, ttl=ttl, tier="l1")
        return stored

    # ── non-vector queries ───────────────────────────────

    def query(self, *, collection, producer, options=None, sender=None,
              filter_expr=None, output_fields=None, limit=None, offset=0,
              partitions=None, consistency=None, extra=None, op=OP_QUERY):
        """Serve a non-vector read from cache, or run and cache it."""
        options = options or {}
        started = time.time()
        version = self.version(collection)
        key = query_key(
            self.alias, collection, version,
            filter_expr=filter_expr, output_fields=output_fields,
            limit=limit, offset=offset, partitions=partitions,
            consistency=consistency, extra=extra, op=op,
        )

        if not options.get("refresh"):
            value, state = self._lookup(key)
            if value is not MISSING:
                elapsed = time.time() - started
                self.stats.record_hit(elapsed=elapsed)
                if state == "stale":
                    self.stats.record_stale_hit()
                if not value:
                    self.stats.record_negative_hit()
                _send(cache_hit, sender=sender, key=key, tier=state,
                      collection=collection, alias=self.alias, elapsed=elapsed)
                return CacheResult(value, hit=True, tier=state, key=key,
                                   elapsed=elapsed)

        def run():
            result = producer(limit)
            self._store(key, result, options, sender=sender,
                        collection=collection)
            return result

        value, _ = self._flight(key, run)
        elapsed = time.time() - started
        self.stats.record_miss(elapsed=elapsed)
        _send(cache_miss, sender=sender, key=key, collection=collection,
              alias=self.alias, reason="miss", elapsed=elapsed)
        return CacheResult(value, hit=False, key=key, elapsed=elapsed)

    def count(self, **kwargs):
        """Cache a ``count(*)`` result."""
        kwargs.setdefault("op", OP_COUNT)
        return self.query(**kwargs)

    # ── vector search ────────────────────────────────────

    def search(self, *, collection, producer, vectors, anns_field,
               limit, options=None, sender=None, search_params=None,
               filter_expr=None, output_fields=None, offset=0,
               partitions=None, consistency=None, extra=None,
               op=OP_SEARCH, metric=None):
        """Serve a vector search from cache, or run and cache it.

        Lookup order: exact key, then nearest cached query vector, then
        Milvus. See :mod:`.semantic` for what the second step trades.
        """
        options = options or {}
        started = time.time()
        version = self.version(collection)

        fingerprint = vector_fingerprint(vectors)
        key = search_key(
            self.alias, collection, version,
            vector_fp=fingerprint, anns_field=anns_field,
            search_params=search_params, filter_expr=filter_expr,
            output_fields=output_fields, limit=limit, offset=offset,
            partitions=partitions, consistency=consistency, extra=extra, op=op,
        )

        # Semantic matching compares one query vector against another, so
        # it only applies to single-vector searches. A batch search still
        # gets exact-key caching.
        single = self._single_vector(vectors)
        semantic = self._semantic_options(options)
        use_semantic = (
            semantic["enabled"] and single is not None and op == OP_SEARCH
        )

        if not options.get("refresh"):
            value, state = self._lookup(key)
            if value is not MISSING:
                elapsed = time.time() - started
                self.stats.record_hit(elapsed=elapsed)
                if state == "stale":
                    self.stats.record_stale_hit()
                _send(cache_hit, sender=sender, key=key, tier=state,
                      collection=collection, alias=self.alias, elapsed=elapsed)
                return CacheResult(
                    self._truncate(value, limit), hit=True, tier=state,
                    key=key, elapsed=elapsed,
                )

            if use_semantic:
                result = self._semantic_lookup(
                    collection=collection, version=version, single=single,
                    anns_field=anns_field, search_params=search_params,
                    filter_expr=filter_expr, output_fields=output_fields,
                    limit=limit, offset=offset, partitions=partitions,
                    consistency=consistency, extra=extra, sender=sender,
                    semantic=semantic, metric=metric, started=started,
                )
                if result is not None:
                    return result

        # Over-fetch on a miss so a later semantic hit has candidates to
        # rerank. Without spare candidates, reranking can only reorder the
        # exact rows the neighbour needed, which is rarely enough.
        fetch_limit = limit
        if use_semantic and semantic["rerank"] and limit:
            fetch_limit = max(limit, int(limit * semantic["overfetch"]))

        def run():
            raw = producer(fetch_limit)
            payload = self._extract_vectors(
                raw, collection, anns_field, options,
            )
            self._store(key, payload, options, sender=sender,
                        collection=collection)
            if use_semantic:
                bucket = search_bucket_id(
                    self.alias, collection, version, anns_field=anns_field,
                    search_params=search_params, filter_expr=filter_expr,
                    output_fields=output_fields, limit=limit, offset=offset,
                    partitions=partitions, consistency=consistency, extra=extra,
                )
                self.semantic.remember(
                    bucket, key, single, collection=collection,
                    metric=metric or semantic["metric"],
                )
            return payload

        payload, _ = self._flight(key, run)
        elapsed = time.time() - started
        self.stats.record_miss(elapsed=elapsed)
        _send(cache_miss, sender=sender, key=key, collection=collection,
              alias=self.alias, reason="miss", elapsed=elapsed)
        return CacheResult(
            self._truncate(payload, limit), hit=False, key=key, elapsed=elapsed
        )

    def _semantic_lookup(self, *, collection, version, single, anns_field,
                         search_params, filter_expr, output_fields, limit,
                         offset, partitions, consistency, extra, sender,
                         semantic, metric, started):
        """Try to answer from a neighbouring query's cached results."""
        bucket = search_bucket_id(
            self.alias, collection, version, anns_field=anns_field,
            search_params=search_params, filter_expr=filter_expr,
            output_fields=output_fields, limit=limit, offset=offset,
            partitions=partitions, consistency=consistency, extra=extra,
        )
        try:
            neighbour, similarity = self.semantic.lookup(
                bucket, single, threshold=semantic["threshold"],
                dim=len(single), collection=collection,
                metric=metric or semantic["metric"],
            )
        except Exception:
            self.stats.record_error()
            logger.warning(
                "django-milvus cache: semantic lookup failed", exc_info=True
            )
            return None

        if neighbour is None:
            return None

        value, state = self._lookup(neighbour)
        if value is MISSING:
            # The neighbour's payload is gone (evicted or expired) but its
            # vector lingered. Drop the vector so it stops advertising an
            # answer nobody can produce.
            self.semantic.forget(bucket, neighbour)
            return None

        payload = value
        if semantic["rerank"]:
            payload = self._rerank_payload(
                value, single, collection, anns_field, limit,
                metric or semantic["metric"],
            )

        elapsed = time.time() - started
        self.stats.record_hit(elapsed=elapsed)
        self.stats.record_semantic_hit()
        _send(cache_hit, sender=sender, key=neighbour, tier="semantic",
              collection=collection, alias=self.alias,
              similarity=similarity, elapsed=elapsed)
        return CacheResult(
            self._truncate(payload, limit), hit=True, tier="semantic",
            similarity=similarity, key=neighbour, elapsed=elapsed,
        )

    # ── payload shaping ──────────────────────────────────

    @staticmethod
    def _single_vector(vectors):
        """The one query vector, or None for a batch / sparse search."""
        if vectors is None:
            return None
        if isinstance(vectors, dict):
            return None
        try:
            if len(vectors) == 1 and isinstance(vectors[0], (list, tuple)):
                candidate = vectors[0]
            elif vectors and not isinstance(vectors[0], (list, tuple, dict)):
                candidate = vectors
            else:
                return None
        except TypeError:
            return None
        if isinstance(candidate, dict) or not candidate:
            return None
        return candidate

    def _semantic_options(self, options):
        """Merge per-query semantic overrides over the configured defaults."""
        base = self.config.semantic
        override = options.get("semantic")
        resolved = {
            "enabled": base.enabled,
            "threshold": base.threshold,
            "metric": base.metric,
            "overfetch": base.overfetch,
            "rerank": base.rerank,
        }
        if override is False:
            resolved["enabled"] = False
        elif isinstance(override, (int, float)) and not isinstance(override, bool):
            # .cache(semantic=0.99) is shorthand for a threshold.
            resolved["enabled"] = True
            resolved["threshold"] = float(override)
        elif override is True:
            resolved["enabled"] = True
        elif isinstance(override, dict):
            resolved["enabled"] = override.get("enabled", True)
            resolved.update(
                {k: v for k, v in override.items() if k in resolved}
            )
        return resolved

    def _extract_vectors(self, raw, collection, anns_field, options):
        """Move embeddings out of the payload and into the vector cache.

        Cached search results are stored per query, so an embedding
        present in ten cached result sets would otherwise be stored ten
        times. Hoisting it into the shared matrix keeps one copy, keeps
        the cached payloads small, and is what makes reranking possible
        even for queries that never asked for the vector field.
        """
        if not raw or not anns_field:
            return raw

        keep = bool(options.get("keep_vectors"))
        cache = None
        slim = []

        for group in raw:
            rows = []
            for hit in group:
                entity = hit.get("entity") if isinstance(hit, dict) else None
                if not isinstance(entity, dict) or anns_field not in entity:
                    rows.append(hit)
                    continue

                embedding = entity.get(anns_field)
                pk = hit.get("id", entity.get("id"))
                if embedding is not None and pk is not None:
                    if cache is None:
                        try:
                            cache = self.vectors.for_field(
                                collection, anns_field, dim=len(embedding)
                            )
                        except TypeError:
                            cache = None
                    if cache is not None:
                        cache.put(pk, embedding)

                if keep:
                    rows.append(hit)
                else:
                    trimmed = dict(hit)
                    trimmed["entity"] = {
                        k: v for k, v in entity.items() if k != anns_field
                    }
                    rows.append(trimmed)
            slim.append(rows)
        return slim

    def _rerank_payload(self, payload, query_vector, collection, anns_field,
                        limit, metric):
        """Re-score a neighbour's grouped hits against the real query."""
        if not payload:
            return payload
        try:
            return [
                self.semantic.rerank(
                    group, query_vector, collection, anns_field, limit,
                    metric=metric,
                )
                for group in payload
            ]
        except Exception:
            logger.warning(
                "django-milvus cache: rerank failed; returning the cached "
                "order unchanged", exc_info=True,
            )
            return payload

    @staticmethod
    def _truncate(payload, limit):
        """Trim over-fetched rows back to what the caller asked for."""
        if not limit or not payload:
            return payload
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            return [group[:limit] for group in payload]
        return payload

    # ── stampede ─────────────────────────────────────────

    def _flight(self, key, run):
        """Run ``run`` under single-flight, if stampede protection is on."""
        if not self.config.stampede_enabled:
            return run(), True

        def reader():
            value, _ = self._lookup(key)
            return None if value is MISSING else value

        try:
            return self.flight.run(key, run, reader=reader)
        except TypeError:
            # A SingleFlight without the reader argument.
            return self.flight.run(key, run)

    # ── introspection ────────────────────────────────────

    def stats_dict(self):
        data = self.stats.as_dict()
        data["alias"] = self.alias
        data["enabled"] = self.config.enabled
        try:
            data["backend"] = self.backend.stats_dict()
            for field in ("entries", "bytes", "utilization"):
                if field in data["backend"]:
                    data[field] = data["backend"][field]
        except Exception:
            data["backend"] = {"error": "unavailable"}
        data["semantic"] = self.semantic.stats_dict()
        data["vectors"] = self.vectors.stats_dict()
        data["versions"] = self.versions.stats_dict()
        data["stampede"] = self.flight.stats_dict()
        return data

    def close(self):
        try:
            self.backend.close()
        except Exception:  # pragma: no cover
            pass

    def __repr__(self):
        return f"<MilvusCache alias={self.alias!r} enabled={self.enabled}>"


class CacheRegistry:
    """Lazily builds and holds one :class:`MilvusCache` per alias."""

    def __init__(self):
        self._caches = {}
        self._lock = threading.Lock()

    def __getitem__(self, alias):
        cache = self._caches.get(alias)
        if cache is not None:
            return cache
        with self._lock:
            cache = self._caches.get(alias)
            if cache is None:
                cache = MilvusCache(alias)
                self._caches[alias] = cache
            return cache

    def get(self, alias="default", default=None):
        """Return the cache for ``alias``, or ``default`` if unavailable.

        Returns ``default`` rather than raising when caching is not
        configured, since every call site treats "no cache" as a normal
        state rather than an error.
        """
        try:
            return self[alias]
        except Exception:
            logger.warning(
                "django-milvus cache: could not build cache alias %r; "
                "queries will run uncached", alias, exc_info=True,
            )
            return default

    def __contains__(self, alias):
        return alias in self._caches

    def __iter__(self):
        return iter(dict(self._caches))

    def all(self):
        return dict(self._caches)

    def reset(self):
        """Discard every built cache (used by tests and settings reloads)."""
        with self._lock:
            for cache in self._caches.values():
                cache.close()
            self._caches.clear()

    def stats(self):
        return {alias: cache.stats_dict() for alias, cache in self._caches.items()}

    def prometheus(self):
        return prometheus_metrics(self.stats())


#: The process-wide registry of configured caches.
caches = CacheRegistry()


def get_cache(alias="default"):
    """Return the :class:`MilvusCache` for ``alias``, or None.

    None means "caching is not available here" - unconfigured, disabled,
    or failed to build. Callers treat all three the same way: query Milvus.
    """
    if not is_configured():
        return None
    cache = caches.get(alias)
    if cache is None or not cache.enabled:
        return None
    return cache


def invalidate(collection, alias="default", reason="manual", sender=None):
    """Invalidate every cached entry for ``collection``."""
    cache = get_cache(alias)
    if cache is None:
        return 0
    return cache.invalidate(collection, reason=reason, sender=sender)


def clear_all(alias=None):
    """Empty one alias's cache, or every configured alias."""
    if alias is not None:
        cache = get_cache(alias)
        return cache.clear() if cache else 0
    return sum(cache.clear() for cache in caches.all().values())


def cache_stats(alias=None):
    """Statistics for one alias, or all of them."""
    if alias is not None:
        cache = get_cache(alias)
        return cache.stats_dict() if cache else {}
    return caches.stats()
