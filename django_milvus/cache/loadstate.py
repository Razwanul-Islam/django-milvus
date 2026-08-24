"""
Collection load-state cache.

Milvus refuses reads against a collection that is not loaded into memory.
django-milvus handles that reactively: catch the error, create indexes,
load, retry. That works, but without any memory of what it has already
done it repeats the whole sequence - including rebuilding index
parameters - on every future miss.

This module remembers two things per ``(alias, collection)``:

* whether the collection has been loaded;
* whether this process has already created its indexes.

Both are cached with a TTL, because the answer can change underneath us -
another service can release a collection, and a restart of Milvus loses
load state entirely. The TTL bounds how long a wrong "yes" can persist,
and any :class:`MilvusException` clears the entry immediately, so the next
attempt does the full reactive dance again.

This is deliberately optimistic. Being wrong costs one failed query that
is retried; being right saves a round trip and an index rebuild.
"""

import threading
import time


class LoadStateCache:
    """Remembers which collections are loaded and indexed."""

    #: How long a "loaded" answer is trusted. Short enough that an
    #: externally released collection recovers quickly, long enough to
    #: cover a burst of queries.
    DEFAULT_TTL = 300.0

    def __init__(self, ttl=None):
        self.ttl = self.DEFAULT_TTL if ttl is None else ttl
        self._lock = threading.RLock()
        self._loaded = {}    # (alias, collection) -> timestamp
        self._indexed = {}   # (alias, collection) -> timestamp

        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    @staticmethod
    def _key(alias, collection):
        return (alias, collection)

    def _fresh(self, store, key, now):
        stamp = store.get(key)
        if stamp is None:
            return False
        if self.ttl and (now - stamp) > self.ttl:
            store.pop(key, None)
            return False
        return True

    # ── load state ───────────────────────────────────────

    def is_loaded(self, collection, alias="default"):
        with self._lock:
            fresh = self._fresh(
                self._loaded, self._key(alias, collection), time.time()
            )
            if fresh:
                self.hits += 1
            else:
                self.misses += 1
            return fresh

    def mark_loaded(self, collection, alias="default"):
        with self._lock:
            self._loaded[self._key(alias, collection)] = time.time()

    def mark_released(self, collection, alias="default"):
        with self._lock:
            self._loaded.pop(self._key(alias, collection), None)

    # ── index state ──────────────────────────────────────

    def has_indexes(self, collection, alias="default"):
        with self._lock:
            return self._fresh(
                self._indexed, self._key(alias, collection), time.time()
            )

    def mark_indexed(self, collection, alias="default"):
        with self._lock:
            self._indexed[self._key(alias, collection)] = time.time()

    # ── invalidation ─────────────────────────────────────

    def invalidate(self, collection=None, alias="default"):
        """Forget what we believed, after an error or an explicit release."""
        with self._lock:
            self.invalidations += 1
            if collection is None:
                self._loaded.clear()
                self._indexed.clear()
                return
            key = self._key(alias, collection)
            self._loaded.pop(key, None)
            self._indexed.pop(key, None)

    def clear(self):
        self.invalidate(None)

    def stats_dict(self):
        with self._lock:
            return {
                "ttl": self.ttl,
                "loaded": len(self._loaded),
                "indexed": len(self._indexed),
                "hits": self.hits,
                "misses": self.misses,
                "invalidations": self.invalidations,
            }


#: Process-wide load-state cache.
load_state = LoadStateCache()
