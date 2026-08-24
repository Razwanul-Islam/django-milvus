"""
Semantic cache: serve a query from a *near* neighbour's cached results.

Exact-key caching only helps when the same embedding arrives twice. In
practice it rarely does - two users phrasing the same question produce
vectors that are 0.98 similar but never byte-identical, and an exact
cache misses every one of them.

The semantic cache closes that gap. Each cached query keeps its vector in
an in-memory index. A new query is compared against them, and if the
closest is within ``threshold`` the cached results answer it.

Lookup order in the query path::

    1. exact key      byte-identical vector          -> hit
    2. semantic       nearest cached vector >= t     -> hit (reranked)
    3. miss           ask Milvus, cache, index it

**Buckets.** Comparing vectors only makes sense between queries that are
otherwise identical, so each combination of collection, version, filter,
limit, output fields and search params owns its own index. Two searches
differing only in ``limit`` never share candidates - the shorter one
cannot answer the longer one.

**Reranking is what makes this safe.** Cached results were ordered for the
neighbour's vector, not the caller's. On a miss the cache deliberately
over-fetches (``limit * overfetch`` rows) and stores the wide list; on a
semantic hit the candidates are re-scored against the *caller's* actual
vector using embeddings from :mod:`.vectors`, re-sorted, and truncated.
The caller gets results ordered for its own query.

**What you trade.** Even reranked, the candidate *set* comes from the
neighbour's search. If a document is a genuine top-5 match for this query
but was not in the neighbour's top-15, no amount of reranking will find
it. That is the real cost, and it shrinks as ``threshold`` rises:

======  =========================================================
0.99+   Near-duplicates only. Essentially no recall loss.
0.97    A good default. Paraphrases hit; results stay accurate.
0.95    Aggressive. Higher hit rate, visible drift on the tail.
<0.95   Not recommended for user-facing search.
======  =========================================================

Set ``rerank: False`` only if you have no embeddings to rerank with and
accept the neighbour's ordering as-is.
"""

import logging
import threading

logger = logging.getLogger("django_milvus.cache")

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy ships with pymilvus
    np = None


def _require_numpy():
    if np is None:  # pragma: no cover
        raise RuntimeError(
            "Semantic caching requires NumPy. Install it with: pip install numpy"
        )


class SemanticIndex:
    """Nearest-neighbour index over the query vectors of one bucket.

    A preallocated ``(capacity, dim)`` float32 matrix with L2-normalized
    rows, so a probe is one matrix-vector product::

        sims = matrix[:n] @ query        # cosine, since rows are unit

    Brute force is the right choice at cache scale. A cache holds
    thousands of query vectors, not millions: a 10,000 x 768 matvec is
    single-digit milliseconds on one core, needs no index build, and -
    unlike HNSW - never misses a true neighbour. ``index: "hnswlib"``
    exists for caches large enough to want an approximate structure.
    """

    def __init__(self, dim, capacity=20_000, metric="COSINE", name=""):
        _require_numpy()
        self.dim = dim
        self.capacity = max(1, capacity)
        self.metric = metric.upper()
        self.name = name

        self._lock = threading.RLock()
        self._matrix = np.zeros((self.capacity, dim), dtype=np.float32)
        # Rows stay densely packed in [0, _count): removal swaps the last
        # row into the hole. That keeps `matrix[:count]` a contiguous view,
        # so a probe is one matvec with no gather and no Python-level scan
        # over unused capacity.
        self._keys = [None] * self.capacity     # row -> cache key
        self._rows = {}                         # cache key -> row
        self._order = {}                        # cache key -> recency clock
        self._clock = 0
        self._count = 0

        self.probes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ── writes ───────────────────────────────────────────

    def add(self, key, vector):
        """Index ``vector`` as answerable by cache entry ``key``."""
        with self._lock:
            array = self._prepare(vector)
            if array is None:
                return False

            row = self._rows.get(key)
            if row is None:
                row = self._claim_row(key)
                if row is None:
                    return False
            self._matrix[row] = array
            self._clock += 1
            self._order[key] = self._clock
            return True

    def _prepare(self, vector):
        """Normalize and dimension-check an incoming vector."""
        try:
            array = np.asarray(vector, dtype=np.float32).ravel()
        except (ValueError, TypeError):
            return None
        if array.shape[0] != self.dim:
            return None
        if self.metric in ("COSINE", "IP"):
            norm = float(np.linalg.norm(array))
            if norm == 0:
                return None
            array = array / norm
        return array

    def _claim_row(self, key):
        """Caller holds the lock."""
        if self._count >= self.capacity:
            self._evict_one()
        if self._count >= self.capacity:
            return None
        row = self._count
        self._rows[key] = row
        self._keys[row] = key
        self._count += 1
        return row

    def _evict_one(self):
        """Caller holds the lock. Drops the least recently used entry."""
        if not self._order:
            return
        victim = min(self._order, key=self._order.get)
        self.remove(victim, _locked=True)
        self.evictions += 1

    def remove(self, key, _locked=False):
        """Stop offering ``key`` as a semantic answer."""
        lock = self._lock if not _locked else _NullContext()
        with lock:
            row = self._rows.pop(key, None)
            self._order.pop(key, None)
            if row is None:
                return False

            # Swap the tail row into the hole so the block stays dense.
            last = self._count - 1
            if row != last:
                self._matrix[row] = self._matrix[last]
                moved = self._keys[last]
                self._keys[row] = moved
                if moved is not None:
                    self._rows[moved] = row
            self._keys[last] = None
            self._count -= 1
            return True

    # ── probing ──────────────────────────────────────────

    def probe(self, vector, threshold):
        """Find the nearest indexed vector at or above ``threshold``.

        Returns ``(key, similarity)`` or ``(None, best_similarity)`` so a
        caller can log how close a near-miss came.
        """
        with self._lock:
            self.probes += 1
            if self._count == 0:
                self.misses += 1
                return None, 0.0

            query = self._prepare(vector)
            if query is None:
                self.misses += 1
                return None, 0.0

            # Contiguous view over the live rows - no copy, no gather.
            block = self._matrix[:self._count]

            if self.metric == "L2":
                # Squared L2: lower is closer, so the threshold is an upper
                # bound on distance rather than a lower bound on similarity.
                diffs = block - query
                distances = np.einsum("ij,ij->i", diffs, diffs)
                best = int(np.argmin(distances))
                score = float(distances[best])
                matched = score <= threshold
            else:
                # COSINE / IP over unit rows: a single matvec.
                sims = block @ query
                best = int(np.argmax(sims))
                score = float(sims[best])
                matched = score >= threshold

            if matched:
                key = self._keys[best]
                if key is not None:
                    self._touch(key)
                    self.hits += 1
                    return key, score
            self.misses += 1
            return None, score

    def _touch(self, key):
        """Caller holds the lock."""
        self._clock += 1
        self._order[key] = self._clock

    # ── housekeeping ─────────────────────────────────────

    def clear(self):
        with self._lock:
            count = self._count
            self._rows.clear()
            self._order.clear()
            self._keys = [None] * self.capacity
            self._count = 0
            return count

    def __len__(self):
        with self._lock:
            return self._count

    @property
    def nbytes(self):
        return int(self._matrix.nbytes)

    def stats_dict(self):
        with self._lock:
            return {
                "name": self.name,
                "dim": self.dim,
                "metric": self.metric,
                "entries": self._count,
                "capacity": self.capacity,
                "bytes": self.nbytes,
                "probes": self.probes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }

    def __repr__(self):
        return (
            f"<SemanticIndex {self.name!r} dim={self.dim} "
            f"entries={len(self)}/{self.capacity} metric={self.metric}>"
        )


class _NullContext:
    """No-op context manager for reentrant internal calls."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class HNSWSemanticIndex(SemanticIndex):
    """Approximate index for caches large enough to need one.

    Only worth it past roughly 100k cached query vectors, where the brute
    force scan stops being free. It trades exactness for sublinear search:
    HNSW can miss a true nearest neighbour, which here costs a cache miss,
    not a wrong answer.

    Falls back to the brute-force parent whenever ``hnswlib`` is missing.
    """

    def __init__(self, dim, capacity=20_000, metric="COSINE", name="",
                 ef_construction=200, M=16, ef_search=50):
        super().__init__(dim, capacity=capacity, metric=metric, name=name)
        import hnswlib

        space = {"COSINE": "cosine", "IP": "ip", "L2": "l2"}[self.metric]
        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(
            max_elements=self.capacity, ef_construction=ef_construction, M=M
        )
        self._index.set_ef(ef_search)
        self._labels = {}
        self._next_label = 0

    def add(self, key, vector):
        with self._lock:
            array = self._prepare(vector)
            if array is None:
                return False
            label = self._labels.get(key)
            if label is None:
                if len(self._labels) >= self.capacity:
                    self._evict_one()
                label = self._next_label
                self._next_label += 1
                self._labels[key] = label
                self._keys_by_label = getattr(self, "_keys_by_label", {})
                self._keys_by_label[label] = key
                self._count += 1
            self._index.add_items(array.reshape(1, -1), [label])
            self._clock += 1
            self._order[key] = self._clock
            return True

    def probe(self, vector, threshold):
        with self._lock:
            self.probes += 1
            if self._count == 0:
                self.misses += 1
                return None, 0.0
            array = self._prepare(vector)
            if array is None:
                self.misses += 1
                return None, 0.0
            try:
                labels, distances = self._index.knn_query(
                    array.reshape(1, -1), k=1
                )
            except Exception:
                self.misses += 1
                return None, 0.0

            label = int(labels[0][0])
            distance = float(distances[0][0])
            key = getattr(self, "_keys_by_label", {}).get(label)
            if key is None or key not in self._order:
                self.misses += 1
                return None, 0.0

            if self.metric == "L2":
                score = distance
                matched = score <= threshold
            else:
                # hnswlib reports cosine *distance*; convert to similarity.
                score = 1.0 - distance
                matched = score >= threshold

            if matched:
                self._touch(key)
                self.hits += 1
                return key, score
            self.misses += 1
            return None, score

    def remove(self, key, _locked=False):
        lock = self._lock if not _locked else _NullContext()
        with lock:
            label = self._labels.pop(key, None)
            self._order.pop(key, None)
            if label is None:
                return False
            getattr(self, "_keys_by_label", {}).pop(label, None)
            try:
                self._index.mark_deleted(label)
            except Exception:  # pragma: no cover - already deleted
                pass
            self._count -= 1
            return True


def create_index(dim, capacity, metric="COSINE", name="", kind="auto"):
    """Build the best available semantic index.

    ``auto`` uses hnswlib only when it is installed *and* the cache is big
    enough to benefit; below that the brute-force scan is both faster and
    exact.
    """
    _require_numpy()

    if kind == "numpy":
        return SemanticIndex(dim, capacity, metric, name)

    if kind in ("auto", "hnswlib"):
        try:
            import hnswlib  # noqa: F401
        except ImportError:
            if kind == "hnswlib":
                logger.warning(
                    "django-milvus cache: SEMANTIC.index='hnswlib' requested "
                    "but hnswlib is not installed; using the exact NumPy "
                    "index instead. Install it with: "
                    "pip install django-milvus[fast]"
                )
            return SemanticIndex(dim, capacity, metric, name)

        if kind == "hnswlib" or capacity >= 100_000:
            try:
                return HNSWSemanticIndex(dim, capacity, metric, name)
            except Exception:
                logger.warning(
                    "django-milvus cache: could not build an hnswlib index; "
                    "falling back to the NumPy scan", exc_info=True,
                )
        return SemanticIndex(dim, capacity, metric, name)

    return SemanticIndex(dim, capacity, metric, name)


class SemanticCache:
    """Owns one index per bucket, and the rerank step.

    Buckets are created on demand and dropped wholesale when their
    collection is invalidated - a version bump means the cached results
    behind every vector in the bucket are stale, so the vectors are
    worthless too.
    """

    def __init__(self, config, vector_cache=None):
        self.config = config
        self.vector_cache = vector_cache
        self._lock = threading.Lock()
        self._buckets = {}
        self._bucket_collections = {}

        self.semantic_hits = 0
        self.semantic_misses = 0
        self.reranks = 0

    # ── bucket management ────────────────────────────────

    def get_bucket(self, bucket, dim=None, collection=None, metric=None):
        with self._lock:
            index = self._buckets.get(bucket)
            if index is not None:
                return index
            if dim is None:
                return None
            index = create_index(
                dim=dim,
                capacity=self.config.max_vectors,
                metric=(metric or self.config.metric),
                name=bucket[:16],
                kind=self.config.index,
            )
            self._buckets[bucket] = index
            if collection is not None:
                self._bucket_collections.setdefault(collection, set()).add(bucket)
            return index

    def drop_collection(self, collection):
        """Discard every bucket belonging to ``collection``."""
        with self._lock:
            buckets = self._bucket_collections.pop(collection, set())
            for bucket in buckets:
                index = self._buckets.pop(bucket, None)
                if index is not None:
                    index.clear()
            return len(buckets)

    def clear(self):
        with self._lock:
            count = len(self._buckets)
            for index in self._buckets.values():
                index.clear()
            self._buckets.clear()
            self._bucket_collections.clear()
            return count

    # ── the query path ───────────────────────────────────

    def lookup(self, bucket, vector, threshold=None, dim=None,
               collection=None, metric=None):
        """Find a cache key whose query vector is close enough.

        Returns ``(key, similarity)``; ``key`` is None on a miss.
        """
        index = self.get_bucket(bucket, dim=dim, collection=collection,
                                metric=metric)
        if index is None:
            return None, 0.0
        threshold = (
            self.config.threshold if threshold is None else threshold
        )
        key, score = index.probe(vector, threshold)
        if key is not None:
            self.semantic_hits += 1
        else:
            self.semantic_misses += 1
        return key, score

    def remember(self, bucket, key, vector, dim=None, collection=None,
                 metric=None):
        """Record that ``key`` answers queries near ``vector``."""
        if dim is None:
            try:
                dim = len(vector)
            except TypeError:
                return False
        index = self.get_bucket(bucket, dim=dim, collection=collection,
                                metric=metric)
        if index is None:
            return False
        return index.add(key, vector)

    def forget(self, bucket, key):
        with self._lock:
            index = self._buckets.get(bucket)
        return index.remove(key) if index is not None else False

    # ── reranking ────────────────────────────────────────

    def rerank(self, hits, query_vector, collection, vector_field, limit,
               metric=None):
        """Re-score a neighbour's cached hits against the real query.

        ``hits`` is the raw Milvus payload: a list of dicts with ``id``,
        ``distance`` and ``entity``. Returns a newly ordered, truncated
        list.

        Embeddings come from the entity payload when the vector field was
        requested, and otherwise from :class:`~.vectors.VectorCache`. A
        candidate whose embedding cannot be found keeps its original
        score, so a partially-populated vector cache degrades the ordering
        rather than dropping results.
        """
        if not hits or query_vector is None or np is None:
            return hits[:limit]

        metric = (metric or self.config.metric).upper()
        try:
            query = np.asarray(query_vector, dtype=np.float32).ravel()
        except (ValueError, TypeError):
            return hits[:limit]
        if metric in ("COSINE", "IP"):
            norm = float(np.linalg.norm(query))
            if norm == 0:
                return hits[:limit]
            query = query / norm

        cache = None
        if self.vector_cache is not None:
            cache = self.vector_cache.for_field(collection, vector_field)

        rescored = []
        rescored_count = 0
        for hit in hits:
            embedding = self._embedding_for(hit, vector_field, cache)
            if embedding is None:
                # Unknown embedding: keep the neighbour's score so the hit
                # is not silently dropped.
                rescored.append((self._original_score(hit, metric), hit))
                continue
            rescored_count += 1
            vector = np.asarray(embedding, dtype=np.float32).ravel()
            if metric == "L2":
                diff = vector - query
                score = -float(diff @ diff)       # negate: higher is better
            else:
                norm = float(np.linalg.norm(vector))
                if norm == 0:
                    score = self._original_score(hit, metric)
                else:
                    score = float((vector / norm) @ query)
            rescored.append((score, hit))

        if not rescored_count:
            # Nothing could be rescored; the neighbour's order is all we
            # have, so return it untouched rather than pretend otherwise.
            return hits[:limit]

        self.reranks += 1
        rescored.sort(key=lambda pair: pair[0], reverse=True)

        ordered = []
        for score, hit in rescored[:limit]:
            updated = dict(hit)
            updated["distance"] = (
                -score if metric == "L2" else score
            )
            ordered.append(updated)
        return ordered

    def _embedding_for(self, hit, vector_field, cache):
        entity = hit.get("entity") if isinstance(hit, dict) else None
        if isinstance(entity, dict) and vector_field in entity:
            candidate = entity.get(vector_field)
            if candidate is not None:
                return candidate
        if cache is None:
            return None
        pk = hit.get("id") if isinstance(hit, dict) else None
        if pk is None and isinstance(entity, dict):
            pk = entity.get("id")
        if pk is None:
            return None
        return cache.get(pk)

    @staticmethod
    def _original_score(hit, metric):
        distance = hit.get("distance") if isinstance(hit, dict) else None
        if distance is None:
            return float("-inf")
        return -float(distance) if metric == "L2" else float(distance)

    # ── introspection ────────────────────────────────────

    def stats_dict(self):
        with self._lock:
            buckets = dict(self._buckets)
        return {
            "enabled": self.config.enabled,
            "threshold": self.config.threshold,
            "metric": self.config.metric,
            "rerank": self.config.rerank,
            "overfetch": self.config.overfetch,
            "buckets": len(buckets),
            "vectors": sum(len(i) for i in buckets.values()),
            "bytes": sum(i.nbytes for i in buckets.values()),
            "hits": self.semantic_hits,
            "misses": self.semantic_misses,
            "reranks": self.reranks,
            "index_type": (
                type(next(iter(buckets.values()))).__name__
                if buckets else None
            ),
        }
