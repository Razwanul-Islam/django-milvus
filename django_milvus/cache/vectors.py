"""
Vector cache: primary key -> embedding.

Exists to serve reranking. When the semantic cache answers a query with a
*neighbouring* query's cached results, those results were ordered for the
neighbour, not for the caller. To reorder them correctly we need each
candidate's embedding - and fetching embeddings from Milvus would defeat
the purpose of not going to Milvus.

So embeddings are kept here, keyed by primary key, in one contiguous
``float32`` matrix per collection. Entries arrive either because a search
already returned the vector field, or because ``.cache(store_vectors=True)``
asked for it explicitly.

Cost is predictable and worth stating: a 768-dim ``float32`` embedding is
3,072 bytes, so 10,000 of them is about 30 MB. The matrix is preallocated
and rows are recycled on eviction, so there is no per-entry Python object
and no fragmentation.

Vectors are stored **L2-normalized**. Cosine similarity is then a plain
dot product, which turns reranking a candidate list into a single matrix
multiply.
"""

import logging
import threading

logger = logging.getLogger("django_milvus.cache")

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy ships with pymilvus
    np = None


def normalize(vector, out=None):
    """Return ``vector`` as a unit-length float32 array."""
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm > 0:
        array = array / norm
    if out is not None:
        out[:] = array
        return out
    return array


class VectorCache:
    """Bounded pk -> embedding store for one collection and field.

    Uses an LRU over matrix rows. Eviction frees a row for reuse rather
    than shrinking the matrix, so steady-state operation performs no
    allocation at all.
    """

    def __init__(self, dim, capacity=20_000, name=""):
        if np is None:  # pragma: no cover - numpy is a hard dependency
            raise RuntimeError(
                "The vector cache requires NumPy. Install it with: "
                "pip install numpy"
            )
        self.dim = dim
        self.capacity = max(1, capacity)
        self.name = name

        self._lock = threading.RLock()
        self._matrix = np.zeros((self.capacity, dim), dtype=np.float32)
        self._rows = {}        # pk -> row index
        self._row_owner = {}   # row index -> pk
        self._free_rows = list(range(self.capacity - 1, -1, -1))
        # Recency order over pks; the head is the eviction candidate.
        self._order = {}
        self._clock = 0

        self.stores = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ── writes ───────────────────────────────────────────

    def put(self, pk, vector):
        """Store one embedding, normalizing it on the way in."""
        with self._lock:
            row = self._rows.get(pk)
            if row is None:
                row = self._claim_row(pk)
                if row is None:
                    return False
            try:
                normalize(vector, out=self._matrix[row])
            except (ValueError, TypeError):
                # Wrong dimensionality: skip rather than corrupt the
                # matrix. Reranking degrades, the query still works.
                self._release_row(pk)
                return False
            self._clock += 1
            self._order[pk] = self._clock
            self.stores += 1
            return True

    def put_many(self, mapping):
        return sum(bool(self.put(pk, vec)) for pk, vec in mapping.items())

    def _claim_row(self, pk):
        """Caller holds the lock."""
        if not self._free_rows:
            self._evict_one()
        if not self._free_rows:
            return None
        row = self._free_rows.pop()
        self._rows[pk] = row
        self._row_owner[row] = pk
        return row

    def _evict_one(self):
        """Caller holds the lock. Drops the least-recently-used pk."""
        if not self._order:
            return
        victim = min(self._order, key=self._order.get)
        self._release_row(victim)
        self.evictions += 1

    def _release_row(self, pk):
        """Caller holds the lock."""
        row = self._rows.pop(pk, None)
        self._order.pop(pk, None)
        if row is not None:
            self._row_owner.pop(row, None)
            self._free_rows.append(row)

    # ── reads ────────────────────────────────────────────

    def get(self, pk):
        """Return the normalized embedding for ``pk``, or None."""
        with self._lock:
            row = self._rows.get(pk)
            if row is None:
                self.misses += 1
                return None
            self._clock += 1
            self._order[pk] = self._clock
            self.hits += 1
            return self._matrix[row]

    def get_matrix(self, pks):
        """Stack the embeddings for ``pks``.

        Returns ``(matrix, found_pks)`` covering only the pks actually
        held, so a caller can tell how much of its candidate list can be
        reranked.
        """
        with self._lock:
            rows = []
            found = []
            for pk in pks:
                row = self._rows.get(pk)
                if row is not None:
                    rows.append(row)
                    found.append(pk)
            if not rows:
                self.misses += len(pks)
                return None, []
            self.hits += len(rows)
            self.misses += len(pks) - len(rows)
            # Fancy indexing copies, which is what we want: the caller
            # must not hold a view into a matrix that eviction may reuse.
            return self._matrix[rows], found

    def has(self, pk):
        with self._lock:
            return pk in self._rows

    def delete(self, pk):
        with self._lock:
            if pk not in self._rows:
                return False
            self._release_row(pk)
            return True

    def clear(self):
        with self._lock:
            count = len(self._rows)
            self._rows.clear()
            self._row_owner.clear()
            self._order.clear()
            self._free_rows = list(range(self.capacity - 1, -1, -1))
            return count

    # ── introspection ────────────────────────────────────

    def __len__(self):
        with self._lock:
            return len(self._rows)

    @property
    def nbytes(self):
        return int(self._matrix.nbytes)

    def stats_dict(self):
        with self._lock:
            return {
                "name": self.name,
                "dim": self.dim,
                "entries": len(self._rows),
                "capacity": self.capacity,
                "bytes": self.nbytes,
                "bytes_per_vector": self.dim * 4,
                "stores": self.stores,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }

    def __repr__(self):
        return (
            f"<VectorCache {self.name!r} dim={self.dim} "
            f"entries={len(self)}/{self.capacity}>"
        )


class VectorCacheRegistry:
    """Lazily creates one :class:`VectorCache` per collection and field.

    Dimensionality is not known until the first vector arrives, and
    different vector fields on the same model may differ, so caches are
    built on demand rather than declared up front.
    """

    def __init__(self, capacity=20_000):
        self.capacity = capacity
        self._lock = threading.Lock()
        self._caches = {}

    def for_field(self, collection, field, dim=None):
        key = (collection, field)
        with self._lock:
            cache = self._caches.get(key)
            if cache is not None:
                return cache
            if dim is None:
                return None
            cache = VectorCache(
                dim=dim, capacity=self.capacity, name=f"{collection}.{field}"
            )
            self._caches[key] = cache
            return cache

    def drop_collection(self, collection):
        """Forget every vector cache belonging to ``collection``."""
        with self._lock:
            doomed = [k for k in self._caches if k[0] == collection]
            for key in doomed:
                self._caches.pop(key).clear()
            return len(doomed)

    def clear(self):
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            self._caches.clear()

    def stats_dict(self):
        with self._lock:
            return {
                "caches": len(self._caches),
                "entries": sum(len(c) for c in self._caches.values()),
                "bytes": sum(c.nbytes for c in self._caches.values()),
                "detail": {
                    f"{c}.{f}": cache.stats_dict()
                    for (c, f), cache in self._caches.items()
                },
            }
