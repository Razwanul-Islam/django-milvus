"""
Cache key construction for django-milvus.

Two kinds of identity matter here:

``bucket_id``
    Everything about a query *except* the query vector. Two searches with
    the same filter, limit, output fields and partitions share a bucket,
    which is exactly the set within which comparing query vectors by
    similarity is meaningful. Each bucket owns one semantic index.

``entry key``
    The bucket plus the vector fingerprint (searches) or nothing more
    (plain queries). This is what the backends store under.

Keys embed the collection's version stamp, so bumping a version makes
every key for that collection unreachable at O(1) - no key scanning and
no reverse index. Orphaned entries are reclaimed by TTL or eviction.

Key layout, chosen so ``SCAN MATCH dmv:default:documents:*`` works::

    dmv:{alias}:{collection}:v{version}:{op}:{digest}
"""

import hashlib
import struct

KEY_PREFIX = "dmv"

# Query operation tags, kept short since they land in every key.
OP_QUERY = "q"
OP_SEARCH = "s"
OP_HYBRID = "h"
OP_COUNT = "c"


def _canonical(value):
    """Render ``value`` as a stable, order-independent string.

    Dicts are emitted with sorted keys and sets as sorted lists, so two
    logically identical queries always produce the same digest regardless
    of how the caller happened to build them.
    """
    if value is None:
        return "~"
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, (int, float, str)):
        return repr(value)
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(_canonical(v) for v in value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical(v) for v in value) + "]"
    if isinstance(value, dict):
        items = sorted((str(k), _canonical(v)) for k, v in value.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    return repr(value)


def _digest(*parts):
    payload = "|".join(_canonical(p) for p in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:24]


def _is_sequence_of_vectors(vector):
    """Whether ``vector`` is a batch rather than a single embedding.

    Written without truthiness tests and without indexing more than one
    element: a NumPy array raises ``ValueError`` on ``bool(array)``, and
    a generator cannot be indexed at all.
    """
    if isinstance(vector, dict):
        return False
    try:
        first = vector[0]
    except (TypeError, IndexError, KeyError):
        return False
    return isinstance(first, (list, tuple, dict)) or hasattr(first, "__len__")


def vector_fingerprint(vector):
    """Fingerprint a query vector (or batch of them) for exact matching.

    Values are cast to ``float32`` before hashing, so an embedding that
    Milvus would treat as identical fingerprints identically whether it
    arrived as a Python list, a tuple or a NumPy array.

    Sparse vectors arrive as ``{index: weight}`` dicts and are hashed
    through the canonical form, which sorts keys - two dicts built in a
    different order describe the same vector and must agree.
    """
    hasher = hashlib.blake2b(digest_size=16)

    def feed(vec):
        if isinstance(vec, dict):
            hasher.update(_canonical(vec).encode("utf-8"))
            return
        # NumPy arrays expose tolist(); everything else is already
        # iterable. Going through a list keeps struct.pack happy with
        # numpy scalars.
        values = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        try:
            hasher.update(struct.pack(f"<{len(values)}f", *values))
        except (struct.error, TypeError, ValueError):
            hasher.update(_canonical(values).encode("utf-8"))

    if _is_sequence_of_vectors(vector):
        for sub in vector:
            feed(sub)
            hasher.update(b"\x00")
    else:
        feed(vector)

    return hasher.hexdigest()


def _context(filter_expr, output_fields, limit, offset, partitions,
             consistency, extra):
    """Normalise the non-vector parts of a query into a hashable tuple."""
    return (
        filter_expr or "",
        sorted(output_fields) if output_fields else None,
        limit,
        offset or 0,
        sorted(partitions) if partitions else None,
        consistency or "",
        extra or None,
    )


def bucket_id(alias, collection, version, op, *, filter_expr=None,
              output_fields=None, limit=None, offset=0, partitions=None,
              consistency=None, extra=None):
    """Identify the set of queries that differ only by their vector."""
    return _digest(
        alias, collection, version, op,
        _context(filter_expr, output_fields, limit, offset, partitions,
                 consistency, extra),
    )


def make_key(alias, collection, version, op, digest):
    """Assemble the final backend key from its parts."""
    return f"{KEY_PREFIX}:{alias}:{collection}:v{version}:{op}:{digest}"


def query_key(alias, collection, version, *, filter_expr=None,
              output_fields=None, limit=None, offset=0, partitions=None,
              consistency=None, extra=None, op=OP_QUERY):
    """Key for a non-vector query (``.filter()``, ``.count()``)."""
    digest = _digest(
        _context(filter_expr, output_fields, limit, offset, partitions,
                 consistency, extra),
    )
    return make_key(alias, collection, version, op, digest)


def search_key(alias, collection, version, *, vector_fp, anns_field=None,
               search_params=None, filter_expr=None, output_fields=None,
               limit=None, offset=0, partitions=None, consistency=None,
               extra=None, op=OP_SEARCH):
    """Key for a vector search, identified exactly by its query vector."""
    digest = _digest(
        vector_fp, anns_field, search_params,
        _context(filter_expr, output_fields, limit, offset, partitions,
                 consistency, extra),
    )
    return make_key(alias, collection, version, op, digest)


def search_bucket_id(alias, collection, version, *, anns_field=None,
                     search_params=None, filter_expr=None, output_fields=None,
                     limit=None, offset=0, partitions=None, consistency=None,
                     extra=None):
    """Semantic namespace for a search: everything but the vector.

    Note that ``limit`` is part of the bucket. Two searches asking for
    different result counts never share cached candidates, because the
    shorter one could not answer the longer one.
    """
    return bucket_id(
        alias, collection, version, OP_SEARCH,
        filter_expr=filter_expr, output_fields=output_fields, limit=limit,
        offset=offset, partitions=partitions, consistency=consistency,
        extra=(anns_field, _canonical(search_params), extra),
    )


def collection_prefix(alias, collection, version=None):
    """Key prefix for one collection, for ``SCAN``-style bulk operations."""
    if version is None:
        return f"{KEY_PREFIX}:{alias}:{collection}:"
    return f"{KEY_PREFIX}:{alias}:{collection}:v{version}:"


def version_key(alias, collection):
    """Key under which a collection's version stamp is stored in L2."""
    return f"{KEY_PREFIX}:ver:{alias}:{collection}"
