"""
Milvus index types for vector fields.

Defines all supported Milvus index types with their parameters.
These can be used in MilvusMeta.indexes to configure vector indexes.
"""


class BaseIndex:
    """Base class for all Milvus index types."""

    index_type = None

    def __init__(self, field, metric_type="COSINE", **params):
        self.field = field
        self.metric_type = metric_type
        self.params = params

    def to_dict(self):
        """Convert to dict for pymilvus create_index."""
        result = {
            "field_name": self.field,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "params": self.params,
        }
        return result

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}(field={self.field!r}, "
            f"metric={self.metric_type})>"
        )


# ─────────────────────────────────────────────────────────
# Float Vector Indexes
# ─────────────────────────────────────────────────────────

class FLAT(BaseIndex):
    """Flat (brute-force) index. Best for small datasets.

    No additional parameters required.
    """
    index_type = "FLAT"


class IVF_FLAT(BaseIndex):
    """Inverted File with Flat quantization.

    Args:
        nlist: Number of cluster units (1-65536). Default 128.
    """
    index_type = "IVF_FLAT"

    def __init__(self, field, metric_type="COSINE", nlist=128, **params):
        params["nlist"] = nlist
        super().__init__(field, metric_type, **params)


class IVF_SQ8(BaseIndex):
    """Inverted File with Scalar Quantization (8-bit).

    Args:
        nlist: Number of cluster units. Default 128.
    """
    index_type = "IVF_SQ8"

    def __init__(self, field, metric_type="COSINE", nlist=128, **params):
        params["nlist"] = nlist
        super().__init__(field, metric_type, **params)


class IVF_PQ(BaseIndex):
    """Inverted File with Product Quantization.

    Args:
        nlist: Number of cluster units. Default 128.
        m: Number of sub-vectors. Default 8.
        nbits: Quantization bits per sub-vector. Default 8.
    """
    index_type = "IVF_PQ"

    def __init__(self, field, metric_type="COSINE", nlist=128,
                 m=8, nbits=8, **params):
        params["nlist"] = nlist
        params["m"] = m
        params["nbits"] = nbits
        super().__init__(field, metric_type, **params)


class HNSW(BaseIndex):
    """Hierarchical Navigable Small World graph.

    Best balance of speed and accuracy for most use cases.

    Args:
        M: Maximum degree of graph node. Default 16.
        efConstruction: Search breadth during build. Default 256.
    """
    index_type = "HNSW"

    def __init__(self, field, metric_type="COSINE", M=16,
                 efConstruction=256, **params):
        params["M"] = M
        params["efConstruction"] = efConstruction
        super().__init__(field, metric_type, **params)


class SCANN(BaseIndex):
    """ScaNN (Scalable Nearest Neighbors) index.

    Args:
        nlist: Number of cluster units. Default 128.
    """
    index_type = "SCANN"

    def __init__(self, field, metric_type="COSINE", nlist=128, **params):
        params["nlist"] = nlist
        super().__init__(field, metric_type, **params)


class DISKANN(BaseIndex):
    """Disk-based Approximate Nearest Neighbor index.

    No additional parameters required. Good for large datasets.
    """
    index_type = "DISKANN"


class AUTOINDEX(BaseIndex):
    """Automatically choose the best index type.

    Milvus selects the optimal index based on data characteristics.
    """
    index_type = "AUTOINDEX"


# ─────────────────────────────────────────────────────────
# Binary Vector Indexes
# ─────────────────────────────────────────────────────────

class BIN_FLAT(BaseIndex):
    """Binary flat index.

    For binary vectors only.
    """
    index_type = "BIN_FLAT"

    def __init__(self, field, metric_type="HAMMING", **params):
        super().__init__(field, metric_type, **params)


class BIN_IVF_FLAT(BaseIndex):
    """Binary inverted file with flat quantization.

    For binary vectors only.

    Args:
        nlist: Number of cluster units. Default 128.
    """
    index_type = "BIN_IVF_FLAT"

    def __init__(self, field, metric_type="HAMMING", nlist=128, **params):
        params["nlist"] = nlist
        super().__init__(field, metric_type, **params)


# ─────────────────────────────────────────────────────────
# Sparse Vector Indexes
# ─────────────────────────────────────────────────────────

class SPARSE_INVERTED_INDEX(BaseIndex):
    """Sparse inverted index for sparse vectors.

    Args:
        drop_ratio_build: Ratio of small values to drop during build. Default 0.2.
    """
    index_type = "SPARSE_INVERTED_INDEX"

    def __init__(self, field, metric_type="IP", drop_ratio_build=0.2, **params):
        params["drop_ratio_build"] = drop_ratio_build
        super().__init__(field, metric_type, **params)


class SPARSE_WAND(BaseIndex):
    """WAND index for sparse vectors.

    Args:
        drop_ratio_build: Ratio of small values to drop during build. Default 0.2.
    """
    index_type = "SPARSE_WAND"

    def __init__(self, field, metric_type="IP", drop_ratio_build=0.2, **params):
        params["drop_ratio_build"] = drop_ratio_build
        super().__init__(field, metric_type, **params)


# ─────────────────────────────────────────────────────────
# Scalar Indexes
# ─────────────────────────────────────────────────────────

class ScalarIndex(BaseIndex):
    """Index for scalar fields to speed up filtering.

    Supports: BOOL, INT8, INT16, INT32, INT64, FLOAT, DOUBLE, VARCHAR.
    """
    index_type = "STL_SORT"

    def __init__(self, field, index_type="STL_SORT", **params):
        self.index_type = index_type
        self.field = field
        self.metric_type = ""
        self.params = params

    def to_dict(self):
        return {
            "field_name": self.field,
            "index_type": self.index_type,
            "params": self.params,
        }


class TrieIndex(ScalarIndex):
    """Trie index for VARCHAR fields. Efficient for prefix queries."""

    def __init__(self, field, **params):
        super().__init__(field, index_type="Trie", **params)


class InvertedIndex(ScalarIndex):
    """Inverted index for scalar fields. Best general-purpose scalar index."""

    def __init__(self, field, **params):
        super().__init__(field, index_type="INVERTED", **params)
