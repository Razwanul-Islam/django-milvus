"""Tests for django_milvus indexes."""

import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.settings")
django.setup()

from django_milvus.indexes import (
    FLAT, IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, SCANN, DISKANN, AUTOINDEX,
    BIN_FLAT, BIN_IVF_FLAT,
    SPARSE_INVERTED_INDEX, SPARSE_WAND,
    ScalarIndex, TrieIndex, InvertedIndex,
)


class TestVectorIndexes:
    def test_flat(self):
        idx = FLAT(field="emb", metric_type="L2")
        d = idx.to_dict()
        assert d["index_type"] == "FLAT"
        assert d["field_name"] == "emb"
        assert d["metric_type"] == "L2"

    def test_ivf_flat(self):
        idx = IVF_FLAT(field="emb", nlist=256)
        d = idx.to_dict()
        assert d["index_type"] == "IVF_FLAT"
        assert d["params"]["nlist"] == 256

    def test_ivf_sq8(self):
        idx = IVF_SQ8(field="emb")
        assert idx.index_type == "IVF_SQ8"

    def test_ivf_pq(self):
        idx = IVF_PQ(field="emb", m=16, nbits=8)
        d = idx.to_dict()
        assert d["params"]["m"] == 16
        assert d["params"]["nbits"] == 8

    def test_hnsw(self):
        idx = HNSW(field="emb", M=32, efConstruction=512)
        d = idx.to_dict()
        assert d["index_type"] == "HNSW"
        assert d["params"]["M"] == 32
        assert d["params"]["efConstruction"] == 512

    def test_scann(self):
        idx = SCANN(field="emb", nlist=64)
        d = idx.to_dict()
        assert d["params"]["nlist"] == 64

    def test_diskann(self):
        idx = DISKANN(field="emb")
        assert idx.index_type == "DISKANN"

    def test_autoindex(self):
        idx = AUTOINDEX(field="emb")
        assert idx.index_type == "AUTOINDEX"

    def test_default_metric(self):
        idx = HNSW(field="emb")
        assert idx.metric_type == "COSINE"


class TestBinaryIndexes:
    def test_bin_flat(self):
        idx = BIN_FLAT(field="bin_emb")
        assert idx.index_type == "BIN_FLAT"
        assert idx.metric_type == "HAMMING"

    def test_bin_ivf_flat(self):
        idx = BIN_IVF_FLAT(field="bin_emb", nlist=64)
        d = idx.to_dict()
        assert d["params"]["nlist"] == 64


class TestSparseIndexes:
    def test_sparse_inverted(self):
        idx = SPARSE_INVERTED_INDEX(field="sparse_emb")
        d = idx.to_dict()
        assert d["index_type"] == "SPARSE_INVERTED_INDEX"
        assert d["metric_type"] == "IP"
        assert d["params"]["drop_ratio_build"] == 0.2

    def test_sparse_wand(self):
        idx = SPARSE_WAND(field="sparse_emb", drop_ratio_build=0.1)
        d = idx.to_dict()
        assert d["params"]["drop_ratio_build"] == 0.1


class TestScalarIndexes:
    def test_scalar_index(self):
        idx = ScalarIndex(field="age")
        d = idx.to_dict()
        assert d["field_name"] == "age"
        assert d["index_type"] == "STL_SORT"

    def test_trie_index(self):
        idx = TrieIndex(field="name")
        d = idx.to_dict()
        assert d["index_type"] == "Trie"

    def test_inverted_index(self):
        idx = InvertedIndex(field="category")
        d = idx.to_dict()
        assert d["index_type"] == "INVERTED"

    def test_repr(self):
        idx = HNSW(field="emb", metric_type="COSINE")
        r = repr(idx)
        assert "HNSW" in r
        assert "emb" in r
        assert "COSINE" in r
