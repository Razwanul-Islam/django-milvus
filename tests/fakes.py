"""
A fake MilvusClient for tests that must not touch a real server.

Records every call so tests can assert on *how many* times Milvus was
reached - which is the only way to prove a cache is actually working. A
test that checks the returned rows proves nothing: an uncached query
returns the same rows.

Usage::

    client = FakeMilvusClient(search_results=[[hit(1), hit(2)]])
    with patch_client(Document, client):
        list(Document.objects.search(vector, limit=2).cache())
        list(Document.objects.search(vector, limit=2).cache())
    assert client.calls["search"] == 1
"""

import contextlib
from collections import Counter


def hit(pk, distance=0.9, entity=None, **fields):
    """Build one search hit in the shape pymilvus returns."""
    payload = {"id": pk}
    payload.update(fields)
    if entity is not None:
        payload.update(entity)
    return {"id": pk, "distance": distance, "entity": payload}


class FakeMilvusClient:
    """Minimal MilvusClient stand-in that counts calls."""

    def __init__(self, search_results=None, query_results=None,
                 count=0, raises=None, loaded=True):
        self.calls = Counter()
        self.history = []
        self._search_results = search_results
        self._query_results = query_results if query_results is not None else []
        self._count = count
        self.raises = raises
        self.loaded = loaded
        self.load_calls = 0
        self.index_calls = 0

    # ── recording ────────────────────────────────────────

    def _record(self, name, **kwargs):
        self.calls[name] += 1
        self.history.append((name, kwargs))
        if self.raises is not None:
            raise self.raises

    def reset(self):
        self.calls.clear()
        self.history.clear()
        self.load_calls = 0
        self.index_calls = 0

    # ── reads ────────────────────────────────────────────

    def search(self, **kwargs):
        self._record("search", **kwargs)
        self._require_loaded()
        results = self._search_results
        if results is None:
            results = [[]]
        limit = kwargs.get("limit")
        if limit:
            results = [group[:limit] for group in results]
        return results

    def hybrid_search(self, **kwargs):
        self._record("hybrid_search", **kwargs)
        self._require_loaded()
        return self._search_results if self._search_results is not None else [[]]

    def query(self, **kwargs):
        self._record("query", **kwargs)
        self._require_loaded()
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": self._count}]
        results = self._query_results
        limit = kwargs.get("limit")
        if limit:
            results = results[:limit]
        return results

    def _require_loaded(self):
        if not self.loaded:
            from pymilvus.exceptions import MilvusException
            raise MilvusException(message="collection not loaded")

    # ── writes ───────────────────────────────────────────

    def insert(self, **kwargs):
        self._record("insert", **kwargs)
        return {"insert_count": len(kwargs.get("data") or []), "ids": [1]}

    def upsert(self, **kwargs):
        self._record("upsert", **kwargs)
        return {"upsert_count": len(kwargs.get("data") or [])}

    def delete(self, **kwargs):
        self._record("delete", **kwargs)
        return {"delete_count": 1}

    # ── collection management ────────────────────────────

    def load_collection(self, collection_name, **kwargs):
        self._record("load_collection", collection_name=collection_name)
        self.load_calls += 1
        self.loaded = True

    def release_collection(self, collection_name, **kwargs):
        self.calls["release_collection"] += 1
        self.loaded = False

    def create_index(self, collection_name, index_params, **kwargs):
        self.calls["create_index"] += 1
        self.index_calls += 1

    def prepare_index_params(self):
        return _FakeIndexParams()

    def list_collections(self):
        self.calls["list_collections"] += 1
        return ["documents"]

    def has_collection(self, name):
        return True

    def describe_collection(self, name):
        return {"collection_name": name}

    def get_collection_stats(self, name):
        return {"row_count": self._count}

    def get_load_state(self, name):
        return {"state": "Loaded" if self.loaded else "NotLoad"}

    def refresh_load(self, name):
        self.calls["refresh_load"] += 1

    def close(self):
        self.calls["close"] += 1


class _FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


@contextlib.contextmanager
def patch_client(model, client):
    """Point a model's ``get_client`` at ``client`` for the duration."""
    original = model.get_client
    model.get_client = classmethod(lambda cls: client)
    try:
        yield client
    finally:
        model.get_client = original
