"""Cursor wrapper for Milvus backend."""


class CursorWrapper:
    """
    Minimal cursor implementation wrapping MilvusClient.

    Since Milvus doesn't use SQL, this cursor provides a compatibility
    layer that Django's internals can interact with without crashing.
    Real operations go through MilvusManager and MilvusQuerySet.
    """

    def __init__(self, client):
        self.client = client
        self.lastrowid = None
        self.rowcount = -1
        self._results = []
        self._result_index = 0

    def execute(self, sql, params=None):
        """Execute is a no-op for Milvus - operations go through pymilvus."""
        self._results = []
        self._result_index = 0
        return None

    def executemany(self, sql, param_list):
        """Execute many is a no-op for Milvus."""
        self._results = []
        self._result_index = 0
        return None

    def fetchone(self):
        """Fetch one result."""
        if self._result_index < len(self._results):
            result = self._results[self._result_index]
            self._result_index += 1
            return result
        return None

    def fetchmany(self, size=None):
        """Fetch many results."""
        if size is None:
            size = len(self._results) - self._result_index
        end = min(self._result_index + size, len(self._results))
        results = self._results[self._result_index:end]
        self._result_index = end
        return results

    def fetchall(self):
        """Fetch all results."""
        results = self._results[self._result_index:]
        self._result_index = len(self._results)
        return results

    def close(self):
        """Close the cursor."""
        self._results = []
        self._result_index = 0

    @property
    def description(self):
        """Column description - not applicable for Milvus."""
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return iter(self._results)
