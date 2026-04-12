"""Database client for Milvus backend."""

from django.db.backends.base.client import BaseDatabaseClient


class DatabaseClient(BaseDatabaseClient):
    """Command-line client for Milvus - not applicable."""

    executable_name = "milvus"

    def runshell(self, parameters=None):
        raise NotImplementedError(
            "Milvus does not have a command-line shell. "
            "Use the 'milvus_shell' management command for interactive access."
        )
