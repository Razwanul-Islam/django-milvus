"""
Connection management for Milvus.

Provides thread-safe connection pooling using Django settings.
"""

import threading
from django.conf import settings
from pymilvus import MilvusClient

from .exceptions import ConnectionError


class MilvusConnectionManager:
    """Thread-safe Milvus connection manager."""

    def __init__(self):
        self._connections = {}
        self._lock = threading.Lock()

    def _get_settings(self, alias="default"):
        """Get Milvus connection settings from Django settings.

        Supports two configuration styles:

        1. DATABASES setting (recommended):
            DATABASES = {
                'milvus': {
                    'ENGINE': 'django_milvus.backend',
                    'HOST': 'localhost',
                    'PORT': 19530,
                    'USER': 'root',
                    'PASSWORD': 'Milvus',
                    'NAME': 'default',
                    'OPTIONS': {},
                }
            }

        2. MILVUS setting:
            MILVUS = {
                'default': {
                    'URI': 'http://localhost:19530',
                    'TOKEN': '',
                    'DB_NAME': 'default',
                }
            }
        """
        milvus_settings = getattr(settings, "MILVUS", None)
        if milvus_settings and alias in milvus_settings:
            return milvus_settings[alias]

        databases = getattr(settings, "DATABASES", {})
        if alias in databases:
            db = databases[alias]
            host = db.get("HOST", "localhost")
            port = db.get("PORT", 19530)
            user = db.get("USER", "")
            password = db.get("PASSWORD", "")
            db_name = db.get("NAME", "default")
            options = db.get("OPTIONS", {})
            token = ""
            if user and password:
                token = f"{user}:{password}"
            elif user:
                token = user
            return {
                "URI": f"http://{host}:{port}",
                "TOKEN": token,
                "DB_NAME": db_name,
                **options,
            }

        raise ConnectionError(
            f"No Milvus connection settings found for alias '{alias}'. "
            f"Add it to DATABASES or MILVUS in your Django settings."
        )

    def get_connection(self, alias="default"):
        """Get or create a MilvusClient connection for the given alias."""
        thread_id = threading.current_thread().ident
        key = (alias, thread_id)

        if key in self._connections:
            client = self._connections[key]
            try:
                client.list_collections()
                return client
            except Exception:
                with self._lock:
                    self._connections.pop(key, None)

        with self._lock:
            if key in self._connections:
                return self._connections[key]

            conf = self._get_settings(alias)
            try:
                kwargs = {"uri": conf["URI"]}
                if conf.get("TOKEN"):
                    kwargs["token"] = conf["TOKEN"]
                if conf.get("DB_NAME"):
                    kwargs["db_name"] = conf["DB_NAME"]
                if conf.get("TIMEOUT"):
                    kwargs["timeout"] = conf["TIMEOUT"]
                if conf.get("SECURE"):
                    kwargs["secure"] = conf["SECURE"]
                if conf.get("SERVER_PEM_PATH"):
                    kwargs["server_pem_path"] = conf["SERVER_PEM_PATH"]
                if conf.get("SERVER_NAME"):
                    kwargs["server_name"] = conf["SERVER_NAME"]

                client = MilvusClient(**kwargs)
                self._connections[key] = client
                return client
            except Exception as e:
                raise ConnectionError(
                    f"Failed to connect to Milvus (alias='{alias}'): {e}"
                ) from e

    def close_connection(self, alias="default"):
        """Close connection for the given alias in the current thread."""
        thread_id = threading.current_thread().ident
        key = (alias, thread_id)
        with self._lock:
            client = self._connections.pop(key, None)
            if client:
                try:
                    client.close()
                except Exception:
                    pass

    def close_all(self):
        """Close all connections."""
        with self._lock:
            for client in self._connections.values():
                try:
                    client.close()
                except Exception:
                    pass
            self._connections.clear()


# Global connection manager instance
connections = MilvusConnectionManager()


def get_milvus_client(alias="default"):
    """Shortcut to get a MilvusClient for the given alias."""
    return connections.get_connection(alias)
