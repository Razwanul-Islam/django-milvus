"""
Django database backend for Milvus.

This module provides a minimal Django database backend that allows
Milvus to be configured via Django's DATABASES setting. The backend
manages MilvusClient connections and provides schema operations.

Usage in settings.py:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        },
        'milvus': {
            'ENGINE': 'django_milvus.backend',
            'HOST': 'localhost',
            'PORT': 19530,
            'USER': 'root',
            'PASSWORD': 'Milvus',
            'NAME': 'default',
        }
    }
"""

import threading
from django.db.backends.base.base import BaseDatabaseWrapper
from django.db.backends.base.validation import BaseDatabaseValidation
from pymilvus import MilvusClient

from .features import DatabaseFeatures
from .operations import DatabaseOperations
from .client import DatabaseClient
from .creation import DatabaseCreation
from .introspection import DatabaseIntrospection
from .schema import DatabaseSchemaEditor
from .cursor import CursorWrapper


class DatabaseWrapper(BaseDatabaseWrapper):
    """Django database wrapper for Milvus vector database."""

    vendor = "milvus"
    display_name = "Milvus"

    # Milvus doesn't use SQL data types, but Django requires this mapping
    data_types = {
        "AutoField": "INT64",
        "BigAutoField": "INT64",
        "BinaryField": "BINARY_VECTOR",
        "BooleanField": "BOOL",
        "CharField": "VARCHAR(%(max_length)s)",
        "DateField": "VARCHAR(32)",
        "DateTimeField": "VARCHAR(64)",
        "DecimalField": "DOUBLE",
        "DurationField": "INT64",
        "FileField": "VARCHAR(%(max_length)s)",
        "FilePathField": "VARCHAR(%(max_length)s)",
        "FloatField": "FLOAT",
        "IntegerField": "INT64",
        "BigIntegerField": "INT64",
        "IPAddressField": "VARCHAR(15)",
        "GenericIPAddressField": "VARCHAR(39)",
        "JSONField": "JSON",
        "OneToOneField": "INT64",
        "PositiveBigIntegerField": "INT64",
        "PositiveIntegerField": "INT64",
        "PositiveSmallIntegerField": "INT32",
        "SlugField": "VARCHAR(%(max_length)s)",
        "SmallAutoField": "INT32",
        "SmallIntegerField": "INT16",
        "TextField": "VARCHAR(65535)",
        "TimeField": "VARCHAR(32)",
        "UUIDField": "VARCHAR(36)",
    }

    # No SQL operators in Milvus
    operators = {}

    # Pattern lookups - not used by Milvus but required by Django
    pattern_esc = ""
    pattern_ops = {}

    SchemaEditorClass = DatabaseSchemaEditor
    client_class = DatabaseClient
    creation_class = DatabaseCreation
    features_class = DatabaseFeatures
    introspection_class = DatabaseIntrospection
    ops_class = DatabaseOperations
    validation_class = BaseDatabaseValidation

    def __init__(self, settings_dict, alias="default"):
        super().__init__(settings_dict, alias=alias)
        self._milvus_client = None
        self._lock = threading.Lock()

    def get_connection_params(self):
        """Build connection parameters from Django settings."""
        settings = self.settings_dict
        host = settings.get("HOST", "localhost")
        port = settings.get("PORT", 19530)
        user = settings.get("USER", "")
        password = settings.get("PASSWORD", "")
        db_name = settings.get("NAME", "default")
        options = settings.get("OPTIONS", {})

        params = {
            "uri": f"http://{host}:{port}",
            "db_name": db_name,
        }

        if user and password:
            params["token"] = f"{user}:{password}"
        elif user:
            params["token"] = user

        # Pass through any extra options
        for key in ("timeout", "secure", "server_pem_path", "server_name"):
            if key in options:
                params[key] = options[key]

        return params

    def get_new_connection(self, conn_params):
        """Create a new MilvusClient connection."""
        return MilvusClient(**conn_params)

    def init_connection_state(self):
        """Initialize connection state."""
        pass

    def create_cursor(self, name=None):
        """Create a cursor that wraps MilvusClient operations."""
        return CursorWrapper(self.connection)

    def _set_autocommit(self, autocommit):
        """Milvus doesn't support transactions."""
        pass

    def is_usable(self):
        """Check if the connection is usable."""
        try:
            if self.connection is not None:
                self.connection.list_collections()
                return True
        except Exception:
            pass
        return False

    @property
    def milvus_client(self):
        """Get the MilvusClient instance."""
        self.ensure_connection()
        return self.connection

    def ensure_connection(self):
        """Ensure we have an active connection."""
        if self.connection is None:
            with self.wrap_database_errors:
                self.connect()

    def _close(self):
        """Close the connection."""
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass

    def _commit(self):
        """Milvus doesn't support transactions."""
        pass

    def _rollback(self):
        """Milvus doesn't support transactions."""
        pass
