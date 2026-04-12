"""Database introspection for Milvus backend."""

from django.db.backends.base.introspection import BaseDatabaseIntrospection


class DatabaseIntrospection(BaseDatabaseIntrospection):
    """Introspection for Milvus collections."""

    data_types_reverse = {
        "BOOL": "BooleanField",
        "INT8": "SmallIntegerField",
        "INT16": "SmallIntegerField",
        "INT32": "IntegerField",
        "INT64": "BigIntegerField",
        "FLOAT": "FloatField",
        "DOUBLE": "FloatField",
        "VARCHAR": "CharField",
        "JSON": "JSONField",
        "FLOAT_VECTOR": "FloatField",
        "BINARY_VECTOR": "BinaryField",
    }

    def get_table_list(self, cursor):
        """Return list of collections as table info."""
        try:
            client = self.connection.milvus_client
            collections = client.list_collections()
            from collections import namedtuple
            TableInfo = namedtuple("TableInfo", ["name", "type"])
            return [TableInfo(name=c, type="t") for c in collections]
        except Exception:
            return []

    def get_table_description(self, cursor, table_name):
        """Describe collection fields."""
        try:
            client = self.connection.milvus_client
            desc = client.describe_collection(table_name)
            from collections import namedtuple
            FieldInfo = namedtuple(
                "FieldInfo",
                [
                    "name", "type_code", "display_size", "internal_size",
                    "precision", "scale", "null_ok", "default", "collation",
                ],
            )
            fields = desc.get("fields", [])
            result = []
            for f in fields:
                result.append(FieldInfo(
                    name=f.get("name", ""),
                    type_code=f.get("type", ""),
                    display_size=None,
                    internal_size=f.get("params", {}).get("max_length"),
                    precision=None,
                    scale=None,
                    null_ok=True,
                    default=None,
                    collation=None,
                ))
            return result
        except Exception:
            return []

    def get_constraints(self, cursor, table_name):
        """Return constraints - primary key info for Milvus collections."""
        try:
            client = self.connection.milvus_client
            desc = client.describe_collection(table_name)
            constraints = {}
            fields = desc.get("fields", [])
            for f in fields:
                if f.get("is_primary", False):
                    constraints["primary_key"] = {
                        "columns": [f["name"]],
                        "primary_key": True,
                        "unique": True,
                        "foreign_key": None,
                        "check": False,
                        "index": True,
                        "definition": None,
                        "options": None,
                    }
            return constraints
        except Exception:
            return {}
