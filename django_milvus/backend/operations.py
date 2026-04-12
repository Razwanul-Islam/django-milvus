"""Database operations for Milvus backend."""

from django.db.backends.base.operations import BaseDatabaseOperations


class DatabaseOperations(BaseDatabaseOperations):
    """Milvus-specific database operations."""

    compiler_module = "django_milvus.backend.compiler"

    def quote_name(self, name):
        """Milvus doesn't require quoting."""
        return name

    def no_limit_value(self):
        """Return the value for unlimited results."""
        return 16384  # Milvus default max query limit

    def sql_flush(self, style, tables, *, reset_sequences=False,
                  allow_cascade=False):
        """Return SQL to flush tables - not applicable for Milvus."""
        return []

    def adapt_datefield_value(self, value):
        if value is None:
            return None
        return str(value)

    def adapt_datetimefield_value(self, value):
        if value is None:
            return None
        return str(value)

    def adapt_timefield_value(self, value):
        if value is None:
            return None
        return str(value)

    def adapt_decimalfield_value(self, value, max_digits=None,
                                 decimal_places=None):
        if value is None:
            return None
        return float(value)

    def last_insert_id(self, cursor, table_name, pk_name):
        """Return the last inserted ID."""
        return cursor.lastrowid

    def max_name_length(self):
        """Maximum length for collection/field names in Milvus."""
        return 255

    def pk_default_value(self):
        """Not applicable for Milvus auto IDs."""
        return "DEFAULT"

    def bulk_insert_sql(self, fields, placeholder_rows):
        """Not used - bulk insert handled via MilvusClient."""
        return ""
