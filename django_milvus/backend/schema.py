"""Schema editor for Milvus backend."""

from django.db.backends.base.schema import BaseDatabaseSchemaEditor


class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):
    """Schema editor that creates/drops Milvus collections."""

    # SQL templates - not used but required by base class
    sql_create_table = ""
    sql_delete_table = ""
    sql_rename_table = ""
    sql_create_column = ""
    sql_alter_column = ""
    sql_delete_column = ""
    sql_rename_column = ""
    sql_create_unique = ""
    sql_delete_unique = ""
    sql_create_fk = ""
    sql_create_inline_fk = ""
    sql_delete_fk = ""
    sql_create_index = ""
    sql_delete_index = ""
    sql_create_pk = ""
    sql_delete_pk = ""
    sql_create_check = ""
    sql_delete_check = ""

    def __enter__(self):
        self.deferred_sql = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            for sql in self.deferred_sql:
                pass  # Deferred operations would be executed here
        self.deferred_sql = []

    def create_model(self, model):
        """Create a Milvus collection for a Django model.

        This is called during migrations for models using the Milvus backend.
        """
        pass  # Collection creation handled by MilvusModel.create_collection()

    def delete_model(self, model):
        """Drop a Milvus collection."""
        pass  # Collection deletion handled by MilvusModel.drop_collection()

    def alter_db_table(self, model, old_db_table, new_db_table):
        """Rename a collection."""
        if old_db_table != new_db_table:
            try:
                client = self.connection.milvus_client
                client.rename_collection(old_db_table, new_db_table)
            except Exception:
                pass

    def add_field(self, model, field):
        """Adding fields to existing Milvus collections is limited."""
        pass

    def remove_field(self, model, field):
        """Removing fields from Milvus collections is not supported."""
        pass

    def alter_field(self, model, old_field, new_field, strict=False):
        """Altering fields in Milvus collections is limited."""
        pass
