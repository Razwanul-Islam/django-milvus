"""Database creation for Milvus backend."""

from django.db.backends.base.creation import BaseDatabaseCreation


class DatabaseCreation(BaseDatabaseCreation):
    """Handles creating and destroying Milvus databases."""

    def create_test_db(self, verbosity=1, autoclobber=False, serialize=True,
                       keepdb=False):
        """Create a test database. For Milvus, we use a separate db_name."""
        test_database_name = self._get_test_db_name()
        if verbosity >= 1:
            print(f"Creating test database '{test_database_name}'...")
        # Milvus databases are created on-the-fly
        self.connection.settings_dict["NAME"] = test_database_name
        return test_database_name

    def destroy_test_db(self, old_database_name=None, verbosity=1,
                        keepdb=False, suffix=None):
        """Destroy the test database."""
        if not keepdb:
            test_database_name = self.connection.settings_dict["NAME"]
            if verbosity >= 1:
                print(f"Destroying test database '{test_database_name}'...")
            try:
                client = self.connection.milvus_client
                # Drop all collections in the test database
                collections = client.list_collections()
                for collection in collections:
                    client.drop_collection(collection)
            except Exception:
                pass
        if old_database_name:
            self.connection.settings_dict["NAME"] = old_database_name
