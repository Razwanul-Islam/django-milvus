"""
Database router for Milvus models.

Ensures MilvusModel subclasses are routed to the Milvus database
backend and never to the default SQL database.
"""

from .models import MilvusModel


class MilvusRouter:
    """
    A Django database router that routes MilvusModel subclasses
    to the configured Milvus database.

    Usage in settings.py:
        DATABASE_ROUTERS = ['django_milvus.routers.MilvusRouter']
    """

    def _is_milvus_model(self, model):
        """Check if a model is a MilvusModel subclass."""
        return (
            isinstance(model, type)
            and issubclass(model, MilvusModel)
            and model is not MilvusModel
        )

    def db_for_read(self, model, **hints):
        """Route reads for MilvusModel to the Milvus database."""
        if self._is_milvus_model(model):
            return model.get_database_alias()
        return None

    def db_for_write(self, model, **hints):
        """Route writes for MilvusModel to the Milvus database."""
        if self._is_milvus_model(model):
            return model.get_database_alias()
        return None

    def allow_relation(self, obj1, obj2, **hints):
        """Don't allow relations between Milvus and SQL models."""
        model1 = type(obj1) if not isinstance(obj1, type) else obj1
        model2 = type(obj2) if not isinstance(obj2, type) else obj2
        is_milvus1 = self._is_milvus_model(model1)
        is_milvus2 = self._is_milvus_model(model2)

        if is_milvus1 != is_milvus2:
            return False
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """Prevent Django migrations from running on Milvus database."""
        model = hints.get("model")
        if model and self._is_milvus_model(model):
            # MilvusModel migrations are handled separately
            return False
        if db == "milvus":
            # Don't run regular Django migrations against Milvus
            return False
        return None
