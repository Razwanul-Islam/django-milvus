"""
MilvusModel: Django-like model base class for Milvus collections.

Provides familiar model definition syntax with fields, Meta class,
and instance methods (save, delete, refresh).
"""

import copy
from collections import OrderedDict
from pymilvus import CollectionSchema, FieldSchema

from .fields import MilvusField, PrimaryKeyField
from .connection import get_milvus_client
from .utils import get_collection_name, get_database_alias
from .exceptions import (
    SchemaError, ValidationError, ObjectDoesNotExist
)


class MilvusModelOptions:
    """Holds parsed MilvusMeta options for a model."""

    def __init__(self, meta=None):
        self.collection_name = None
        self.database = "default"
        self.description = ""
        self.shards_num = None
        self.consistency_level = "Bounded"
        self.partition_key_field = None
        self.num_partitions = None
        self.enable_dynamic_field = False
        self.auto_load = True
        self.indexes = []
        # Caching is opt-in per model. None or False leaves this model
        # uncached unless a queryset asks explicitly with .cache().
        # True uses the alias defaults; a dict overrides them, e.g.
        #     cache = {"ttl": 600, "semantic": {"threshold": 0.98}}
        self.cache = None

        if meta:
            for attr in dir(meta):
                if not attr.startswith("_"):
                    setattr(self, attr, getattr(meta, attr))


class MilvusModelMeta(type):
    """Metaclass for MilvusModel that collects fields and options."""

    def __new__(mcs, name, bases, namespace):
        # Collect fields from the class and its bases
        fields = OrderedDict()
        pk_field = None

        # Inherit fields from parent classes
        for base in reversed(bases):
            if hasattr(base, "_fields"):
                fields.update(base._fields)
            if hasattr(base, "_pk_field") and base._pk_field:
                pk_field = base._pk_field

        # Collect fields from the current class
        field_attrs = {}
        for key, value in list(namespace.items()):
            if isinstance(value, MilvusField):
                field_attrs[key] = value

        # Sort by creation_counter to preserve declaration order
        sorted_fields = sorted(
            field_attrs.items(), key=lambda x: x[1].creation_counter
        )

        for key, field in sorted_fields:
            field.contribute_to_class(None, key)  # model set later
            fields[key] = field
            if isinstance(field, PrimaryKeyField):
                pk_field = field

        # Remove field attributes from namespace (we store them in _fields)
        for key in field_attrs:
            namespace.pop(key, None)

        # Parse MilvusMeta
        meta_class = namespace.pop("MilvusMeta", None)
        options = MilvusModelOptions(meta_class)

        # Parse MilvusIndexes
        indexes_class = namespace.pop("MilvusIndexes", None)
        if indexes_class:
            for attr in dir(indexes_class):
                if not attr.startswith("_"):
                    idx = getattr(indexes_class, attr)
                    if hasattr(idx, "to_dict"):
                        options.indexes.append(idx)

        namespace["_fields"] = fields
        namespace["_pk_field"] = pk_field
        namespace["_options"] = options

        cls = super().__new__(mcs, name, bases, namespace)

        # Set collection name default
        if options.collection_name is None and name != "MilvusModel":
            options.collection_name = name.lower()

        # Set model reference on fields
        for field in fields.values():
            field.model = cls

        # Set up the manager
        from .managers import MilvusManager
        if "objects" not in namespace and name != "MilvusModel":
            manager = MilvusManager()
            manager.model = cls
            cls.objects = manager

        return cls


class MilvusModel(metaclass=MilvusModelMeta):
    """
    Base class for Milvus models.

    Usage:
        from django_milvus.models import MilvusModel
        from django_milvus import fields

        class Document(MilvusModel):
            id = fields.PrimaryKeyField(auto_id=True)
            title = fields.VarCharField(max_length=512)
            embedding = fields.FloatVectorField(dim=768)

            class MilvusMeta:
                collection_name = 'documents'
                database = 'milvus'

            class MilvusIndexes:
                embedding_idx = indexes.HNSW(
                    field='embedding', metric_type='COSINE'
                )
    """

    _fields = OrderedDict()
    _pk_field = None
    _options = MilvusModelOptions()

    def __init__(self, **kwargs):
        self._data = {}
        self._is_new = True

        # Set field values from kwargs
        for field_name, field in self._fields.items():
            if field_name in kwargs:
                value = kwargs[field_name]
                field.validate(value)
                self._data[field_name] = value
            elif field.default is not None:
                default = field.default
                if callable(default):
                    default = default()
                self._data[field_name] = default
            else:
                self._data[field_name] = None

        # Check for unknown kwargs
        unknown = set(kwargs.keys()) - set(self._fields.keys())

        # If dynamic fields enabled, store extra kwargs
        if self._options.enable_dynamic_field:
            self._dynamic_data = {k: kwargs[k] for k in unknown}
        elif unknown:
            raise ValidationError(
                f"Unknown fields for {self.__class__.__name__}: {unknown}"
            )
        else:
            self._dynamic_data = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self.__class__._fields:
            return self._data.get(name)
        if hasattr(self, "_dynamic_data") and name in self._dynamic_data:
            return self._dynamic_data[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        if name.startswith("_") or name not in self.__class__._fields:
            super().__setattr__(name, value)
        else:
            field = self.__class__._fields[name]
            field.validate(value)
            self._data[name] = value

    def __repr__(self):
        pk = self.pk
        return f"<{self.__class__.__name__}: pk={pk}>"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.pk == other.pk

    def __hash__(self):
        return hash((self.__class__, self.pk))

    @property
    def pk(self):
        """Return the primary key value."""
        if self._pk_field:
            return self._data.get(self._pk_field.name)
        return None

    @pk.setter
    def pk(self, value):
        if self._pk_field:
            self._data[self._pk_field.name] = value

    def to_dict(self):
        """Convert model instance to dict for Milvus insertion."""
        data = {}
        for field_name, field in self._fields.items():
            value = self._data.get(field_name)
            if isinstance(field, PrimaryKeyField) and field.auto_id and value is None:
                continue  # Skip auto_id fields with no value
            milvus_value = field.to_milvus(value)
            if milvus_value is not None or field.nullable:
                data[field_name] = milvus_value
        if self._dynamic_data:
            data.update(self._dynamic_data)
        return data

    @classmethod
    def from_dict(cls, data):
        """Create a model instance from a Milvus result dict."""
        kwargs = {}
        dynamic_kwargs = {}

        for key, value in data.items():
            if key in cls._fields:
                field = cls._fields[key]
                kwargs[key] = field.to_python(value)
            elif cls._options.enable_dynamic_field:
                dynamic_kwargs[key] = value

        instance = cls.__new__(cls)
        instance._data = {}
        instance._dynamic_data = dynamic_kwargs
        instance._is_new = False

        for field_name in cls._fields:
            if field_name in kwargs:
                instance._data[field_name] = kwargs[field_name]
            else:
                instance._data[field_name] = None

        return instance

    @classmethod
    def get_collection_name(cls):
        """Get the Milvus collection name."""
        return cls._options.collection_name or cls.__name__.lower()

    @classmethod
    def get_database_alias(cls):
        """Get the database alias for this model."""
        return cls._options.database

    @classmethod
    def get_client(cls):
        """Get the MilvusClient for this model."""
        return get_milvus_client(cls.get_database_alias())

    @classmethod
    def get_schema(cls):
        """Build a pymilvus CollectionSchema from the model fields."""
        field_schemas = []
        for field_name, field in cls._fields.items():
            schema_kwargs = field.get_schema_kwargs()
            field_schemas.append(FieldSchema(**schema_kwargs))

        if not any(f.is_primary for f in field_schemas):
            raise SchemaError(
                f"Model {cls.__name__} has no primary key field. "
                f"Add a PrimaryKeyField."
            )

        schema = CollectionSchema(
            fields=field_schemas,
            description=cls._options.description,
            enable_dynamic_field=cls._options.enable_dynamic_field,
        )
        return schema

    @classmethod
    def get_field_names(cls, include_pk=True, include_vectors=True):
        """Get list of field names."""
        names = []
        for name, field in cls._fields.items():
            if not include_pk and isinstance(field, PrimaryKeyField):
                continue
            if not include_vectors and hasattr(field, "dim"):
                continue
            names.append(name)
        return names

    @classmethod
    def invalidate_cache(cls, reason="write"):
        """Invalidate every cached read for this model's collection.

        Called automatically by every write that goes through the ORM.
        Call it yourself after writing with a raw client, which the cache
        cannot see.
        """
        from .cache import get_cache, resolve_cache_options
        options = resolve_cache_options(cls, None)
        # Even models that never opted in may hold cached entries from an
        # explicit .cache() call, so fall back to the default alias.
        alias = options.get("alias", "default") if options else "default"
        cache = get_cache(alias)
        if cache is None:
            return 0
        return cache.invalidate(
            cls.get_collection_name(), reason=reason, sender=cls
        )

    @classmethod
    def get_vector_fields(cls):
        """Get dict of vector field names to field objects."""
        from .fields import (
            FloatVectorField, BinaryVectorField,
            Float16VectorField, BFloat16VectorField,
            SparseFloatVectorField,
        )
        vector_types = (
            FloatVectorField, BinaryVectorField,
            Float16VectorField, BFloat16VectorField,
            SparseFloatVectorField,
        )
        return {
            name: field for name, field in cls._fields.items()
            if isinstance(field, vector_types)
        }

    def save(self):
        """Save this instance to Milvus (insert or upsert)."""
        client = self.get_client()
        collection = self.get_collection_name()
        data = self.to_dict()

        if self._is_new:
            result = client.insert(
                collection_name=collection,
                data=[data],
            )
            # Set the primary key if auto_id
            if self._pk_field and self._pk_field.auto_id:
                if hasattr(result, "primary_keys"):
                    self.pk = result.primary_keys[0]
                elif isinstance(result, dict) and "ids" in result:
                    ids = result["ids"]
                    if ids:
                        self.pk = ids[0]
            self._is_new = False
        else:
            client.upsert(
                collection_name=collection,
                data=[data],
            )

        self.invalidate_cache()
        return self

    def delete(self):
        """Delete this instance from Milvus."""
        if self.pk is None:
            raise ValidationError("Cannot delete instance without primary key")

        client = self.get_client()
        collection = self.get_collection_name()
        pk_name = self._pk_field.name

        if isinstance(self.pk, str):
            filter_expr = f'{pk_name} == "{self.pk}"'
        else:
            filter_expr = f"{pk_name} == {self.pk}"

        client.delete(
            collection_name=collection,
            filter=filter_expr,
        )
        self.invalidate_cache()
        self._is_new = True

    def refresh(self):
        """Reload this instance from Milvus."""
        if self.pk is None:
            raise ValidationError("Cannot refresh instance without primary key")

        client = self.get_client()
        collection = self.get_collection_name()
        pk_name = self._pk_field.name

        if isinstance(self.pk, str):
            filter_expr = f'{pk_name} == "{self.pk}"'
        else:
            filter_expr = f"{pk_name} == {self.pk}"

        results = client.query(
            collection_name=collection,
            filter=filter_expr,
            output_fields=list(self._fields.keys()),
        )

        if not results:
            raise ObjectDoesNotExist(
                f"{self.__class__.__name__} with pk={self.pk} does not exist"
            )

        for key, value in results[0].items():
            if key in self._fields:
                field = self._fields[key]
                self._data[key] = field.to_python(value)
            elif self._options.enable_dynamic_field:
                self._dynamic_data[key] = value

        return self

    @classmethod
    def create_collection(cls, drop_existing=False):
        """Create the Milvus collection for this model."""
        from .schema import create_collection_for_model
        return create_collection_for_model(cls, drop_existing=drop_existing)

    @classmethod
    def drop_collection(cls):
        """Drop the Milvus collection for this model."""
        from .schema import drop_collection_for_model
        return drop_collection_for_model(cls)

    @classmethod
    def collection_exists(cls):
        """Check if the collection exists in Milvus."""
        client = cls.get_client()
        return client.has_collection(cls.get_collection_name())
