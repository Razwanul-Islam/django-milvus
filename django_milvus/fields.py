"""
Milvus field types for Django models.

Maps Python/Django field concepts to Milvus data types. All fields
in Milvus are typed and must be declared in the collection schema.
"""

from pymilvus import DataType


class MilvusField:
    """Base class for all Milvus fields."""

    milvus_type = None
    creation_counter = 0

    def __init__(self, name=None, description="", default=None,
                 nullable=False, primary_key=False, **kwargs):
        self.name = name
        self.description = description
        self.default = default
        self.nullable = nullable
        self.primary_key = primary_key
        self.kwargs = kwargs

        # Track declaration order
        self.creation_counter = MilvusField.creation_counter
        MilvusField.creation_counter += 1

    def contribute_to_class(self, model_class, name):
        """Called when the field is added to a model class."""
        if self.name is None:
            self.name = name
        self.model = model_class

    def get_schema_kwargs(self):
        """Return kwargs for pymilvus FieldSchema."""
        schema_kwargs = {
            "name": self.name,
            "dtype": self.milvus_type,
            "description": self.description,
            "is_primary": self.primary_key,
        }
        if self.nullable:
            schema_kwargs["nullable"] = True
        # Only pass non-callable defaults to schema (callables are Python-side only)
        if self.default is not None and not callable(self.default):
            schema_kwargs["default_value"] = self.default
        schema_kwargs.update(self._extra_schema_kwargs())
        return schema_kwargs

    def _extra_schema_kwargs(self):
        """Override to add field-type-specific schema kwargs."""
        return {}

    def validate(self, value):
        """Validate a value for this field."""
        if value is None and not self.nullable and self.default is None:
            if not (self.primary_key and self.kwargs.get("auto_id", False)):
                pass  # Will be caught at insert time
        return value

    def to_python(self, value):
        """Convert a Milvus value to Python."""
        return value

    def to_milvus(self, value):
        """Convert a Python value to Milvus."""
        if value is None and self.default is not None:
            return self.default() if callable(self.default) else self.default
        return value

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"


# ─────────────────────────────────────────────────────────
# Primary Key Fields
# ─────────────────────────────────────────────────────────

class PrimaryKeyField(MilvusField):
    """Primary key field. Milvus supports INT64 or VARCHAR primary keys."""

    def __init__(self, field_type="int64", auto_id=True, max_length=128, **kwargs):
        kwargs["auto_id"] = auto_id
        self.auto_id = auto_id
        self.field_type = field_type
        self.max_length = max_length
        if field_type == "varchar":
            self.milvus_type = DataType.VARCHAR
        else:
            self.milvus_type = DataType.INT64
        super().__init__(primary_key=True, **kwargs)

    def _extra_schema_kwargs(self):
        extra = {"auto_id": self.auto_id}
        if self.milvus_type == DataType.VARCHAR:
            extra["max_length"] = self.max_length
        return extra


class Int64PrimaryKey(PrimaryKeyField):
    """INT64 auto-increment primary key."""

    def __init__(self, auto_id=True, **kwargs):
        super().__init__(field_type="int64", auto_id=auto_id, **kwargs)


class VarCharPrimaryKey(PrimaryKeyField):
    """VARCHAR primary key (user-specified)."""

    def __init__(self, max_length=128, auto_id=False, **kwargs):
        super().__init__(
            field_type="varchar", auto_id=auto_id,
            max_length=max_length, **kwargs
        )


# ─────────────────────────────────────────────────────────
# Scalar Fields
# ─────────────────────────────────────────────────────────

class BoolField(MilvusField):
    """Boolean field."""
    milvus_type = DataType.BOOL

    def to_python(self, value):
        if value is None:
            return None
        return bool(value)

    def to_milvus(self, value):
        value = super().to_milvus(value)
        if value is None:
            return None
        return bool(value)


class Int8Field(MilvusField):
    """8-bit integer field."""
    milvus_type = DataType.INT8

    def validate(self, value):
        value = super().validate(value)
        if value is not None and not (-128 <= value <= 127):
            raise ValueError(f"Int8Field value must be between -128 and 127, got {value}")
        return value


class Int16Field(MilvusField):
    """16-bit integer field."""
    milvus_type = DataType.INT16

    def validate(self, value):
        value = super().validate(value)
        if value is not None and not (-32768 <= value <= 32767):
            raise ValueError(f"Int16Field value must be between -32768 and 32767, got {value}")
        return value


class Int32Field(MilvusField):
    """32-bit integer field."""
    milvus_type = DataType.INT32

    def validate(self, value):
        value = super().validate(value)
        if value is not None and not (-2147483648 <= value <= 2147483647):
            raise ValueError(f"Int32Field value must be between -2147483648 and 2147483647, got {value}")
        return value


class Int64Field(MilvusField):
    """64-bit integer field."""
    milvus_type = DataType.INT64


class FloatField(MilvusField):
    """32-bit floating point field."""
    milvus_type = DataType.FLOAT

    def to_python(self, value):
        if value is None:
            return None
        return float(value)


class DoubleField(MilvusField):
    """64-bit floating point field."""
    milvus_type = DataType.DOUBLE

    def to_python(self, value):
        if value is None:
            return None
        return float(value)


class VarCharField(MilvusField):
    """Variable-length string field."""
    milvus_type = DataType.VARCHAR

    def __init__(self, max_length=256, **kwargs):
        self.max_length = max_length
        super().__init__(**kwargs)

    def _extra_schema_kwargs(self):
        return {"max_length": self.max_length}

    def validate(self, value):
        value = super().validate(value)
        if value is not None and len(str(value)) > self.max_length:
            raise ValueError(
                f"VarCharField value exceeds max_length={self.max_length}"
            )
        return value

    def to_python(self, value):
        if value is None:
            return None
        return str(value)

    def to_milvus(self, value):
        value = super().to_milvus(value)
        if value is None:
            return None
        return str(value)


class JSONField(MilvusField):
    """JSON field for storing structured data."""
    milvus_type = DataType.JSON

    def to_python(self, value):
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        import json
        return json.loads(value)

    def to_milvus(self, value):
        value = super().to_milvus(value)
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        return value


class ArrayField(MilvusField):
    """Array field for storing lists of scalar values."""
    milvus_type = DataType.ARRAY

    def __init__(self, element_type=DataType.INT64, max_capacity=4096,
                 max_length=256, **kwargs):
        self.element_type = element_type
        self.max_capacity = max_capacity
        self.max_length = max_length  # for VARCHAR elements
        super().__init__(**kwargs)

    def _extra_schema_kwargs(self):
        extra = {
            "element_type": self.element_type,
            "max_capacity": self.max_capacity,
        }
        if self.element_type == DataType.VARCHAR:
            extra["max_length"] = self.max_length
        return extra


# ─────────────────────────────────────────────────────────
# Vector Fields
# ─────────────────────────────────────────────────────────

class FloatVectorField(MilvusField):
    """Dense float vector field. The primary field type for embeddings."""
    milvus_type = DataType.FLOAT_VECTOR

    def __init__(self, dim, **kwargs):
        self.dim = dim
        super().__init__(**kwargs)

    def _extra_schema_kwargs(self):
        return {"dim": self.dim}

    def validate(self, value):
        value = super().validate(value)
        if value is not None:
            if not isinstance(value, (list, tuple)):
                raise ValueError("FloatVectorField value must be a list or tuple")
            if len(value) != self.dim:
                raise ValueError(
                    f"FloatVectorField expects dim={self.dim}, "
                    f"got vector of length {len(value)}"
                )
        return value


class BinaryVectorField(MilvusField):
    """Binary vector field."""
    milvus_type = DataType.BINARY_VECTOR

    def __init__(self, dim, **kwargs):
        self.dim = dim
        if dim % 8 != 0:
            raise ValueError("BinaryVectorField dim must be a multiple of 8")
        super().__init__(**kwargs)

    def _extra_schema_kwargs(self):
        return {"dim": self.dim}


class Float16VectorField(MilvusField):
    """Float16 vector field."""
    milvus_type = DataType.FLOAT16_VECTOR

    def __init__(self, dim, **kwargs):
        self.dim = dim
        super().__init__(**kwargs)

    def _extra_schema_kwargs(self):
        return {"dim": self.dim}


class BFloat16VectorField(MilvusField):
    """BFloat16 vector field."""
    milvus_type = DataType.BFLOAT16_VECTOR

    def __init__(self, dim, **kwargs):
        self.dim = dim
        super().__init__(**kwargs)

    def _extra_schema_kwargs(self):
        return {"dim": self.dim}


class SparseFloatVectorField(MilvusField):
    """Sparse float vector field for sparse embeddings."""
    milvus_type = DataType.SPARSE_FLOAT_VECTOR

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ─────────────────────────────────────────────────────────
# Convenience aliases matching common naming patterns
# ─────────────────────────────────────────────────────────

# Common aliases
VectorField = FloatVectorField
CharField = VarCharField
IntegerField = Int64Field
BigIntegerField = Int64Field
SmallIntegerField = Int16Field
TextField = VarCharField
