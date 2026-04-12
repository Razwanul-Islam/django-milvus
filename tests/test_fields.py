"""Tests for django_milvus fields."""

import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.settings")
django.setup()

import pytest
from pymilvus import DataType

from django_milvus.fields import (
    PrimaryKeyField, Int64PrimaryKey, VarCharPrimaryKey,
    BoolField, Int8Field, Int16Field, Int32Field, Int64Field,
    FloatField, DoubleField, VarCharField, JSONField, ArrayField,
    FloatVectorField, BinaryVectorField, Float16VectorField,
    BFloat16VectorField, SparseFloatVectorField,
    VectorField, CharField, IntegerField,
)


class TestPrimaryKeyField:
    def test_int64_primary_key(self):
        field = PrimaryKeyField()
        assert field.milvus_type == DataType.INT64
        assert field.auto_id is True
        assert field.primary_key is True

    def test_varchar_primary_key(self):
        field = PrimaryKeyField(field_type="varchar", max_length=64, auto_id=False)
        assert field.milvus_type == DataType.VARCHAR
        assert field.auto_id is False
        assert field.max_length == 64

    def test_int64_shortcut(self):
        field = Int64PrimaryKey()
        assert field.milvus_type == DataType.INT64
        assert field.auto_id is True

    def test_varchar_shortcut(self):
        field = VarCharPrimaryKey(max_length=100)
        assert field.milvus_type == DataType.VARCHAR
        assert field.max_length == 100
        assert field.auto_id is False

    def test_schema_kwargs(self):
        field = PrimaryKeyField(name="id")
        kwargs = field.get_schema_kwargs()
        assert kwargs["name"] == "id"
        assert kwargs["dtype"] == DataType.INT64
        assert kwargs["is_primary"] is True
        assert kwargs["auto_id"] is True


class TestScalarFields:
    def test_bool_field(self):
        field = BoolField(name="active")
        assert field.milvus_type == DataType.BOOL
        assert field.to_python(1) is True
        assert field.to_python(0) is False
        assert field.to_python(None) is None

    def test_int8_field_validation(self):
        field = Int8Field(name="tiny")
        assert field.validate(127) == 127
        with pytest.raises(ValueError):
            field.validate(128)
        with pytest.raises(ValueError):
            field.validate(-129)

    def test_int16_field_validation(self):
        field = Int16Field(name="small")
        assert field.validate(32767) == 32767
        with pytest.raises(ValueError):
            field.validate(32768)

    def test_int32_field(self):
        field = Int32Field(name="medium")
        assert field.milvus_type == DataType.INT32

    def test_int64_field(self):
        field = Int64Field(name="large")
        assert field.milvus_type == DataType.INT64

    def test_float_field(self):
        field = FloatField(name="score")
        assert field.milvus_type == DataType.FLOAT
        assert field.to_python(1) == 1.0

    def test_double_field(self):
        field = DoubleField(name="precise")
        assert field.milvus_type == DataType.DOUBLE

    def test_varchar_field(self):
        field = VarCharField(name="title", max_length=256)
        assert field.milvus_type == DataType.VARCHAR
        assert field.max_length == 256
        kwargs = field.get_schema_kwargs()
        assert kwargs["max_length"] == 256

    def test_varchar_field_validation(self):
        field = VarCharField(name="short", max_length=5)
        assert field.validate("hello") == "hello"
        with pytest.raises(ValueError):
            field.validate("toolongstring")

    def test_json_field(self):
        field = JSONField(name="meta")
        assert field.milvus_type == DataType.JSON
        assert field.to_python({"key": "value"}) == {"key": "value"}
        assert field.to_python('{"key": "value"}') == {"key": "value"}

    def test_array_field(self):
        field = ArrayField(name="tags", element_type=DataType.VARCHAR,
                          max_capacity=100, max_length=64)
        assert field.milvus_type == DataType.ARRAY
        kwargs = field.get_schema_kwargs()
        assert kwargs["element_type"] == DataType.VARCHAR
        assert kwargs["max_capacity"] == 100
        assert kwargs["max_length"] == 64


class TestVectorFields:
    def test_float_vector_field(self):
        field = FloatVectorField(name="embedding", dim=768)
        assert field.milvus_type == DataType.FLOAT_VECTOR
        assert field.dim == 768
        kwargs = field.get_schema_kwargs()
        assert kwargs["dim"] == 768

    def test_float_vector_validation(self):
        field = FloatVectorField(name="embedding", dim=3)
        assert field.validate([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
        with pytest.raises(ValueError):
            field.validate([1.0, 2.0])
        with pytest.raises(ValueError):
            field.validate("not a list")

    def test_binary_vector_field(self):
        field = BinaryVectorField(name="bin_emb", dim=128)
        assert field.milvus_type == DataType.BINARY_VECTOR
        assert field.dim == 128

    def test_binary_vector_dim_validation(self):
        with pytest.raises(ValueError):
            BinaryVectorField(name="bad", dim=100)  # Not multiple of 8

    def test_float16_vector_field(self):
        field = Float16VectorField(name="f16_emb", dim=256)
        assert field.milvus_type == DataType.FLOAT16_VECTOR
        assert field.dim == 256

    def test_bfloat16_vector_field(self):
        field = BFloat16VectorField(name="bf16_emb", dim=256)
        assert field.milvus_type == DataType.BFLOAT16_VECTOR

    def test_sparse_float_vector_field(self):
        field = SparseFloatVectorField(name="sparse")
        assert field.milvus_type == DataType.SPARSE_FLOAT_VECTOR

    def test_aliases(self):
        assert VectorField is FloatVectorField
        assert CharField is VarCharField
        assert IntegerField is Int64Field


class TestFieldCreationOrder:
    def test_creation_counter_increments(self):
        f1 = Int64Field(name="a")
        f2 = VarCharField(name="b")
        f3 = FloatVectorField(name="c", dim=128)
        assert f1.creation_counter < f2.creation_counter < f3.creation_counter
