"""Tests for django_milvus models."""

import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.settings")
django.setup()

import pytest
from pymilvus import DataType

from django_milvus.models import MilvusModel
from django_milvus.fields import (
    PrimaryKeyField, VarCharField, FloatVectorField,
    Int64Field, FloatField, JSONField, BoolField,
)
from django_milvus.indexes import HNSW, IVF_FLAT
from django_milvus.exceptions import ValidationError


class Document(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    title = VarCharField(max_length=512)
    content = VarCharField(max_length=4096)
    embedding = FloatVectorField(dim=128)
    score = FloatField(default=0.0)
    is_active = BoolField(default=True)
    metadata = JSONField(default=dict)

    class MilvusMeta:
        collection_name = "test_documents"
        database = "default"
        description = "Test document collection"
        enable_dynamic_field = False

    class MilvusIndexes:
        embedding_idx = HNSW(field="embedding", metric_type="COSINE")


class Article(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    title = VarCharField(max_length=256)
    body_embedding = FloatVectorField(dim=384)
    title_embedding = FloatVectorField(dim=384)

    class MilvusMeta:
        collection_name = "test_articles"
        enable_dynamic_field = True


class TestModelDefinition:
    def test_fields_collected(self):
        assert "id" in Document._fields
        assert "title" in Document._fields
        assert "content" in Document._fields
        assert "embedding" in Document._fields
        assert "score" in Document._fields
        assert "is_active" in Document._fields
        assert "metadata" in Document._fields

    def test_pk_field_detected(self):
        assert Document._pk_field is not None
        assert Document._pk_field.name == "id"
        assert Document._pk_field.auto_id is True

    def test_options_parsed(self):
        assert Document._options.collection_name == "test_documents"
        assert Document._options.database == "default"
        assert Document._options.description == "Test document collection"
        assert Document._options.enable_dynamic_field is False

    def test_indexes_parsed(self):
        assert len(Document._options.indexes) == 1
        idx = Document._options.indexes[0]
        assert idx.field == "embedding"
        assert idx.metric_type == "COSINE"
        assert idx.index_type == "HNSW"

    def test_collection_name(self):
        assert Document.get_collection_name() == "test_documents"

    def test_default_collection_name(self):
        class SimpleModel(MilvusModel):
            id = PrimaryKeyField()
            data = VarCharField(max_length=100)
        assert SimpleModel.get_collection_name() == "simplemodel"

    def test_manager_attached(self):
        assert hasattr(Document, "objects")

    def test_vector_fields(self):
        vf = Document.get_vector_fields()
        assert "embedding" in vf
        assert len(vf) == 1

    def test_multiple_vector_fields(self):
        vf = Article.get_vector_fields()
        assert "body_embedding" in vf
        assert "title_embedding" in vf
        assert len(vf) == 2


class TestModelInstances:
    def test_create_instance(self):
        doc = Document(
            title="Test",
            content="Test content",
            embedding=[0.1] * 128,
            score=0.95,
        )
        assert doc.title == "Test"
        assert doc.content == "Test content"
        assert doc.score == 0.95
        assert doc.is_active is True  # default
        assert len(doc.embedding) == 128

    def test_default_values(self):
        doc = Document(
            title="Test",
            content="Content",
            embedding=[0.0] * 128,
        )
        assert doc.score == 0.0
        assert doc.is_active is True

    def test_pk_property(self):
        doc = Document(
            title="Test",
            content="Content",
            embedding=[0.0] * 128,
        )
        assert doc.pk is None  # auto_id, not yet saved

    def test_to_dict(self):
        doc = Document(
            title="Test",
            content="Content",
            embedding=[0.1] * 128,
            score=0.5,
        )
        data = doc.to_dict()
        assert data["title"] == "Test"
        assert data["content"] == "Content"
        assert data["score"] == 0.5
        assert len(data["embedding"]) == 128
        # auto_id field should not be in dict when None
        assert "id" not in data

    def test_from_dict(self):
        data = {
            "id": 123,
            "title": "Test",
            "content": "Content",
            "embedding": [0.1] * 128,
            "score": 0.8,
            "is_active": True,
            "metadata": {"key": "value"},
        }
        doc = Document.from_dict(data)
        assert doc.pk == 123
        assert doc.title == "Test"
        assert doc.score == 0.8
        assert doc._is_new is False

    def test_unknown_fields_raise_error(self):
        with pytest.raises(ValidationError):
            Document(
                title="Test",
                content="Content",
                embedding=[0.0] * 128,
                unknown_field="value",
            )

    def test_dynamic_fields(self):
        art = Article(
            title="Test",
            body_embedding=[0.0] * 384,
            title_embedding=[0.0] * 384,
            custom_field="dynamic value",
        )
        assert art.custom_field == "dynamic value"

    def test_equality(self):
        d1 = Document.from_dict({"id": 1, "title": "A", "content": "",
                                  "embedding": [0.0]*128, "score": 0,
                                  "is_active": True, "metadata": {}})
        d2 = Document.from_dict({"id": 1, "title": "B", "content": "",
                                  "embedding": [0.0]*128, "score": 0,
                                  "is_active": True, "metadata": {}})
        d3 = Document.from_dict({"id": 2, "title": "A", "content": "",
                                  "embedding": [0.0]*128, "score": 0,
                                  "is_active": True, "metadata": {}})
        assert d1 == d2
        assert d1 != d3

    def test_repr(self):
        doc = Document.from_dict({"id": 42, "title": "Test", "content": "",
                                   "embedding": [0.0]*128, "score": 0,
                                   "is_active": True, "metadata": {}})
        assert "42" in repr(doc)
        assert "Document" in repr(doc)

    def test_schema_generation(self):
        schema = Document.get_schema()
        assert schema is not None
        field_names = [f.name for f in schema.fields]
        assert "id" in field_names
        assert "title" in field_names
        assert "embedding" in field_names


class TestModelFieldAccess:
    def test_set_and_get(self):
        doc = Document(
            title="Initial",
            content="Content",
            embedding=[0.0] * 128,
        )
        doc.title = "Updated"
        assert doc.title == "Updated"

    def test_set_validates(self):
        doc = Document(
            title="Test",
            content="Content",
            embedding=[0.0] * 128,
        )
        with pytest.raises(ValueError):
            doc.embedding = [0.0] * 64  # Wrong dimension
