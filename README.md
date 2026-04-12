# django-milvus

Django integration for [Milvus](https://milvus.io/) vector database. Use Milvus as a Django secondary database with a familiar ORM-like interface for storing and searching vector embeddings.

[![PyPI version](https://badge.fury.io/py/django-milvus.svg)](https://pypi.org/project/django-milvus/)
[![Python](https://img.shields.io/pypi/pyversions/django-milvus.svg)](https://pypi.org/project/django-milvus/)
[![Django](https://img.shields.io/badge/django-4.2%20%7C%205.0%20%7C%205.1-green.svg)](https://djangoproject.com/)

## Features

- **Django ORM-like interface** — Define Milvus collections as Python model classes with typed fields
- **Vector similarity search** — Search by embedding vectors with COSINE, L2, or IP metrics
- **All Milvus field types** — FloatVector, BinaryVector, SparseVector, VarChar, JSON, Array, scalar types
- **All index types** — HNSW, IVF_FLAT, IVF_PQ, DISKANN, AUTOINDEX, and more
- **Django-style filtering** — `filter(score__gt=0.5)`, `filter(category__in=[...])`, `exclude(...)`
- **Chainable QuerySet** — `.filter().limit().only().search()` with lazy evaluation
- **Bulk operations** — `bulk_create()`, `upsert()` with automatic batching
- **Partition support** — Create and query specific partitions
- **Collection management** — Management commands for creating, inspecting, and dropping collections
- **Alias & RBAC management** — Full support for Milvus aliases, users, roles, and privileges
- **Django settings integration** — Configure connections via `DATABASES` or `MILVUS` settings
- **Database router** — Routes MilvusModel operations to the correct backend
- **Interactive shell** — `milvus_shell` management command with connected client

## Installation

```bash
pip install django-milvus
```

### Requirements

- Python >= 3.9
- Django >= 4.2
- pymilvus >= 2.4.0
- A running Milvus instance (2.4.x recommended)

## Quick Start

### 1. Configure Django Settings

```python
# settings.py
INSTALLED_APPS = [
    # ... your apps
    'django_milvus',
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    },
    'milvus': {
        'ENGINE': 'django_milvus.backend',
        'HOST': 'localhost',
        'PORT': 19530,
        'USER': '',           # optional
        'PASSWORD': '',       # optional
        'NAME': 'default',    # Milvus database name
    },
}

DATABASE_ROUTERS = ['django_milvus.routers.MilvusRouter']
```

Or use the `MILVUS` setting for more control:

```python
MILVUS = {
    'default': {
        'URI': 'http://localhost:19530',
        'TOKEN': 'root:Milvus',
        'DB_NAME': 'default',
        'TIMEOUT': 30,
    },
}
```

### 2. Define Models

```python
# myapp/models.py
from django_milvus.models import MilvusModel
from django_milvus.fields import (
    PrimaryKeyField, VarCharField, FloatVectorField,
    Int64Field, FloatField, BoolField, JSONField,
)
from django_milvus.indexes import HNSW, InvertedIndex


class Document(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    title = VarCharField(max_length=512)
    content = VarCharField(max_length=8192)
    embedding = FloatVectorField(dim=768)
    category = VarCharField(max_length=64)
    score = FloatField(default=0.0)
    is_published = BoolField(default=True)
    metadata = JSONField(default=dict)

    class MilvusMeta:
        collection_name = 'documents'
        database = 'milvus'                 # matches DATABASES key
        description = 'Document embeddings'
        consistency_level = 'Bounded'
        enable_dynamic_field = False

    class MilvusIndexes:
        embedding_idx = HNSW(
            field='embedding',
            metric_type='COSINE',
            M=16,
            efConstruction=256,
        )
        category_idx = InvertedIndex(field='category')
```

### 3. Create the Collection

```bash
python manage.py milvus_sync
```

Or programmatically:

```python
Document.create_collection()          # Create if not exists
Document.create_collection(drop_existing=True)  # Recreate
```

### 4. Insert Data

```python
# Single insert
doc = Document(
    title="Introduction to AI",
    content="Artificial intelligence is...",
    embedding=[0.1, 0.2, ...],  # 768-dim vector
    category="tech",
    score=0.95,
)
doc.save()
print(doc.pk)  # Auto-generated ID

# Bulk insert
docs = [
    Document(title="Doc 1", content="...", embedding=[...], category="tech"),
    Document(title="Doc 2", content="...", embedding=[...], category="science"),
]
Document.objects.bulk_create(instances=docs)

# Insert from raw dicts
Document.objects.bulk_create(data=[
    {"title": "Doc 3", "content": "...", "embedding": [...], "category": "tech"},
])

# Create shortcut
doc = Document.objects.create(
    title="Quick doc",
    content="...",
    embedding=[...],
    category="tech",
)
```

### 5. Vector Similarity Search

```python
query_vector = get_embedding("What is machine learning?")  # Your embedding function

# Basic search
results = Document.objects.search(
    vector=query_vector,
    vector_field='embedding',
    limit=10,
    metric_type='COSINE',
)

for result in results:
    print(f"{result.entity.title} (distance: {result.distance})")

# Search with filters
results = Document.objects.filter(
    category='tech',
    is_published=True,
).search(
    vector=query_vector,
    limit=5,
)

# Search with specific output fields
results = Document.objects.search(
    vector=query_vector,
    limit=20,
    output_fields=['title', 'category', 'score'],
)

# Search with custom parameters
results = Document.objects.search(
    vector=query_vector,
    limit=10,
    search_params={"ef": 128},  # HNSW search parameter
)

# Auto-detect vector field (works when model has single vector field)
results = Document.objects.search(vector=query_vector, limit=10)
```

### 6. Query and Filter

```python
# Get all (with limit)
docs = Document.objects.all().limit(100)

# Filter with Django-style lookups
docs = Document.objects.filter(category='tech')
docs = Document.objects.filter(score__gt=0.5)
docs = Document.objects.filter(score__gte=0.5, score__lte=1.0)
docs = Document.objects.filter(category__in=['tech', 'science'])
docs = Document.objects.filter(title__like='AI%')

# Raw Milvus filter expressions
docs = Document.objects.filter(expr='category == "tech" and score > 0.5')

# Chainable operations
docs = (
    Document.objects
    .filter(category='tech')
    .filter(is_published=True)
    .limit(50)
    .only('title', 'score')
)

# Exclude
docs = Document.objects.exclude(category='spam')

# Get single object
doc = Document.objects.get(id=42)
doc = Document.objects.get_or_none(id=999)
first_doc = Document.objects.filter(category='tech').first()

# Count and exists
count = Document.objects.filter(category='tech').count()
has_docs = Document.objects.filter(score__gt=0.9).exists()
```

### 7. Update and Delete

```python
# Update single instance
doc = Document.objects.get(id=42)
doc.title = "Updated Title"
doc.save()  # Upserts

# Upsert multiple
Document.objects.upsert(instances=[doc1, doc2, doc3])

# Delete by filter
Document.objects.delete(category='spam')
Document.objects.filter(score__lt=0.1).delete()

# Delete by IDs
Document.objects.delete_by_ids([1, 2, 3])

# Delete single instance
doc.delete()
```

### 8. Partitions

```python
from django_milvus import schema

# Create partitions
schema.create_partition('documents', 'tech_docs')
schema.create_partition('documents', 'science_docs')

# Query specific partitions
results = (
    Document.objects
    .using_partitions('tech_docs')
    .filter(score__gt=0.5)
    .limit(10)
)

# Search within partitions
results = (
    Document.objects
    .using_partitions('tech_docs')
    .search(vector=query_vector, limit=10)
)

# List/manage partitions
schema.list_partitions('documents')
schema.has_partition('documents', 'tech_docs')
schema.drop_partition('documents', 'tech_docs')
```

## Complete Field Reference

| Field | Milvus Type | Notes |
|-------|-------------|-------|
| `PrimaryKeyField` | INT64 or VARCHAR | `auto_id=True` for auto-increment |
| `Int64PrimaryKey` | INT64 | Shortcut for INT64 PK |
| `VarCharPrimaryKey` | VARCHAR | Shortcut for VARCHAR PK |
| `BoolField` | BOOL | Boolean values |
| `Int8Field` | INT8 | -128 to 127 |
| `Int16Field` | INT16 | -32768 to 32767 |
| `Int32Field` | INT32 | 32-bit integer |
| `Int64Field` | INT64 | 64-bit integer |
| `FloatField` | FLOAT | 32-bit float |
| `DoubleField` | DOUBLE | 64-bit float |
| `VarCharField` | VARCHAR | `max_length` required |
| `JSONField` | JSON | Dict/list data |
| `ArrayField` | ARRAY | `element_type`, `max_capacity` |
| `FloatVectorField` | FLOAT_VECTOR | `dim` required |
| `BinaryVectorField` | BINARY_VECTOR | `dim` (multiple of 8) |
| `Float16VectorField` | FLOAT16_VECTOR | `dim` required |
| `BFloat16VectorField` | BFLOAT16_VECTOR | `dim` required |
| `SparseFloatVectorField` | SPARSE_FLOAT_VECTOR | Sparse embeddings |

**Aliases:** `VectorField` = `FloatVectorField`, `CharField` = `VarCharField`, `IntegerField` = `Int64Field`

## Complete Index Reference

| Index | Type | Best For |
|-------|------|----------|
| `FLAT` | Exact | Small datasets, perfect accuracy |
| `IVF_FLAT` | Approximate | Good balance, `nlist` param |
| `IVF_SQ8` | Approximate | Lower memory, `nlist` param |
| `IVF_PQ` | Approximate | Large datasets, `nlist`, `m`, `nbits` |
| `HNSW` | Graph-based | Best speed/accuracy, `M`, `efConstruction` |
| `SCANN` | Approximate | Fast, `nlist` param |
| `DISKANN` | Disk-based | Very large datasets |
| `AUTOINDEX` | Auto | Let Milvus choose |
| `BIN_FLAT` | Binary | Binary vectors |
| `BIN_IVF_FLAT` | Binary | Binary vectors, `nlist` |
| `SPARSE_INVERTED_INDEX` | Sparse | Sparse vectors |
| `SPARSE_WAND` | Sparse | Sparse vectors |
| `ScalarIndex` | Scalar | Filter acceleration |
| `TrieIndex` | Scalar | VARCHAR prefix queries |
| `InvertedIndex` | Scalar | General-purpose filtering |

### Index Examples

All index classes are imported from `django_milvus.indexes`. Define them inside a `MilvusIndexes` inner class on your model, or create them programmatically via `schema.create_index()`.

#### FLAT — Brute-Force (small datasets, 100% recall)

```python
from django_milvus.models import MilvusModel
from django_milvus.fields import PrimaryKeyField, VarCharField, FloatVectorField
from django_milvus.indexes import FLAT

class SmallCollection(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    text = VarCharField(max_length=512)
    embedding = FloatVectorField(dim=128)

    class MilvusMeta:
        collection_name = 'small_collection'

    class MilvusIndexes:
        embedding_idx = FLAT(field='embedding', metric_type='L2')
```

#### HNSW — Graph-Based (best speed/accuracy trade-off)

```python
from django_milvus.indexes import HNSW

class ArticleEmbedding(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    title = VarCharField(max_length=256)
    embedding = FloatVectorField(dim=768)

    class MilvusMeta:
        collection_name = 'articles'

    class MilvusIndexes:
        # M: max connections per node (higher = better recall, more memory)
        # efConstruction: search breadth during build (higher = better quality)
        embedding_idx = HNSW(
            field='embedding',
            metric_type='COSINE',
            M=16,
            efConstruction=256,
        )
```

#### IVF_FLAT — Inverted File Index (balanced for medium datasets)

```python
from django_milvus.indexes import IVF_FLAT

class ProductEmbedding(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    name = VarCharField(max_length=256)
    embedding = FloatVectorField(dim=512)

    class MilvusMeta:
        collection_name = 'products'

    class MilvusIndexes:
        # nlist: number of clusters (higher = faster search, lower recall)
        embedding_idx = IVF_FLAT(
            field='embedding',
            metric_type='IP',
            nlist=256,
        )
```

#### IVF_PQ — Product Quantization (large datasets, lower memory)

```python
from django_milvus.indexes import IVF_PQ

class LargeScaleDoc(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    embedding = FloatVectorField(dim=768)

    class MilvusMeta:
        collection_name = 'large_docs'

    class MilvusIndexes:
        # m: sub-vector count (must divide dim evenly)
        # nbits: quantization bits per sub-vector
        embedding_idx = IVF_PQ(
            field='embedding',
            metric_type='L2',
            nlist=128,
            m=24,        # 768 / 24 = 32-dim sub-vectors
            nbits=8,
        )
```

#### IVF_SQ8 — Scalar Quantization (lower memory than IVF_FLAT)

```python
from django_milvus.indexes import IVF_SQ8

class CompressedDoc(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    embedding = FloatVectorField(dim=384)

    class MilvusMeta:
        collection_name = 'compressed_docs'

    class MilvusIndexes:
        embedding_idx = IVF_SQ8(
            field='embedding',
            metric_type='COSINE',
            nlist=128,
        )
```

#### SCANN — Scalable Nearest Neighbors (fast approximate search)

```python
from django_milvus.indexes import SCANN

class FastSearchDoc(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    embedding = FloatVectorField(dim=256)

    class MilvusMeta:
        collection_name = 'fast_search'

    class MilvusIndexes:
        embedding_idx = SCANN(
            field='embedding',
            metric_type='COSINE',
            nlist=128,
        )
```

#### DISKANN — Disk-Based Index (billion-scale datasets)

```python
from django_milvus.indexes import DISKANN

class HugeCollection(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    embedding = FloatVectorField(dim=768)

    class MilvusMeta:
        collection_name = 'huge_collection'

    class MilvusIndexes:
        # No extra params needed — data is indexed on disk
        embedding_idx = DISKANN(field='embedding', metric_type='L2')
```

#### AUTOINDEX — Let Milvus Choose

```python
from django_milvus.indexes import AUTOINDEX

class AutoDoc(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    embedding = FloatVectorField(dim=768)

    class MilvusMeta:
        collection_name = 'auto_docs'

    class MilvusIndexes:
        embedding_idx = AUTOINDEX(field='embedding', metric_type='COSINE')
```

#### BIN_FLAT / BIN_IVF_FLAT — Binary Vector Indexes

```python
from django_milvus.fields import BinaryVectorField
from django_milvus.indexes import BIN_FLAT, BIN_IVF_FLAT

class BinaryHashModel(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    hash_vector = BinaryVectorField(dim=256)  # must be multiple of 8

    class MilvusMeta:
        collection_name = 'binary_hashes'

    class MilvusIndexes:
        # Use HAMMING or JACCARD metric for binary vectors
        hash_idx = BIN_FLAT(field='hash_vector', metric_type='HAMMING')

        # Or with clustering for larger datasets:
        # hash_idx = BIN_IVF_FLAT(
        #     field='hash_vector', metric_type='JACCARD', nlist=64
        # )
```

#### SPARSE_INVERTED_INDEX / SPARSE_WAND — Sparse Vector Indexes

```python
from django_milvus.fields import SparseFloatVectorField
from django_milvus.indexes import SPARSE_INVERTED_INDEX, SPARSE_WAND

class SparseEmbeddingModel(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    title = VarCharField(max_length=256)
    sparse_embedding = SparseFloatVectorField()

    class MilvusMeta:
        collection_name = 'sparse_docs'

    class MilvusIndexes:
        # drop_ratio_build: fraction of small values to discard (saves space)
        sparse_idx = SPARSE_INVERTED_INDEX(
            field='sparse_embedding',
            metric_type='IP',
            drop_ratio_build=0.2,
        )

        # Alternative: SPARSE_WAND is faster for top-k retrieval
        # sparse_idx = SPARSE_WAND(
        #     field='sparse_embedding', metric_type='IP', drop_ratio_build=0.2
        # )
```

#### Scalar Indexes — Speed Up Filtering

```python
from django_milvus.fields import Int64Field, BoolField
from django_milvus.indexes import InvertedIndex, TrieIndex, ScalarIndex

class FilterableDoc(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    category = VarCharField(max_length=64)
    author = VarCharField(max_length=128)
    view_count = Int64Field()
    is_published = BoolField(default=True)
    embedding = FloatVectorField(dim=768)

    class MilvusMeta:
        collection_name = 'filterable_docs'

    class MilvusIndexes:
        # HNSW for vector search
        emb_idx = HNSW(field='embedding', metric_type='COSINE')

        # InvertedIndex — best general-purpose scalar index
        category_idx = InvertedIndex(field='category')

        # TrieIndex — optimized for VARCHAR prefix queries (like "AI%")
        author_idx = TrieIndex(field='author')

        # STL_SORT — good for numeric range queries
        views_idx = ScalarIndex(field='view_count', index_type='STL_SORT')
```

#### Programmatic Index Creation (without MilvusIndexes)

```python
from django_milvus import schema

# Create an HNSW index on an existing collection
schema.create_index(
    'documents',
    'embedding',
    index_type='HNSW',
    metric_type='COSINE',
    params={'M': 16, 'efConstruction': 256},
)

# Create a scalar index
schema.create_index(
    'documents',
    'category',
    index_type='INVERTED',
    metric_type='',
)

# List, inspect, and drop indexes
schema.list_indexes('documents')
schema.describe_index('documents', 'embedding')
schema.drop_index('documents', 'embedding')
```

#### Multiple Indexes on One Model (Multi-Vector + Scalar)

```python
class HybridSearchDoc(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    title = VarCharField(max_length=256)
    category = VarCharField(max_length=64)
    dense_embedding = FloatVectorField(dim=768)
    sparse_embedding = SparseFloatVectorField()

    class MilvusMeta:
        collection_name = 'hybrid_docs'

    class MilvusIndexes:
        dense_idx = HNSW(field='dense_embedding', metric_type='COSINE', M=32)
        sparse_idx = SPARSE_INVERTED_INDEX(field='sparse_embedding', metric_type='IP')
        category_idx = InvertedIndex(field='category')

# Search dense vectors with scalar filter
results = HybridSearchDoc.objects.filter(
    category='tech',
).search(
    vector=dense_query,
    vector_field='dense_embedding',
    limit=10,
)
```

## Filter Lookups

| Lookup | Milvus Expression |
|--------|-------------------|
| `field=value` | `field == value` |
| `field__eq=value` | `field == value` |
| `field__ne=value` | `field != value` |
| `field__gt=value` | `field > value` |
| `field__gte=value` | `field >= value` |
| `field__lt=value` | `field < value` |
| `field__lte=value` | `field <= value` |
| `field__in=[...]` | `field in [...]` |
| `field__nin=[...]` | `field not in [...]` |
| `field__like="pat"` | `field like "pat"` |
| `field__exists=True` | `exists field` |
| `field__json_contains=v` | `json_contains(field, v)` |
| `field__array_contains=v` | `array_contains(field, v)` |

## Schema Management Functions

```python
from django_milvus import schema

# Collections
schema.list_collections()
schema.describe_collection('documents')
schema.has_collection('documents')
schema.rename_collection('old_name', 'new_name')
schema.get_collection_stats('documents')
schema.load_collection('documents')
schema.release_collection('documents')
schema.get_load_state('documents')
schema.drop_collection('documents')

# Indexes
schema.create_index('documents', 'embedding', index_type='HNSW',
                    metric_type='COSINE', params={'M': 16})
schema.list_indexes('documents')
schema.describe_index('documents', 'index_name')
schema.drop_index('documents', 'index_name')

# Partitions
schema.create_partition('documents', 'partition_a')
schema.drop_partition('documents', 'partition_a')
schema.has_partition('documents', 'partition_a')
schema.list_partitions('documents')
schema.load_partitions('documents', ['partition_a'])
schema.release_partitions('documents', ['partition_a'])

# Aliases
schema.create_alias('documents', 'docs_alias')
schema.drop_alias('docs_alias')
schema.alter_alias('documents_v2', 'docs_alias')
schema.describe_alias('docs_alias')
schema.list_aliases('documents')

# User & RBAC
schema.create_user('alice', 'password123')
schema.drop_user('alice')
schema.update_password('alice', 'old_pass', 'new_pass')
schema.list_users()
schema.describe_user('alice')
schema.create_role('reader')
schema.drop_role('reader')
schema.list_roles()
schema.grant_role('alice', 'reader')
schema.revoke_role('alice', 'reader')
schema.grant_privilege('reader', 'Collection', 'documents', 'Search')
schema.revoke_privilege('reader', 'Collection', 'documents', 'Search')
```

## Management Commands

```bash
# Create collections for all MilvusModel classes
python manage.py milvus_sync
python manage.py milvus_sync --drop-existing
python manage.py milvus_sync --models myapp.models.Document

# Show collection statistics
python manage.py milvus_stats
python manage.py milvus_stats --collection documents --verbose

# Drop collections
python manage.py milvus_drop --collection documents
python manage.py milvus_drop --all --yes

# Interactive shell
python manage.py milvus_shell
```

## Advanced Usage

### Direct Client Access

```python
# Access the underlying pymilvus MilvusClient
client = Document.objects.get_client()
client.list_collections()

# From connection manager
from django_milvus.connection import get_milvus_client
client = get_milvus_client('milvus')
```

### Raw Operations

```python
# Raw query with Milvus expressions
results = Document.objects.query_raw(
    filter_expr='category == "tech" and score > 0.5',
    output_fields=['title', 'score'],
    limit=100,
)

# Raw search
results = Document.objects.search_raw(
    data=[[0.1, 0.2, ...]],
    anns_field='embedding',
    limit=10,
    search_params={"metric_type": "COSINE", "params": {"ef": 128}},
    filter_expr='is_published == true',
)

# Raw insert
Document.objects.insert_raw([
    {"title": "Doc", "embedding": [...], "category": "tech"},
])
```

### Multiple Vector Fields

```python
class MultiVectorDoc(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    title = VarCharField(max_length=256)
    title_embedding = FloatVectorField(dim=384)
    content_embedding = FloatVectorField(dim=768)

    class MilvusMeta:
        collection_name = 'multi_vector_docs'

    class MilvusIndexes:
        title_idx = HNSW(field='title_embedding', metric_type='COSINE')
        content_idx = HNSW(field='content_embedding', metric_type='COSINE')

# Search specific vector field
results = MultiVectorDoc.objects.search(
    vector=title_query,
    vector_field='title_embedding',
    limit=10,
)
```

### Consistency Levels

```python
# Strong consistency (reads reflect latest writes)
results = Document.objects.consistency('Strong').filter(category='tech')

# Session consistency
results = Document.objects.consistency('Session').search(vector=v, limit=10)
```

### Dynamic Fields

```python
class FlexDoc(MilvusModel):
    id = PrimaryKeyField(auto_id=True)
    embedding = FloatVectorField(dim=128)

    class MilvusMeta:
        collection_name = 'flex_docs'
        enable_dynamic_field = True

# Store arbitrary fields
doc = FlexDoc(embedding=[...], custom_field="value", tags=["a", "b"])
doc.save()
```

## Publishing to PyPI

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Upload to PyPI
twine upload dist/*

# Upload to Test PyPI first
twine upload --repository testpypi dist/*
```

## License

MIT License. See [LICENSE](LICENSE) for details.
