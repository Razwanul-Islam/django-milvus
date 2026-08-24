# django-milvus

Django integration for [Milvus](https://milvus.io/) vector database. Use Milvus as a Django secondary database with a familiar ORM-like interface for storing and searching vector embeddings.

[![PyPI version](https://badge.fury.io/py/django-milvus.svg)](https://pypi.org/project/django-milvus/)
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
- **[Production caching layer](#caching)** — Opt-in, tiered RAM + Redis cache with nine eviction algorithms, byte-accurate memory bounds, **semantic (nearest-vector) caching**, and automatic invalidation on writes

## Installation

```bash
pip install django-milvus

# Optional: shared Redis cache tier
pip install django-milvus[cache]

# Optional: cache accelerators (hnswlib, lz4, msgpack, psutil)
pip install django-milvus[fast]
```

### Requirements

- Python >= 3.9
- Django >= 4.2
- pymilvus >= 2.4.0
- numpy >= 1.21
- A running Milvus instance (2.4.x recommended)

Optional, for the [caching layer](#caching): `redis` (shared tier), and `hnswlib` / `lz4` / `msgpack` / `psutil` (accelerators). None are required — without them the cache runs in-process with an exact NumPy semantic index.

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

# Cache statistics, clearing and warm-up (see the Caching section)
python manage.py milvus_cache_stats
python manage.py milvus_cache_clear --collection documents
python manage.py milvus_cache_warm --model myapp.models.Document --file vectors.json
```

## Caching

django-milvus ships a production-grade caching layer for vector reads. It is **off by default** — nothing changes until you add `MILVUS_CACHE` to your settings and opt a model in.

### What it does

Vector search is expensive: an ANN scan, a network round trip, and entity materialization on every call. Real workloads are also heavily skewed — a small set of queries repeats constantly beneath a long tail that arrives once and never returns. The cache turns those repeats into microsecond dictionary lookups.

It caches three things:

| What | Where | Example |
|---|---|---|
| **Search results** | keyed by the query vector | `.search(v, limit=5)` |
| **Query results** | keyed by filter, fields, limit, offset | `.filter(status="published")`, `.count()` |
| **Query embeddings** | a nearest-neighbour index over past query vectors | serves *near-duplicate* queries |

That last one is the distinctive part. Two users asking the same question in different words produce vectors that are 0.98 similar but never byte-identical, so an exact-match cache misses every one of them. The semantic cache catches them — see [Semantic caching](#semantic-closest-vector-caching).

### Where the cache sits

Below result parsing, above the client:

```
Document.objects.search(v, limit=5)
  └─ _fetch_all()
       └─ _execute_search()          ← the cache boundary
            ├─ exact key hit    → cached payload
            ├─ semantic hit     → neighbour's payload, reranked
            └─ miss             → client.search() → normalize → store
       └─ _parse_search_results(raw) → MilvusSearchResult objects
```

Cached payloads are the raw Milvus wire shape (plain `list[dict]`), never model instances. That keeps them serializer-agnostic, avoids pickle-version coupling in Redis, and means model rehydration is the same code path whether the data came from Milvus or from RAM.

### Topology

Two tiers, the second optional:

- **L1 — in-process RAM.** A hit costs a dict lookup and a policy update: no serialization, no socket, no copy. Bounded by bytes, not entry count, with real eviction algorithms. This is where the speed comes from.
- **L2 — shared (optional).** Redis or `django.core.cache`. One cache for every worker and host, so a query paid for by one process is nearly free for all the others, and an invalidation reaches the whole fleet.

On an L2 hit the entry is **promoted** into L1, so each worker pays the shared round trip at most once per entry.

### When *not* to cache

Caching is not free and not always right:

- **Read-after-write within a request.** Writes through the ORM invalidate correctly, but writes from another service or a raw pymilvus client do not — those rely on `TTL` expiry.
- **Strong-consistency reads.** `.consistency("Strong")` bypasses the cache by default; asking for Strong means asking for the freshest data Milvus has.
- **Write-heavy collections.** If a collection is written more often than it is read, every cached entry is invalidated before it earns its keep.
- **Unbounded scans.** A `limit=16384` dump will be refused by `MAX_ENTRY_BYTES` anyway. Do not raise that limit to force it in — one such entry evicts a large share of everything useful.
- **Queries that are never repeated.** Genuinely unique queries gain nothing and cost a little.

---

### Quick start

**1. Configure a cache.** The minimum, with no extra dependencies:

```python
# settings.py
MILVUS_CACHE = {
    'default': {
        'TTL': 300,
        'L1': {'MAX_MEMORY': '256MB'},
    }
}
```

**2. Opt a model in:**

```python
class Document(MilvusModel):
    id = fields.PrimaryKeyField(auto_id=True)
    title = fields.VarCharField(max_length=512)
    embedding = fields.FloatVectorField(dim=768)

    class MilvusMeta:
        collection_name = 'documents'
        cache = True          # or a dict of overrides
```

**3. Use it — no call-site changes:**

```python
Document.objects.search(vector, limit=5)     # first call: hits Milvus
Document.objects.search(vector, limit=5)     # second: served from RAM
```

**4. Verify:**

```python
>>> Document.objects.cache_stats()['hit_rate']
0.5
```

```bash
python manage.py milvus_cache_stats
```

**Adding a shared tier** (needs `pip install django-milvus[cache]`):

```python
MILVUS_CACHE = {
    'default': {
        'TTL': 300,
        'L1': {'MAX_MEMORY': '256MB'},
        'L2': {
            'BACKEND': 'django_milvus.cache.backends.redis.RedisBackend',
            'LOCATION': 'redis://localhost:6379/2',
        },
    }
}
```

---

### Settings reference

Every key has a default. `MILVUS_CACHE` need only list what you override, and settings are deep-merged, so `{'L1': {'MAX_MEMORY': '1GB'}}` changes only that one value and leaves the rest of `L1` at its defaults.

Settings are validated at start-up via a Django system check — a typo fails `manage.py check` with a readable message instead of raising from inside a view.

#### Root

| Key | Type | Default | What it does |
|---|---|---|---|
| `ENABLED` | bool | `True` | Master switch for this alias. `False` disables caching without deleting the configuration — useful for turning it off in one environment. |
| `TTL` | seconds \| `None` | `300` | How long an entry stays fresh. `None` means it never expires and leaves only capacity to evict it. **This is your backstop against writes the ORM cannot see**, so `None` is only safe if every write goes through django-milvus. |
| `TTL_JITTER` | 0–1 | `0.1` | Randomly spreads each TTL by ±this fraction. Without it, entries written together expire together, and a batch warm-up becomes a synchronised stampede one TTL later. `0.1` on a 300s TTL gives 270–330s. Set `0` for deterministic tests. |
| `NEGATIVE_TTL` | seconds \| `None` | `30` | Separate, shorter TTL for empty result sets. Empty results are worth caching — repeatedly asking for nothing still costs a round trip — but they are the most likely thing an insert invalidates, so they expire sooner. `None` uses `TTL`. |
| `STALE_WHILE_REVALIDATE` | seconds | `60` | Grace period past expiry during which a stale entry may still be served. Smooths the latency cliff at expiry. `0` disables. |
| `STALE_IF_ERROR` | bool | `True` | Serve an expired entry rather than raising when Milvus is unreachable. Trades freshness for availability. |
| `MAX_ENTRY_BYTES` | size \| `None` | `'1MB'` | Payloads larger than this are never admitted. Prevents a single huge query from evicting a large share of the cache to store one result. Rejections are counted in `cache_stats()['rejected']`. |
| `CACHE_COUNT` | bool | `True` | Whether `.count()` results are cached. Counts drift constantly on a busy collection; set `False` if you need them exact. |
| `CACHE_STRONG_CONSISTENCY` | bool | `False` | Whether querysets pinned to `.consistency("Strong")` may be cached. Defaults to `False` because asking for Strong is asking to bypass caches; set `True` only if you use Strong as a blanket default and do not mean it literally. |
| `FAIL_OPEN` | bool | `True` | Cache errors log a warning and fall through to Milvus rather than propagating. Leave this on. |

#### `L1` — the in-process tier

| Key | Type | Default | What it does |
|---|---|---|---|
| `BACKEND` | dotted path | `django_milvus.cache.backends.local.LocalRAMBackend` | Any class implementing `BaseCacheBackend`. See [Writing a custom backend](#writing-a-custom-backend). |
| `ALGORITHM` | str | `'w-tinylfu'` | Eviction policy. One of `lru`, `lfu`, `fifo`, `random`, `ttl`, `slru`, `2q`, `arc`, `w-tinylfu`. See [Eviction algorithms](#eviction-algorithms). |
| `MAX_MEMORY` | size \| `None` | `'256MB'` | Byte ceiling for this tier. Accepts `'256MB'`, `'1.5GiB'`, or an integer. |
| `MAX_ENTRIES` | int \| `None` | `100_000` | Entry ceiling, enforced alongside `MAX_MEMORY`. At least one of the two must be set — an unbounded cache is a memory leak, and the config check rejects it. |
| `SHARDS` | int | `8` | Lock-striping factor. The keyspace is split across N independently locked shards so concurrent requests rarely contend. Raise for high thread counts, lower for small caches. |
| `JANITOR` | bool | `True` | Run the background maintenance thread (expiry sweep, sketch aging, window tuning, memory-pressure checks). Disable in tests to keep things deterministic. |

#### `L1.WINDOW` — admission window control

| Key | Type | Default | What it does |
|---|---|---|---|
| `admission_ratio` | 0–0.8 | `0.01` | Fraction of capacity reserved for newly admitted keys before they must prove their worth. Larger favours bursty traffic where new keys become hot; smaller spends more capacity on proven entries. |
| `adaptive` | bool | `True` | Let the hill climber tune `admission_ratio` from the observed hit rate. Leave on unless you have measured a better fixed value. |
| `probation_ratio` | 0–1 | `0.2` | Split inside the main region between probation (unproven) and protected (hit at least twice). |
| `sample_interval` | seconds | `60` | The temporal window: how often hit rate is sampled and the janitor runs. |
| `step` | 0–1 | `0.05` | Hill-climb step size. Larger converges faster but overshoots more. |

#### `L1.WATERMARK` — batch eviction

| Key | Type | Default | What it does |
|---|---|---|---|
| `high` | 0–1 | `0.95` | Eviction starts once usage crosses this fraction of `MAX_MEMORY`. |
| `low` | 0–1 | `0.80` | ...and continues down to this fraction in one pass. Must be less than `high`. |

Evicting to exactly the limit would put the cache right back at the boundary on the very next insert, turning every write into an eviction. See [Memory management](#memory-management) for a worked example.

#### `L1.MEMORY_PRESSURE` — automatic capacity management

| Key | Type | Default | What it does |
|---|---|---|---|
| `enabled` | bool | `True` | Shrink the cache when the *process* comes under memory pressure. |
| `process_rss_limit` | size | `'2GB'` | Shrink once process RSS exceeds this. |
| `floor_ratio` | 0–1 | `0.25` | Never shrink effective capacity below this fraction of `MAX_MEMORY`. |

Requires `psutil` (`pip install django-milvus[fast]`). Without it this is silently inert, not an error.

#### `L2` — the shared tier

Omit the whole `L2` key for an L1-only cache; there is no code path cost for the tier you are not using.

| Key | Type | Default | What it does |
|---|---|---|---|
| `BACKEND` | dotted path | `django_milvus.cache.backends.redis.RedisBackend` | `RedisBackend`, `DjangoCacheBackend`, or your own. |
| `LOCATION` | str | `'redis://localhost:6379/0'` | Redis URL for `RedisBackend`; a `CACHES` **alias name** for `DjangoCacheBackend`. |
| `PREFIX` | str | `'dmv'` | Key prefix, so django-milvus keys are distinguishable from everything else sharing the store. |
| `SERIALIZER` | str | `'pickle'` | `pickle` (fastest, most permissive), `json` (portable and inspectable, larger, lossy on float precision), or `msgpack` (compact binary, needs the `msgpack` package). |
| `SOCKET_TIMEOUT` | seconds | `0.2` | Keep this **small**. It is added to every request that reaches the shared tier, and the circuit breaker needs failures to be fast to be useful. |

#### `L2.COMPRESS`

| Key | Type | Default | What it does |
|---|---|---|---|
| `algorithm` | str | `'none'` | `none`, `zlib` (available everywhere, slower) or `lz4` (much faster, needs the `lz4` package). |
| `min_bytes` | size | `2048` | Payloads below this skip compression: under a few KB the CPU costs more than the bytes saved on a local network. |
| `level` | int | `1` | Compression level for `zlib`. |

The compressor used is recorded in a framing byte on each payload, so changing this setting does not invalidate data already written.

#### `L2.CIRCUIT_BREAKER`

| Key | Type | Default | What it does |
|---|---|---|---|
| `failures` | int | `5` | Consecutive failures before the tier is skipped entirely. |
| `reset_after` | seconds | `30` | How long to skip it before letting one call probe again. |

Without a breaker, a Redis outage with a 200 ms timeout would add 200 ms to *every* request and turn a cache problem into an availability problem. With one, it costs one failed connection attempt every 30 seconds.

#### `SEMANTIC`

| Key | Type | Default | What it does |
|---|---|---|---|
| `enabled` | bool | `True` | Nearest-vector matching for searches. |
| `threshold` | float | `0.97` | Minimum similarity for a neighbouring query to answer. See the accuracy table in [Semantic caching](#semantic-closest-vector-caching). |
| `metric` | str | `'COSINE'` | `COSINE`, `IP` or `L2`. **Must match your collection's index metric.** |
| `max_vectors` | int | `20_000` | Cached query vectors per bucket. |
| `overfetch` | float | `3` | On a miss, fetch `limit * overfetch` rows so a later semantic hit has candidates to rerank. |
| `rerank` | bool | `True` | Re-score a neighbour's cached results against the caller's actual vector. **Leave this on** — it is what makes serving a neighbour's results safe. |
| `index` | str | `'auto'` | `auto`, `numpy` (exact brute force) or `hnswlib` (approximate). `auto` uses hnswlib only when installed *and* `max_vectors >= 100_000`. |

#### `STAMPEDE`

| Key | Type | Default | What it does |
|---|---|---|---|
| `enabled` | bool | `True` | Deduplicate concurrent misses on the same key. |
| `timeout` | seconds | `5` | How long a waiter blocks for the leader before querying Milvus itself. Stampede protection must never let one slow query stall every request behind it. |

#### `VERSIONING`

| Key | Type | Default | What it does |
|---|---|---|---|
| `enabled` | bool | `True` | Version-stamp cache keys so writes invalidate reads. Disabling leaves TTL as the only invalidation, which means stale reads after writes. |
| `shared` | bool | `True` | Mirror stamps in L2 so every worker agrees. No effect without an L2. |
| `refresh_interval` | seconds | `5` | How long a shared stamp is trusted locally before re-reading it. `0` re-reads on every query — correct but costs a round trip per query, defeating the local tier. |

#### `STATS`

| Key | Type | Default | What it does |
|---|---|---|---|
| `enabled` | bool | `True` | Record counters and latency. Cheap; leave on. |
| `window` | seconds | `300` | Sliding window for `recent_hit_rate`, which is what the adaptive window controller reads. A lifetime average would take hours to reflect a workload change. |

#### Complete annotated configuration

Every key at its default. Copy and edit down:

```python
MILVUS_CACHE = {
    'default': {
        'ENABLED': True,
        'TTL': 300,                        # seconds; None = never expires
        'TTL_JITTER': 0.1,                 # ±10% spread on every TTL
        'NEGATIVE_TTL': 30,                # shorter TTL for empty results
        'STALE_WHILE_REVALIDATE': 60,      # serve stale this long past expiry
        'STALE_IF_ERROR': True,            # serve stale when Milvus errors
        'MAX_ENTRY_BYTES': '1MB',          # refuse oversized payloads
        'CACHE_COUNT': True,               # cache .count()
        'CACHE_STRONG_CONSISTENCY': False, # .consistency("Strong") bypasses
        'FAIL_OPEN': True,                 # cache errors never break reads

        'L1': {
            'BACKEND': 'django_milvus.cache.backends.local.LocalRAMBackend',
            'ALGORITHM': 'w-tinylfu',      # see Eviction algorithms
            'MAX_MEMORY': '256MB',
            'MAX_ENTRIES': 100_000,
            'SHARDS': 8,                   # lock striping
            'JANITOR': True,               # background maintenance thread
            'WINDOW': {
                'admission_ratio': 0.01,   # capacity given to new keys
                'adaptive': True,          # hill-climb the ratio
                'probation_ratio': 0.2,    # unproven share of the main region
                'sample_interval': 60,     # temporal window, seconds
                'step': 0.05,              # hill-climb step size
            },
            'WATERMARK': {
                'high': 0.95,              # start evicting here
                'low': 0.80,               # ...and stop here
            },
            'MEMORY_PRESSURE': {
                'enabled': True,
                'process_rss_limit': '2GB',  # needs psutil
                'floor_ratio': 0.25,
            },
        },

        # Omit L2 entirely for an L1-only cache.
        'L2': None,
        # 'L2': {
        #     'BACKEND': 'django_milvus.cache.backends.redis.RedisBackend',
        #     'LOCATION': 'redis://localhost:6379/0',
        #     'PREFIX': 'dmv',
        #     'SERIALIZER': 'pickle',       # pickle | json | msgpack
        #     'SOCKET_TIMEOUT': 0.2,
        #     'COMPRESS': {
        #         'algorithm': 'none',      # none | zlib | lz4
        #         'min_bytes': 2048,
        #         'level': 1,
        #     },
        #     'CIRCUIT_BREAKER': {
        #         'failures': 5,
        #         'reset_after': 30,
        #     },
        #     'OPTIONS': {},                # passed to the backend constructor
        # },

        'SEMANTIC': {
            'enabled': True,
            'threshold': 0.97,             # minimum similarity for a hit
            'metric': 'COSINE',            # must match your index metric
            'max_vectors': 20_000,         # cached query vectors per bucket
            'overfetch': 3,                # fetch limit*3 to enable reranking
            'rerank': True,                # re-score against the real query
            'index': 'auto',               # auto | numpy | hnswlib
        },
        'STAMPEDE': {
            'enabled': True,
            'timeout': 5,
        },
        'VERSIONING': {
            'enabled': True,
            'shared': True,
            'refresh_interval': 5,
        },
        'STATS': {
            'enabled': True,
            'window': 300,
        },
    }
}
```

Multiple aliases are supported — add more top-level keys and select one with `.cache(alias='...')` or `MilvusMeta.cache = {'alias': '...'}`.

---

### Enabling caching

Three levels, each overriding the last.

**1. Global — `MILVUS_CACHE`.** Defines what caches exist and their defaults. Configuring it does *not* start caching anything.

**2. Per model — `MilvusMeta.cache`.** This is the opt-in:

```python
class Document(MilvusModel):
    class MilvusMeta:
        collection_name = 'documents'
        cache = True                       # use the alias defaults
```

```python
class Document(MilvusModel):
    class MilvusMeta:
        cache = {                          # ...or override them
            'ttl': 600,
            'alias': 'default',
            'semantic': {'threshold': 0.98},
        }
```

Omitting `cache`, or setting it to `None` or `False`, leaves the model uncached unless a queryset asks explicitly.

**3. Per query — `.cache()` / `.no_cache()`.** Wins over both:

```python
Document.objects.search(v, limit=5).cache(ttl=60)   # override the model's TTL
Document.objects.filter(x=1).cache()                # cache a model that opted out
Document.objects.search(v, limit=5).no_cache()      # bypass for one query
```

**Precedence rules:**

| Model `cache` | Queryset call | Result |
|---|---|---|
| unset / `None` / `False` | — | not cached |
| unset / `None` / `False` | `.cache()` | cached with alias defaults |
| `True` | — | cached with alias defaults |
| `True` or dict | `.no_cache()` | not cached |
| dict | `.cache(ttl=60)` | dict merged, `ttl` overridden |
| any | any | not cached if `ENABLED: False`, or `MILVUS_CACHE` is unset |

Resolution is exposed for inspection:

```python
>>> from django_milvus.cache import resolve_cache_options
>>> resolve_cache_options(Document, None)
{'ttl': 600, 'alias': 'default', 'semantic': {'threshold': 0.98}}
>>> resolve_cache_options(Document, {'enabled': False})
None
```

---

### QuerySet and Manager API

#### `.cache(...)`

```python
.cache(ttl=None, semantic=None, alias=None, store_vectors=None,
       keep_vectors=None, refresh=None, **extra)
```

Returns a new queryset (chainable, lazy — nothing runs until evaluation).

| Argument | Type | Meaning |
|---|---|---|
| `ttl` | seconds | Freshness for this query. Defaults to the model's, then the alias `TTL`. |
| `semantic` | bool \| float \| dict | `False` disables nearest-vector matching; `True` enables it with defaults; a float is shorthand for `threshold`; a dict overrides any of `threshold`, `metric`, `overfetch`, `rerank`. |
| `alias` | str | Which `MILVUS_CACHE` alias to use. |
| `store_vectors` | bool | Fetch the vector field alongside results so embeddings are available for reranking. |
| `keep_vectors` | bool | Leave embeddings in the returned entities instead of stripping them. |
| `refresh` | bool | Skip the lookup and repopulate from Milvus. |

```python
Document.objects.search(vector, limit=5).cache(ttl=60)
Document.objects.search(vector, limit=5).cache(semantic=0.99)
Document.objects.search(vector, limit=5).cache(
    semantic={'threshold': 0.99, 'rerank': False}
)
Document.objects.search(vector, limit=5).cache(store_vectors=True, ttl=3600)
Document.objects.filter(category='tech').limit(20).cache(alias='analytics')
```

#### `.no_cache()`

Bypasses the cache for this query, whatever the model says. Does not clear anything.

```python
fresh = Document.objects.filter(status='draft').no_cache()
```

#### `.refresh_cache()`

Queries Milvus and **overwrites** the cached entry — a forced miss that repopulates. Use after a write made outside the ORM, or to warm one entry ahead of traffic.

```python
Document.objects.search(vector, limit=5).refresh_cache()   # repopulates
list(Document.objects.search(vector, limit=5))             # now a hit
```

Contrast with `.no_cache()`, which leaves the cache untouched.

#### `.cache_key()`

Returns the key this query would use, or `None` if it will not be cached. The first thing to reach for when a cache appears not to be working: two queries you expect to share a result must produce the same key.

```python
>>> Document.objects.search(v, limit=5).cache().cache_key()
'dmv:default:documents:v3:s:44adb3ded3ed78018b166602'
>>> Document.objects.search(v, limit=5).no_cache().cache_key()
None
```

The key encodes `dmv:{alias}:{collection}:v{version}:{op}:{digest}`, where `op` is `q` (query), `s` (search), `h` (hybrid) or `c` (count).

#### `Model.objects.cache_stats()`

Full statistics for the cache this model uses, plus `collection` and `collection_version`. Returns `{}` when caching is not configured. See [Monitoring](#monitoring).

#### `Model.objects.cache_clear()`

Drops every cached entry for this model's collection and bumps its version. Returns the number of entries removed. Prefer it to clearing everything — other collections are unaffected.

```python
>>> Document.objects.cache_clear()
47
```

#### `Model.objects.cache_warm(...)`

```python
cache_warm(queries=None, vectors=None, limit=10, **cache_kwargs)
```

Populates the cache ahead of traffic. Returns a `WarmupResult` with `.warmed`, `.skipped` and `.errors`. Failures are collected, not raised — one bad query should not abort a deploy step.

```python
result = Document.objects.cache_warm(vectors=common_embeddings, limit=10, ttl=3600)
print(result.warmed, len(result.errors))

Document.objects.cache_warm(queries=[
    Document.objects.filter(status='published').limit(50),
    lambda: Document.objects.filter(featured=True).limit(20),
])
```

#### Module-level functions

```python
from django_milvus.cache import (
    caches, get_cache, invalidate, clear_all, cache_stats,
)

get_cache('default')            # -> MilvusCache, or None if unavailable
invalidate('documents')         # bump one collection's version -> new version
clear_all('default')            # empty one alias -> entries removed
clear_all()                     # empty every configured alias
cache_stats('default')          # one alias's statistics
cache_stats()                   # {alias: stats} for all of them
caches['default']               # the MilvusCache for an alias (builds lazily)
caches.reset()                  # discard every built cache (tests)
caches.prometheus()             # Prometheus text exposition
```

`get_cache()` returns `None` — never raises — when caching is unconfigured, disabled, or failed to build. Every call site treats all three the same way: query Milvus.

---

### Eviction algorithms

Set with `L1.ALGORITHM`. Each is a pure data structure over keys — it decides ordering only; the backend owns storage and byte accounting.

| Algorithm | Evicts | Overhead | Use it when |
|---|---|---|---|
| `lru` | Least recently used | Very low — one ordered dict | The working set is stable and fits. Weak against scans. |
| `lfu` | Least frequently used | Low — frequency buckets, O(1) hits | Popularity is genuinely stable. Slow to forget. |
| `fifo` | Oldest insertion | Lowest — hits cost nothing | Entries have uniform value and a natural lifetime. Good baseline. |
| `random` | An arbitrary key | Lowest — one set | Access is near-uniform. Immune to scan pollution by construction. |
| `ttl` | Nearest deadline | Low — one min-heap | Freshness dominates value; evicting a nearly-expired row costs little. |
| `slru` | Oldest *unproven* key | Low — two ordered dicts | You want scan resistance at almost LRU's cost. A key must be hit twice to be protected. |
| `2q` | Oldest admission-queue key | Moderate — plus a ghost list | A big scan runs alongside a steady hot set. Recognises keys whose reuse distance exceeds the admission queue. |
| `arc` | Whichever half is over its adaptive target | Higher — four ordered dicts, up to 2× capacity in ghost keys | The workload shifts between recency-driven and frequency-driven phases and you would rather not think about it. |
| **`w-tinylfu`** | The loser of a frequency duel | Moderate — a few bits per key | **The default.** Best hit rate on skewed traffic. |

#### Why `w-tinylfu` is the default

```
new key -> [ window LRU ]
                |
           (overflow: candidate)
                |
           frequency duel  --- candidate loses --> evict candidate
                |
           candidate wins --> [ probation | protected ]  (SLRU main)
                                    evict main victim
```

Every key first enters a small *window* sized by `admission_ratio`. When the window overflows, its oldest key becomes a *candidate* and is weighed against the main region's victim using estimated frequencies from a Count-Min Sketch. The more popular of the two survives.

That duel is the whole idea: what earns a key a place is how often it has been seen, not the accident of having just arrived. Plain LRU admits every newcomer unconditionally, so a stream of one-off queries walks straight through the cache evicting proven entries.

Measured on a Zipfian trace (5,000 distinct queries, 60,000 requests, 200-entry cache) — the package's own test suite asserts the first of these:

| Algorithm | Hit rate | Hot keys surviving a 3,000-key scan |
|---|---|---|
| `random` | 36.9% | — |
| `fifo` | 59.9% | — |
| `lru` | 64.9% | **0 / 50** |
| `2q` | 70.1% | 50 / 50 |
| `arc` | 71.0% | — |
| `slru` | 71.6% | — |
| `lfu` | 71.7% | — |
| **`w-tinylfu`** | **71.8%** | **49 / 50** |

The frequency sketch ages on a schedule (all counters halve periodically) so popularity can drift and yesterday's hot keys stop crowding out today's.

One consequence worth knowing: because admission is decided at the window boundary, a brand-new key needs roughly three sightings to displace an established one *once the cache is full*. Below capacity everything is admitted. That is correct behaviour — a query seen once gains nothing from being cached — but if your workload has no repetition at all, no policy will help.

#### Writing a custom policy

```python
from django_milvus.cache.policies.base import EvictionPolicy, register

@register
class MyPolicy(EvictionPolicy):
    name = "mine"                     # the ALGORITHM value

    def __init__(self, capacity=1024, **options):
        super().__init__(capacity, **options)
        self._keys = {}

    # --- required ---
    def on_admit(self, key, size=0, expires_at=None):
        """Record that `key` has just been stored."""
        self._keys[key] = 0

    def on_hit(self, key):
        """Record a successful lookup."""
        if key in self._keys:
            self._keys[key] += 1

    def on_remove(self, key):
        """Forget `key`; it is no longer stored."""
        self._keys.pop(key, None)

    def select_victim(self):
        """Return the key to evict next, or None if empty."""
        return min(self._keys, key=self._keys.get, default=None)

    def clear(self):
        self._keys.clear()

    def keys(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._keys

    # --- optional ---
    def should_admit(self, key):
        """Is this key worth the eviction it would cause? Default: True."""
        return True

    def on_reject(self, key):
        """Called when should_admit refused the key."""

    def age(self):
        """Periodic maintenance, called by the janitor thread."""

    def set_capacity(self, capacity):
        """Resize, re-proportioning any internal segments."""
        super().set_capacity(capacity)

    def stats(self):
        """Extra numbers for cache_stats()."""
        return {"algorithm": self.name, "tracked": len(self)}
```

Contract: every key handed to `on_admit` is eventually passed to `on_remove`, and `select_victim` must return a key the policy currently tracks. Capacity is in **entries**, not bytes; the backend derives an entry capacity from its byte budget and pushes it down via `set_capacity`.

Then:

```python
MILVUS_CACHE = {'default': {'L1': {'ALGORITHM': 'mine'}}}
```

Import the module once at start-up (for example in `AppConfig.ready`) so the registration runs.

```python
>>> from django_milvus.cache import available_policies
>>> available_policies()
['2q', 'arc', 'fifo', 'lfu', 'lru', 'mine', 'random', 'slru', 'ttl', 'w-tinylfu']
```

---

### Memory management

#### Why bytes, not entry counts

Vector payloads vary by orders of magnitude. A 5-row search over a 1536-dim collection with vectors stored is roughly 100× a 5-row scalar query. Bounding by entry count gives you no real control over footprint; bounding by bytes does.

#### How bytes are measured

`estimate_size()` walks the payload with fast arithmetic paths for the shapes actually stored — a 768-float embedding is computed, not walked element by element — and memoises shared objects so they are counted once. It is an estimate within a few percent, which is all that eviction decisions need.

```python
>>> from django_milvus.cache import estimate_size
>>> estimate_size([0.1] * 768)
24632
```

#### `MAX_MEMORY` vs `MAX_ENTRIES`

Both are enforced; whichever binds first wins. At least one must be set. `MAX_ENTRIES` is a hard ceiling evicted one-for-one; `MAX_MEMORY` uses watermarks.

#### Watermark eviction, worked

With `MAX_MEMORY: '256MB'`, `high: 0.95`, `low: 0.80` and 8 shards, each shard gets 32 MB:

1. Shard usage climbs past **30.4 MB** (0.95 × 32).
2. One eviction pass runs, down to **25.6 MB** (0.80 × 32) — reclaiming ~4.8 MB at once.
3. Nothing else evicts until usage climbs back past 30.4 MB.

Evicting to exactly 32 MB instead would put the shard back at the boundary on the very next insert, making every write trigger an eviction. The gap is what amortises the cost.

#### The janitor thread

One daemon thread per cache (not per shard), running every `sample_interval` seconds:

1. sweeps expired entries, so dead payloads do not occupy budget until they happen to be selected as victims;
2. ages frequency sketches;
3. ticks the adaptive window controller;
4. checks process RSS and adjusts effective capacity.

It never blocks shutdown, and a failed pass is logged rather than killing the thread. Set `JANITOR: False` to disable it — expiry then happens lazily on read, which is correct but leaves expired entries occupying capacity longer.

#### Automatic pressure handling

When `MEMORY_PRESSURE.enabled` and `psutil` is installed:

- process RSS exceeds `process_rss_limit` → effective `MAX_MEMORY` halves (bounded by `floor_ratio`), and a batch eviction brings usage down immediately;
- RSS falls below 85% of the limit → capacity recovers gradually (×1.5 per interval), not all at once, so it does not oscillate across the threshold.

```python
>>> Document.objects.cache_stats()['backend']['governor']
{'sweeps': 12, 'expired_swept': 340, 'pressure_events': 1,
 'shrink_factor': 0.5, 'effective_max_memory': 134217728, 'running': True}
```

#### Sizing a cache

Measure one payload, then multiply:

```python
from django_milvus.cache import estimate_size
from django_milvus.managers import normalize_search_results

raw = Document.objects.get_client().search(
    collection_name='documents', data=[vector],
    anns_field='embedding', limit=10,
    output_fields=Document.get_field_names(),
)
per_entry = estimate_size(normalize_search_results(raw))
print(per_entry)          # e.g. 48_000 bytes
```

`MAX_MEMORY ≈ distinct_queries_to_cache × per_entry ÷ 0.8` (the 0.8 accounts for the low watermark). For 5,000 distinct queries at 48 KB: 5000 × 48000 ÷ 0.8 ≈ **286 MB**.

Then check reality: if `evictions` climbs steadily while `hit_rate` stays low, the cache is too small for the working set. If `utilization` sits far below 1.0, it is larger than it needs to be.

#### Shard tuning

`SHARDS: 8` suits most deployments. Raise it if you run many threads per worker and see lock contention; lower it for small caches, where splitting a 16 MB budget across 8 shards leaves each with only 2 MB and can evict entries a single shard would have kept.

---

### Window size control

"Window" means two independently controllable things.

**Capacity window** — what share of the cache is reserved for freshly admitted keys before they must prove their worth:

```
  |<-- admission_ratio -->|<--------------- main region --------------->|
  [      window LRU       ][   probation   ][        protected         ]
                            |<- probation_ratio ->|
```

- Large window: favours bursty workloads where new keys are about to become hot.
- Small window: favours a stable hot set, spending nearly all capacity on proven entries.

**Temporal window** — `sample_interval`, the period over which hit rate is measured. Every decision below is made against the *recent* hit rate, never a lifetime average.

#### Manual tuning

```python
'WINDOW': {'adaptive': False, 'admission_ratio': 0.15, 'probation_ratio': 0.2}
```

#### Adaptive tuning (the default)

`WindowController` hill-climbs `admission_ratio`:

- hit rate **improved** → keep stepping in the same direction;
- hit rate **worsened** → reverse direction and halve the step, so the search converges instead of oscillating;
- step decayed to nothing → reset it, letting the controller escape a stale optimum after the workload shifts;
- **fewer than 50 lookups** in an interval → hold position rather than chase noise.

Bounded to `[0.0, 0.8]`; beyond that the policy degenerates toward plain LRU and loses its admission filter.

#### Confirming convergence

```python
>>> Document.objects.cache_stats()['backend']['window']
{'adaptive': True, 'admission_ratio': 0.35, 'probation_ratio': 0.2,
 'sample_interval': 60, 'step': 0.05, 'direction': 1,
 'ticks': 240, 'adjustments': 31, 'best_hit_rate': 0.82, 'best_ratio': 0.35}
```

Read it like this:

- `admission_ratio` ≈ `best_ratio` and `adjustments` growing slowly → converged.
- `adjustments` ≈ `ticks` → still hunting; the workload may be changing faster than `sample_interval`. Try a larger interval or a smaller `step`.
- `admission_ratio` pinned at `0.8` → the workload is dominated by new keys, and admission filtering is not buying anything. Consider `lru` or `2q`.
- `admission_ratio` near `0.0` → a very stable hot set. The default `w-tinylfu` is doing exactly what it should.

---

### Semantic (closest-vector) caching

Exact-key caching only helps when the identical embedding arrives twice. Two users phrasing the same question differently produce vectors 0.98 similar and never byte-identical — an exact cache misses every one.

#### Lookup order

```
1. exact key      byte-identical vector          -> hit
2. semantic       nearest cached vector >= t     -> hit, reranked
3. miss           ask Milvus, cache it, index the vector
```

#### Buckets

Comparing query vectors only makes sense between queries that are otherwise identical, so each combination of **collection + version + filter + limit + output fields + vector field + search params** owns its own index.

This is why two searches differing only in `limit` never share candidates: a cached `limit=5` result cannot answer a `limit=10` query. The same applies to different filters — a result set filtered to `category == "tech"` says nothing about the unfiltered top-k.

#### Thresholds by metric

For `COSINE` and `IP`, vectors are L2-normalized on insert, so similarity is a dot product in `[-1, 1]` and higher is closer:

| Threshold | Behaviour | Recall impact |
|---|---|---|
| `0.99`+ | Near-duplicates only | Essentially none |
| **`0.97`** | **Recommended default.** Paraphrases hit; results stay accurate | Minimal with reranking on |
| `0.95` | Aggressive. Noticeably higher hit rate | Visible drift on the result tail |
| `< 0.95` | Not recommended for user-facing search | Substantial |

For `L2`, `threshold` is an upper bound on **squared distance** — lower is closer, and it is unbounded rather than in `[0, 1]`. Its scale depends entirely on your embeddings; measure a few known-similar pairs before choosing one.

`metric` **must match your collection's index metric**. A COSINE threshold applied to L2 distances is meaningless.

#### Reranking — what makes this safe

A neighbour's cached results were ordered for *their* vector, not yours. Reranking fixes that:

1. On a **miss**, the cache fetches `limit * overfetch` rows (default 3×) and stores the wide list.
2. On a **semantic hit**, those candidates are re-scored against the **caller's actual vector**, re-sorted, and truncated to `limit`.

So the caller gets results ordered for their own query, not the neighbour's. Embeddings come from the entity payload when the vector field was requested, and otherwise from the vector cache. A candidate whose embedding cannot be found keeps its original score rather than being dropped — a partially-populated vector cache degrades the ordering rather than losing results.

#### The accuracy caveat, stated plainly

Even with reranking, the candidate *set* comes from the neighbour's search. If a document is a genuine top-5 match for your query but was not in the neighbour's top-15, no amount of reranking will surface it.

That is the real cost, and it shrinks as `threshold` rises and as `overfetch` grows. If you need exact top-k on every call, set `'SEMANTIC': {'enabled': False}` and keep exact-key caching, which is always exact.

#### `store_vectors`

```python
Document.objects.search(v, limit=5).cache(store_vectors=True)
```

Adds the vector field to the Milvus request so embeddings come back and can be filed for reranking. The embeddings are then **hoisted into the shared vector cache and stripped from the returned entities**, so this widens the Milvus response but *not* the cached payload — an embedding present in ten cached result sets is stored once, not ten times.

Pass `keep_vectors=True` if you actually want the embeddings in your results.

Without `store_vectors`, reranking still works for any candidate whose embedding is already in the vector cache from an earlier query.

#### Sizing `max_vectors`

This is cached query vectors per bucket. A 768-dim `float32` vector is 3,072 bytes, and the matrix is preallocated:

| `max_vectors` | dim 384 | dim 768 | dim 1536 |
|---|---|---|---|
| 1,000 | 1.5 MB | 3 MB | 6 MB |
| 20,000 (default) | 29 MB | 59 MB | 117 MB |
| 100,000 | 147 MB | 293 MB | 586 MB |

**Multiply by your bucket count.** A collection queried through five distinct filter/limit combinations has five buckets. If you use many filter combinations, lower `max_vectors`.

#### NumPy vs hnswlib

`index: 'auto'` (the default) uses the exact NumPy scan unless `hnswlib` is installed *and* `max_vectors >= 100_000`.

Brute force is the right default at cache scale. A probe is one matrix-vector product over a contiguous block:

| Cached vectors × dim | Probe time (single core) |
|---|---|
| 1,000 × 768 | ~0.5 ms |
| 10,000 × 768 | ~5–8 ms |
| 100,000 × 768 | ~60–80 ms |

It has zero index-build cost and — unlike HNSW — never misses a true neighbour. Past ~100k vectors the scan stops being free and `hnswlib` (`pip install django-milvus[fast]`) wins; a missed neighbour there costs a cache miss, not a wrong answer.

Requesting `'hnswlib'` without the package installed logs a warning and falls back to NumPy rather than failing.

#### Complete example

```python
class Document(MilvusModel):
    id = fields.PrimaryKeyField(auto_id=True)
    title = fields.VarCharField(max_length=512)
    embedding = fields.FloatVectorField(dim=768)

    class MilvusMeta:
        collection_name = 'documents'
        cache = {'ttl': 600, 'semantic': {'threshold': 0.98}}

    class MilvusIndexes:
        emb = indexes.HNSW(field='embedding', metric_type='COSINE')


# Cold: hits Milvus, fetches 5*3=15 rows, caches them, indexes the vector.
list(Document.objects.search(embed("how do I reset my password"), limit=5)
     .cache(store_vectors=True))

# A paraphrase: ~0.98 similar. Semantic hit, reranked, no Milvus call.
list(Document.objects.search(embed("password reset instructions"), limit=5)
     .cache(store_vectors=True))
```

```python
>>> Document.objects.cache_stats()['semantic']
{'enabled': True, 'threshold': 0.98, 'metric': 'COSINE', 'rerank': True,
 'overfetch': 3, 'buckets': 1, 'vectors': 1, 'bytes': 61440000,
 'hits': 1, 'misses': 1, 'reranks': 1, 'index_type': 'SemanticIndex'}
```

---

### The vector cache

A separate store mapping **primary key → embedding**, backing the rerank step.

- **What it holds:** one L2-normalized `float32` row per entity, in a preallocated contiguous matrix per `(collection, vector_field)`.
- **Where entries come from:** search results that included the vector field — either because you asked for it, or because `store_vectors=True` did.
- **Eviction:** LRU over matrix rows, capped by `SEMANTIC.max_vectors`. Rows are recycled rather than reallocated, so steady-state operation performs no allocation.
- **Cost:** `dim × 4` bytes per entity — 3,072 bytes at dim 768. Predictable and preallocated.
- **Invalidation:** dropped wholesale when its collection is invalidated.

Cosine similarity against a normalized row is a plain dot product, which is what makes reranking a candidate list a single matrix multiply.

```python
>>> Document.objects.cache_stats()['vectors']
{'caches': 1, 'entries': 4820, 'bytes': 61440000,
 'detail': {'documents.embedding': {'dim': 768, 'entries': 4820,
                                    'capacity': 20000, 'bytes': 61440000,
                                    'bytes_per_vector': 3072,
                                    'hits': 1204, 'misses': 33,
                                    'evictions': 0}}}
```

---

### Invalidation and consistency

#### Version stamping

Every collection carries a monotonic counter, baked into every cache key:

```
before:  dmv:default:documents:v7:s:44adb3ded3ed78018b166602
                              ^^
after a write:
         dmv:default:documents:v8:s:44adb3ded3ed78018b166602
                              ^^
```

Bump 7 → 8 and every existing key for that collection becomes unreachable in one integer increment. No key scanning, no reverse index, no `KEYS documents:*` against a live Redis. Orphaned entries are reclaimed later by TTL or normal eviction, which costs nothing extra because eviction was going to happen anyway.

The semantic index for that collection is dropped at the same moment — its vectors point at keys that no longer resolve.

**Why not track which entries contain which primary keys?** Because it cannot be made exact. A filter like `score > 0.8` matches rows that do not exist yet, so an insert can change its result without touching any key the cache has ever seen. Since precision is unattainable, the cheap correct option is: on a write, assume everything for that collection is suspect.

#### What bumps a version

Automatically, on every write through the ORM:

| Operation | Bumps |
|---|---|
| `Model.objects.create(...)` | ✅ (via `save()`) |
| `instance.save()` | ✅ |
| `instance.delete()` | ✅ |
| `Model.objects.bulk_create(...)` | ✅ |
| `Model.objects.upsert(...)` | ✅ |
| `Model.objects.delete(...)` | ✅ |
| `Model.objects.delete_by_ids(...)` | ✅ |
| `Model.objects.insert_raw(...)` | ✅ |
| `Model.objects.cache_clear()` | ✅ |
| `django_milvus.cache.invalidate(...)` | ✅ |
| **Direct `client.insert()` / `client.delete()`** | ❌ — see below |
| **Writes from another service** | ❌ — see below |

#### Out-of-band writes

Anything the ORM does not see cannot bump anything. Two options:

```python
# Invalidate explicitly after writing with a raw client
client = Document.objects.get_client()
client.insert(collection_name='documents', data=rows)
Document.invalidate_cache()
```

```python
# Or from anywhere, by collection name
from django_milvus.cache import invalidate
invalidate('documents')
```

Otherwise, `TTL` is the backstop — which is exactly why `TTL` should not be `None` unless every write goes through django-milvus.

#### Scope

A bump invalidates **one collection**, not the whole cache. Writing to `documents` leaves `articles` entirely alone.

#### Across workers

With an L2 the counter lives there (`INCR` is atomic on Redis), so a write in worker 1 invalidates worker 3's local cache too. Reads of the shared stamp are themselves cached for `refresh_interval` seconds — without that, checking the version would cost a round trip per query and undo the point of a local tier.

**Without an L2, a bump is local to the process that made the write.** Other workers keep serving their own cached copies until those expire. If read-after-write correctness across workers matters, configure an L2.

#### Stale-while-revalidate and stale-if-error

`STALE_WHILE_REVALIDATE: 60` lets an entry up to 60 seconds past expiry still be served, smoothing the latency cliff at expiry. `STALE_IF_ERROR: True` serves an expired entry rather than raising when Milvus is unreachable — trading freshness for availability. Both are counted in `cache_stats()['stale_hits']`.

#### The guarantees you actually get

- A cached read never reflects data older than the last ORM write to that collection **in this process** — or across the fleet with a shared L2.
- A cached read may be up to `TTL` (plus `STALE_WHILE_REVALIDATE`) behind a write the ORM did not see.
- `.consistency("Strong")` bypasses the cache entirely by default.
- A semantic hit returns results derived from a *similar* query's candidate set, reranked for yours. Set `SEMANTIC.enabled = False` if you need exact top-k on every call.

---

### Backends

#### `LocalRAMBackend` (L1)

```python
'L1': {
    'BACKEND': 'django_milvus.cache.backends.local.LocalRAMBackend',
    'ALGORITHM': 'w-tinylfu',
    'MAX_MEMORY': '256MB',
    'SHARDS': 8,
}
```

Live Python objects in this process's heap. Microsecond hits — no serialization, no socket, no copy. Byte-accounted, sharded, watermarked, with real eviction algorithms. No extra dependencies.

**Caveat, stated plainly:** this tier is **per process**. Four Gunicorn workers means four independent caches, each with its own copy of a hot entry, each warming separately. That is usually the right trade — RAM is cheap next to a Milvus round trip — but if you need one shared cache, configure an L2.

#### `RedisBackend` (L2)

```python
'L2': {
    'BACKEND': 'django_milvus.cache.backends.redis.RedisBackend',
    'LOCATION': 'redis://localhost:6379/2',
    'SERIALIZER': 'msgpack',
    'COMPRESS': {'algorithm': 'lz4', 'min_bytes': 2048},
    'SOCKET_TIMEOUT': 0.2,
}
```

Requires `pip install django-milvus[cache]`.

One cache shared by every worker and host. TTLs are handed to Redis via `PSETEX` so entries expire on schedule even if nobody asks for them again, and version bumps use atomic `INCR`, which is what makes cross-worker invalidation correct without a distributed lock. Bulk deletes use `SCAN`, never `KEYS` — this may run against a production instance, and `KEYS` blocks the whole server.

- **Characteristics:** shared, persistent, atomic counters, cross-process locking for stampede protection.
- **Caveats:** every access costs a round trip plus serialization, which is why it belongs *behind* L1. A payload that cannot be decoded (after a serializer change, say) is dropped and repopulated rather than failing forever.

#### `DjangoCacheBackend` (L2)

```python
'L2': {
    'BACKEND': 'django_milvus.cache.backends.djangocache.DjangoCacheBackend',
    'LOCATION': 'default',        # a CACHES alias, not a URL
}
```

Delegates to `django.core.cache`. Use it when you already run a cache Django knows about — django-redis, Memcached, database, LocMem — and would rather not configure a second connection.

- **Characteristics:** inherits that backend's eviction, persistence and clustering.
- **Caveats:** Django's cache API exposes no byte accounting, no eviction-algorithm choice and no key enumeration, so `MAX_MEMORY`, `ALGORITHM` and the window settings do not apply here. `delete_prefix` can only clear keys *this process* wrote — version stamping is what actually makes invalidation correct. LocMem is per-process and correctly reported as *not* shared, so version stamps are not mirrored into it.

#### `TieredCache`

Constructed automatically when you configure an `L2`. You never name it in settings.

```
get(key)
  -> L1 hit?  return                    (~microseconds)
  -> L2 hit?  promote into L1, return   (~1ms, once per worker per entry)
  -> miss                               (query Milvus)
```

Promotion preserves the entry's **original deadline** rather than restarting the clock, so a promoted entry cannot outlive its L2 twin. Writes go to both tiers; a payload L1 refuses as too large is still written to L2, where another worker may want it.

#### `DummyBackend`

```python
'L1': {'BACKEND': 'django_milvus.cache.backends.dummy.DummyBackend',
       'MAX_ENTRIES': 1}
```

Accepts everything, stores nothing, always misses. Structurally disables caching without deleting the configuration.

#### Writing a custom backend

```python
from django_milvus.cache.backends.base import MISSING, BaseCacheBackend

class MyBackend(BaseCacheBackend):
    name = "mine"
    shared = False      # True if other processes can see this store

    def __init__(self, config=None, stats=None, **options):
        super().__init__(config=config, stats=stats, **options)
        self._store = {}

    # --- required ---
    def get(self, key, default=MISSING):
        """Return the value, or `default` if absent or expired.

        MUST return `default` (not None) for a miss: None is a legitimate
        cached value, and conflating the two breaks negative caching.
        """
        return self._store.get(key, default)

    def set(self, key, value, ttl=None, size=None):
        """Store `value`. Return True if admitted.

        Returning False is not an error - a backend may legitimately
        refuse a payload as too large.
        """
        self._store[key] = value
        return True

    def delete(self, key):
        """Remove `key`. Return True if something was removed."""
        return self._store.pop(key, MISSING) is not MISSING

    def clear(self):
        """Remove everything. Return how many entries went."""
        count = len(self._store)
        self._store.clear()
        return count

    # --- optional ---
    def get_entry(self, key):
        """Return a CacheEntry including expired ones (enables
        stale-while-revalidate and deadline-preserving promotion).
        Return MISSING if unsupported."""
        return MISSING

    def get_many(self, keys): ...
    def set_many(self, items, ttl=None): ...
    def delete_many(self, keys): ...
    def delete_prefix(self, prefix): ...
    def incr_version(self, key, delta=1):
        """Atomically bump a counter. MUST be atomic on a shared backend:
        it underpins cross-worker invalidation."""
    def purge_expired(self): ...
    def touch(self, key, ttl=None): ...
    def close(self): ...
    def stats_dict(self): ...

    # For cross-process stampede protection on a shared backend:
    def acquire_lock(self, key, timeout=5): ...
    def release_lock(self, key): ...
```

Rules:

- **Thread safety is required.** Every method may be called concurrently.
- **Raise on genuine failures.** Do not swallow them — the tier above catches, records and falls through to Milvus, so a failure degrades performance but never correctness.
- **`MISSING`, not `None`, means absent.**
- You need not subclass `BaseCacheBackend`; matching the interface is enough.

```python
MILVUS_CACHE = {'default': {'L1': {'BACKEND': 'myapp.cache.MyBackend',
                                   'MAX_ENTRIES': 10_000}}}
```

---

### Production operations

#### Stampede protection

A popular entry expires, and the fifty requests already in flight all miss at once and all fire the same expensive search. The cache made the spike *worse* — without it those fifty would have been spread out.

Single-flight fixes it: the first caller to miss becomes the leader and queries Milvus; the rest wait and reuse its answer. One query instead of fifty.

Two levels, matching the two scopes: a `threading.Event` per key within a process, and a Redis `SET NX PX` lock across processes when a shared tier is configured. Non-leaders do not block on the lock — they poll the cache briefly and pick up the winner's answer.

Waiters always have an escape hatch: after `STAMPEDE.timeout` they query Milvus themselves. Stampede protection is an optimisation and must never let one slow query stall every request behind it. Errors from the leader propagate to all waiters, so a failure is visible rather than silently retried fifty times.

```python
>>> Document.objects.cache_stats()['stampede']
{'active': 0, 'leads': 143, 'joins': 892, 'timeouts': 0, 'timeout': 5,
 'shared_lock_acquired': 143, 'shared_lock_missed': 12}
```

`joins` far exceeding `leads` means it is doing real work. Rising `timeouts` means queries are slower than `STAMPEDE.timeout`.

#### Fail-open guarantee

Every cache operation is wrapped. A cache exception logs at `warning` and falls through to Milvus. **A caching layer must never be able to break reads** — and the test suite asserts this, including against a completely broken backend.

What this does *not* cover: errors from Milvus itself still propagate, as they must.

#### Circuit breaker

See [`L2.CIRCUIT_BREAKER`](#l2circuit_breaker). States are `closed` (healthy), `open` (skipping the tier), `half` (cool-off elapsed, letting one call probe).

#### Negative caching

Empty results are cached with the shorter `NEGATIVE_TTL`, so hammering a genuinely-empty filter does not hammer Milvus. Tracked as `negative_hits`.

#### Serializer and compression choice

| Situation | Choose |
|---|---|
| Default, single application | `pickle`, no compression |
| Large payloads, network-bound | `pickle` + `lz4` at `min_bytes: 2048` |
| Cache inspected by other tools | `json` (larger, lossy on float precision) |
| Bandwidth-constrained | `msgpack` + `lz4` |

Do not compress below ~2 KB: the CPU costs more than the bytes saved on a local network.

#### Multi-process implications

| Concern | L1 only | L1 + L2 |
|---|---|---|
| Cache warm-up | Per worker | Once, fleet-wide |
| Memory used | `MAX_MEMORY` × workers | `MAX_MEMORY` × workers + Redis |
| Invalidation reach | The writing process only | Every worker |
| `cache_clear()` reach | The current process only | Every worker |
| Hit latency | ~µs | ~µs (L1), ~1 ms (L2) |
| Survives restart | No | Yes |

**Size `MAX_MEMORY` per worker, not per host.** `'256MB'` across 8 workers is 2 GB of resident memory.

#### Warm-up on deploy

```python
# apps.py — warms each worker's own L1
class MyAppConfig(AppConfig):
    def ready(self):
        import os
        if os.environ.get('RUN_MAIN') or not settings.DEBUG:
            from myapp.models import Document
            Document.objects.cache_warm(vectors=COMMON_QUERIES, limit=10)
```

```bash
# With a shared L2, one command warms the whole fleet.
python manage.py milvus_cache_warm --model myapp.models.Document \
    --file common_vectors.json --limit 10 --ttl 3600
```

Without an L2, a management command warms a *separate* process and does not help your workers. Warm from inside each worker instead.

#### Tuning playbook

1. **Start with defaults** and `TTL` matched to how stale you can tolerate.
2. **Run a representative load.** Statistics only mean something under real traffic.
3. **Read `hit_rate`.**
   - Below 20% → is the workload actually repetitive? Check `cache_key()` for two queries you expect to share.
   - 20–50% → try enabling semantic caching, or raising `TTL`.
   - Above 50% → working well.
4. **Read `evictions` and `utilization`.** High evictions with low hit rate → raise `MAX_MEMORY`. Utilization far below 1.0 → lower it.
5. **Read `rejected`.** Non-zero means payloads exceed `MAX_ENTRY_BYTES`. Usually the right answer is a smaller `limit`, not a bigger ceiling.
6. **Check the window** converged (see [Window size control](#window-size-control)).
7. **Consider semantic caching** if queries are user-authored text. Start at `0.97` and lower cautiously while watching result quality.
8. **Add an L2** when you run multiple workers and warm-up cost or invalidation reach matters.

---

### Monitoring

#### `cache_stats()`

```python
>>> Document.objects.cache_stats()
```

**Effectiveness**

| Field | Meaning |
|---|---|
| `lookups` | Total cache lookups (`hits + misses`) |
| `hits` / `misses` | Served from cache / went to Milvus |
| `hit_rate` | Lifetime hit rate, 0–1 |
| `recent_hit_rate` | Hit rate over the `STATS.window` sliding window — what the adaptive controller reads |
| `l1_hits` | Answered by the in-process tier |
| `l2_hits` | Answered by the shared tier (each is also a promotion) |
| `semantic_hits` | Found by nearest-vector matching |
| `negative_hits` | Hits that returned an empty result set |
| `stale_hits` | Served past expiry under stale-while-revalidate |

`l1_hits + l2_hits == hits` — every hit is answered by exactly one storage tier. `semantic_hits`, `negative_hits` and `stale_hits` are **orthogonal**: they describe *how* an entry was found or *what* it contained, not where it was stored, so they overlap with the tier counters rather than adding to them.

**Writes and eviction**

| Field | Meaning |
|---|---|
| `sets` | Payloads written |
| `rejected` | Payloads refused as exceeding `MAX_ENTRY_BYTES` |
| `evictions` | Entries evicted to reclaim capacity |
| `expirations` | Entries removed after expiring |
| `invalidations` | Collections invalidated by writes |
| `bytes_written` / `bytes_evicted` | Byte totals |

**Health and latency**

| Field | Meaning |
|---|---|
| `errors` | Backend errors — all failed open; queries fell through to Milvus |
| `stampede_waits` | Callers that waited on an in-flight leader |
| `uptime` | Seconds since counters were reset |
| `latency_p50_ms` / `p95` / `p99` | Approximate lookup latency percentiles (bucket upper bounds) |

**Nested sections**

| Field | Contains |
|---|---|
| `backend` | Tier detail: `entries`, `bytes`, `utilization`, `algorithm`, `shards`, `policy`, `window`, `governor`, and `l1`/`l2` when tiered |
| `semantic` | Threshold, metric, buckets, vectors, hits, misses, reranks, index memory |
| `vectors` | Vector-cache entries and bytes, with per-field detail |
| `versions` | Current version stamp per collection |
| `stampede` | Leads, joins, timeouts, shared-lock counters |
| `collection`, `collection_version` | Added by `Model.objects.cache_stats()` |

#### Signals

```python
from django.dispatch import receiver
from django_milvus.cache.signals import (
    cache_hit, cache_miss, cache_set, cache_evicted, cache_invalidated,
)

@receiver(cache_hit)
def on_hit(sender, key, tier, collection, alias, **kwargs):
    statsd.increment(f'milvus.cache.hit.{tier}')
    if tier == 'semantic':
        statsd.histogram('milvus.cache.similarity', kwargs['similarity'])
```

| Signal | `sender` | kwargs |
|---|---|---|
| `cache_hit` | model class | `key`, `tier` (`fresh`/`stale`/`semantic`), `collection`, `alias`, `elapsed`, `similarity` (semantic only) |
| `cache_miss` | model class | `key`, `collection`, `alias`, `reason`, `elapsed` |
| `cache_set` | model class | `key`, `collection`, `alias`, `size`, `ttl`, `tier` |
| `cache_evicted` | `None` | `keys`, `collection`, `alias`, `reason` (`capacity`/`entries`/`expired`/`pressure`), `freed` |
| `cache_invalidated` | model class | `collection`, `alias`, `version`, `reason` (`write`/`manual`/`clear`) |

Receivers run **inline on the query path**, so keep them cheap; anything slow belongs on a queue. A receiver that raises is logged and swallowed — it cannot break a query.

#### Prometheus

```python
from django.http import HttpResponse
from django_milvus.cache import caches

def metrics(request):
    return HttpResponse(caches.prometheus(), content_type='text/plain')
```

```
# HELP milvus_cache_hits_total Total cache hits
# TYPE milvus_cache_hits_total counter
milvus_cache_hits_total{alias="default"} 8421
# HELP milvus_cache_hit_rate Lifetime hit rate
# TYPE milvus_cache_hit_rate gauge
milvus_cache_hit_rate{alias="default"} 0.8214
```

Counters (all suffixed `_total`): `lookups`, `hits`, `misses`, `l1_hits`, `l2_hits`, `semantic_hits`, `sets`, `rejected`, `evictions`, `expirations`, `invalidations`, `errors`, `bytes_written`, `bytes_evicted`.

Gauges: `hit_rate`, `recent_hit_rate`, `entries`, `bytes`, `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`.

```yaml
scrape_configs:
  - job_name: 'django-milvus-cache'
    metrics_path: '/metrics/milvus-cache/'
    static_configs:
      - targets: ['localhost:8000']
```

Note that with an L1-only cache each worker reports its own numbers, so scrape each instance rather than a load-balanced endpoint.

#### Management commands

```bash
# Formatted report for every configured alias
python manage.py milvus_cache_stats

# One alias
python manage.py milvus_cache_stats --alias analytics

# Raw JSON, for scripting
python manage.py milvus_cache_stats --json

# Prometheus text exposition
python manage.py milvus_cache_stats --prometheus
```

```
Cache alias: default
============================================================

  Effectiveness
    Lookups:          6
    Hit rate:         66.7%
    Recent hit rate:  66.7%
    Hits:             4 (local 4, shared 0, semantic 0)
    Misses:           2
    Milvus queries avoided: 4

  Latency
    p50:              0.05 ms
    p95:              1 ms
    p99:              1 ms

  Memory
    Entries:          2
    Bytes used:       4.3 KB
    Limit:            32.0 MB
    Utilization:      0.0%
    Evictions:        0
    Rejected (too large): 0

  Policy
    Algorithm:        w-tinylfu
    Admission ratio:  0.01 (adaptive)
    Window ticks:     0 (0 adjustments)
```

```bash
# Clear one collection (preferred - leaves others alone)
python manage.py milvus_cache_clear --collection documents

# Clear everything
python manage.py milvus_cache_clear --all

# Skip the confirmation prompt (CI, deploy scripts)
python manage.py milvus_cache_clear --all --yes

# Target a specific alias
python manage.py milvus_cache_clear --collection documents --alias analytics
```

| Flag | Meaning |
|---|---|
| `--collection <name>` | Clear one collection's entries |
| `--all` | Clear every collection |
| `--alias <name>` | Which cache alias (default: `default`) |
| `--yes` | Skip the confirmation prompt |

The command refuses to run without `--collection` or `--all` rather than guessing, and warns when no shared tier is configured — clearing then affects only the current process.

```bash
# Warm from a JSON file of query vectors
python manage.py milvus_cache_warm --model myapp.models.Document \
    --file common_vectors.json

python manage.py milvus_cache_warm --model myapp.models.Document \
    --file common_vectors.json --limit 20 --ttl 3600 \
    --vector-field embedding --alias default
```

| Flag | Meaning |
|---|---|
| `--model <dotted.path>` | The `MilvusModel` to warm (required) |
| `--file <path>` | JSON file of query vectors |
| `--limit <n>` | Result limit per warmed search (default 10) |
| `--ttl <seconds>` | TTL for warmed entries |
| `--vector-field <name>` | Vector field (auto-detected when unambiguous) |
| `--alias <name>` | Cache alias to warm |

The file accepts either layout:

```json
[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
```
```json
{"vectors": [[0.1, 0.2, 0.3]], "limit": 10, "ttl": 3600}
```

---

### Related caches

Two smaller caches are always on and need no configuration. Both address costs that predate this feature.

#### Collection load state

Milvus refuses reads against a collection that is not loaded. django-milvus handles that reactively — catch the error, create indexes, load, retry — but without any memory it repeated the whole sequence, including rebuilding index parameters, on every future miss.

`LoadStateCache` remembers, per `(alias, collection)`, whether the collection has been loaded and whether this process has already created its indexes. A cold collection now costs the error-index-load-retry sequence **once** instead of on every miss.

Entries carry a 300-second TTL and are cleared immediately on any Milvus error, so a collection released out from under the process recovers on the next query. `Model.objects.release_collection()` clears it explicitly. This is deliberately optimistic: being wrong costs one retried query, being right saves a round trip and an index rebuild.

```python
>>> from django_milvus.cache.loadstate import load_state
>>> load_state.stats_dict()
{'ttl': 300.0, 'loaded': 2, 'indexed': 2, 'hits': 1840, 'misses': 2,
 'invalidations': 0}
>>> load_state.invalidate('documents')     # force a reload on the next query
```

#### Connection liveness probe

`MilvusConnectionManager.get_connection()` validated its cached client with `client.list_collections()` on **every call** — a full network round trip added to every query in the application, more than a cache hit costs in total.

That probe is now throttled to once per 30 seconds per connection (`MilvusConnectionManager.PROBE_INTERVAL`). A dropped connection is still detected and replaced; one that dies inside the window surfaces as a normal query error and is discarded then.

```python
from django_milvus.connection import connections
connections.PROBE_INTERVAL = 60      # or 0 to probe on every call
```

---

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| **Cache appears to do nothing** | `MILVUS_CACHE` unset, or the model never opted in | Check `Model.objects.cache_stats()` — `{}` means no cache is configured. Then check `MilvusMeta.cache` is set, or use `.cache()` explicitly. |
| | Every query is genuinely distinct | Compare `cache_key()` for two queries you expect to share. Different keys mean different filters, limits, output fields, partitions or consistency levels. |
| | `ENABLED: False` | Check the alias config. |
| **Low hit rate** | Cache too small for the working set | `evictions` climbing with low `hit_rate` → raise `MAX_MEMORY`. |
| | `TTL` too short | Raise it, or check `expirations` against `evictions` to see which is dominating. |
| | Writes invalidating constantly | Check `invalidations`. A write-heavy collection may not be worth caching. |
| | Near-duplicate rather than identical queries | Enable semantic caching. |
| **Memory keeps growing** | `MAX_MEMORY` sized per host, not per worker | It is per process. 8 workers × 256 MB = 2 GB. |
| | Semantic index and vector cache are additional | They are *not* counted against `MAX_MEMORY`. Budget `max_vectors × dim × 4 × buckets` separately. |
| | Payloads larger than expected | Check `avg_entry_bytes` in `stats_dict()['backend']`. |
| **Stale results** | Writes made outside the ORM | Call `Model.invalidate_cache()` or `invalidate('collection')` after them. |
| | Multiple workers with no shared tier | A bump is local without an L2. Configure one, or shorten `TTL`. |
| | `VERSIONING.enabled = False` | Re-enable it. |
| | Stale-while-revalidate serving expired entries | Check `stale_hits`; lower `STALE_WHILE_REVALIDATE`. |
| **Redis errors in the logs** | Shared tier unreachable | Reads still work — this is fail-open. Check `stats_dict()['backend']['circuit_breaker']['state']`; `open` means it has stopped trying. |
| | `SOCKET_TIMEOUT` too tight | Raise it a little, but keep it well under your request budget. |
| | Serializer changed | Undecodable entries are dropped and repopulated automatically. |
| **Semantic false positives** (wrong-looking results) | `threshold` too low | Raise it. 0.97 → 0.99 is the usual first move. |
| | `rerank` disabled | Re-enable it; it is what makes neighbour hits safe. |
| | `metric` does not match the index | A COSINE threshold on L2 distances is meaningless. |
| | Candidate set too narrow | Raise `overfetch`, and use `store_vectors=True` so embeddings are available. |
| **Semantic cache never hits** | `threshold` too strict for your embeddings | Log `similarity` from the `cache_hit` signal, or lower the threshold and observe. |
| | Queries land in different buckets | Different `limit` or filters never share candidates. |
| | Batch searches | Semantic matching applies to single-vector searches only. |
| **`rejected` climbing** | Payloads exceed `MAX_ENTRY_BYTES` | Usually the right fix is a smaller `limit`, not a bigger ceiling. |
| **Slow first request per worker** | Cold L1 in every process | Warm on start-up, or add an L2 so only the first worker pays. |
| **`manage.py check` fails on `django_milvus.E001`** | Invalid `MILVUS_CACHE` | The message names the offending key. |

Enable debug logging to watch decisions:

```python
LOGGING = {
    'version': 1,
    'loggers': {
        'django_milvus.cache': {'handlers': ['console'], 'level': 'DEBUG'},
    },
}
```

---

### Migration and compatibility

**Upgrading changes nothing.** Caching is off by default. Without `MILVUS_CACHE` in settings, every read goes to Milvus exactly as before — the package's test suite asserts this by counting client calls.

**Nothing in the existing API changed.** `search()`, `filter()`, `count()` and the rest keep their signatures and return types. `.cache()`, `.no_cache()`, `.refresh_cache()` and `.cache_key()` are additive.

**Safe rollout:**

1. Add `MILVUS_CACHE` with a conservative `TTL` (60–300s). Nothing is cached yet.
2. Enable one model with `MilvusMeta.cache = True`, or one query with `.cache()`.
3. Watch `milvus_cache_stats` under real traffic. Confirm hit rate and result quality.
4. Widen to more models. Add `SEMANTIC` last — it is the setting most likely to need tuning.
5. Add an L2 when multi-worker warm-up or invalidation reach becomes the limiting factor.

**Instant rollback,** in decreasing order of scope:

```python
MILVUS_CACHE = None                       # remove caching entirely
MILVUS_CACHE = {'default': {'ENABLED': False, ...}}   # disable one alias
# or drop `cache` from a single model's MilvusMeta
```

```python
Document.objects.search(v, limit=5).no_cache()   # bypass one query
```

No data migration, no schema change, no collection change. The cache is entirely derived state.

**Version note:** the caching layer was added in **0.2.0**. `numpy` became a direct dependency (it was already present transitively via pymilvus). `redis`, `hnswlib`, `lz4`, `msgpack` and `psutil` are all optional; without them the cache runs L1-only with an exact NumPy semantic index, pickle serialization, no compression and no memory-pressure handling.


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

## License

MIT License. See [LICENSE](LICENSE) for details.
