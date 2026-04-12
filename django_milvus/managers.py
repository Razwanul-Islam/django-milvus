"""
MilvusManager and MilvusQuerySet: Django ORM-like interface for Milvus.

Provides chainable, lazy-evaluated query interface for Milvus collections
including vector similarity search, filtering, and CRUD operations.
"""

import copy
from pymilvus.exceptions import MilvusException
from .connection import get_milvus_client
from .utils import build_filter_expr
from .schema import _create_indexes_for_model
from .exceptions import (
    ObjectDoesNotExist, MultipleObjectsReturned,
    SearchError, ValidationError,
)


class MilvusQuerySet:
    """
    Lazy, chainable query interface for Milvus collections.

    Supports Django-style filtering, limiting, and iteration.
    Results are not fetched until the queryset is evaluated
    (iteration, slicing, len, bool, list).
    """

    def __init__(self, model):
        self.model = model
        self._filters = {}
        self._filter_expr = ""
        self._output_fields = None
        self._limit = None
        self._offset = 0
        self._order_by = None
        self._partition_names = None
        self._consistency_level = None
        self._result_cache = None

        # Search-specific
        self._search_params = None
        self._is_search = False
        self._is_hybrid = False

    def _clone(self):
        """Return a deep copy of this queryset."""
        qs = self.__class__(self.model)
        qs._filters = copy.deepcopy(self._filters)
        qs._filter_expr = self._filter_expr
        qs._output_fields = (
            list(self._output_fields) if self._output_fields else None
        )
        qs._limit = self._limit
        qs._offset = self._offset
        qs._order_by = self._order_by
        qs._partition_names = self._partition_names
        qs._consistency_level = self._consistency_level
        qs._search_params = (
            copy.deepcopy(self._search_params) if self._search_params else None
        )
        qs._is_search = self._is_search
        qs._is_hybrid = self._is_hybrid
        return qs

    def _invalidate_cache(self):
        self._result_cache = None

    # ─────────────────────────────────────────────────────
    # Chainable methods
    # ─────────────────────────────────────────────────────

    def filter(self, expr=None, **kwargs):
        """Filter results using Django-style lookups or raw expressions.

        Examples:
            .filter(category="test")
            .filter(score__gt=0.5)
            .filter(title__like="hello%")
            .filter(expr='category == "test" and score > 0.5')
        """
        qs = self._clone()
        if expr:
            if qs._filter_expr:
                qs._filter_expr = f"({qs._filter_expr}) and ({expr})"
            else:
                qs._filter_expr = expr
        if kwargs:
            qs._filters.update(kwargs)
        return qs

    def exclude(self, **kwargs):
        """Exclude results matching the given conditions."""
        qs = self._clone()
        # Build negated expressions
        for key, value in kwargs.items():
            negated_key = key
            parts = key.split("__")
            if len(parts) == 1:
                negated_key = f"{key}__ne"
            elif parts[-1] == "eq":
                negated_key = "__".join(parts[:-1]) + "__ne"
            elif parts[-1] == "in":
                negated_key = "__".join(parts[:-1]) + "__nin"
            else:
                # For other operators, use raw expression negation
                single_filter = build_filter_expr({key: value})
                if qs._filter_expr:
                    qs._filter_expr = f"({qs._filter_expr}) and not ({single_filter})"
                else:
                    qs._filter_expr = f"not ({single_filter})"
                continue
            qs._filters[negated_key] = value
        return qs

    def limit(self, count):
        """Limit the number of results."""
        qs = self._clone()
        qs._limit = count
        return qs

    def offset(self, count):
        """Skip the first N results."""
        qs = self._clone()
        qs._offset = count
        return qs

    def only(self, *fields):
        """Only return specific fields."""
        qs = self._clone()
        qs._output_fields = list(fields)
        return qs

    def values(self, *fields):
        """Return results as dicts with only specified fields."""
        qs = self._clone()
        if fields:
            qs._output_fields = list(fields)
        return qs

    def using_partitions(self, *partition_names):
        """Restrict query to specific partitions."""
        qs = self._clone()
        qs._partition_names = list(partition_names)
        return qs

    def consistency(self, level):
        """Set consistency level for this query.

        Args:
            level: "Strong", "Bounded", "Session", "Eventually"
        """
        qs = self._clone()
        qs._consistency_level = level
        return qs

    def all(self):
        """Return a copy of this queryset."""
        return self._clone()

    def none(self):
        """Return an empty queryset."""
        qs = self._clone()
        qs._result_cache = []
        return qs

    # ─────────────────────────────────────────────────────
    # Vector Search Methods
    # ─────────────────────────────────────────────────────

    def search(self, vector, vector_field=None, limit=10,
               metric_type=None, search_params=None,
               output_fields=None, **kwargs):
        """Perform similarity search on vector field.

        Args:
            vector: Query vector (list of floats) or list of vectors.
            vector_field: Name of the vector field to search. Auto-detected
                          if the model has only one vector field.
            limit: Maximum number of results per query vector.
            metric_type: Distance metric ("COSINE", "L2", "IP").
            search_params: Additional search parameters dict.
            output_fields: Fields to include in results.
            **kwargs: Extra search parameters (nprobe, ef, etc.).

        Returns:
            MilvusSearchResults
        """
        qs = self._clone()

        # Auto-detect vector field if not specified
        if vector_field is None:
            vector_fields = self.model.get_vector_fields()
            if len(vector_fields) == 1:
                vector_field = list(vector_fields.keys())[0]
            elif len(vector_fields) == 0:
                raise SearchError(
                    f"Model {self.model.__name__} has no vector fields"
                )
            else:
                raise SearchError(
                    f"Model {self.model.__name__} has multiple vector fields: "
                    f"{list(vector_fields.keys())}. Specify vector_field."
                )

        params = search_params or {}
        if kwargs:
            params.update(kwargs)

        qs._is_search = True
        qs._search_params = {
            "data": vector if isinstance(vector[0], (list, tuple)) else [vector],
            "anns_field": vector_field,
            "limit": limit,
            "search_params": params,
            "output_fields": output_fields or qs._output_fields,
        }
        if metric_type:
            qs._search_params["search_params"]["metric_type"] = metric_type

        return qs

    def hybrid_search(self, requests, ranker, limit=10,
                      output_fields=None, **kwargs):
        """Perform hybrid search combining multiple vector searches.

        Args:
            requests: List of AnnSearchRequest objects or dicts with:
                - data: query vectors
                - anns_field: vector field name
                - param: search params dict
                - limit: limit per request
            ranker: Ranker object or dict for result fusion.
            limit: Final limit for fused results.
            output_fields: Fields to include in results.

        Returns:
            MilvusSearchResults
        """
        qs = self._clone()
        qs._is_hybrid = True
        qs._search_params = {
            "requests": requests,
            "ranker": ranker,
            "limit": limit,
            "output_fields": output_fields or qs._output_fields,
        }
        return qs

    # ─────────────────────────────────────────────────────
    # Evaluation / Execution
    # ─────────────────────────────────────────────────────

    def _build_filter(self):
        """Build final filter expression."""
        parts = []
        if self._filters:
            expr = build_filter_expr(self._filters)
            if expr:
                parts.append(expr)
        if self._filter_expr:
            parts.append(self._filter_expr)
        return " and ".join(f"({p})" for p in parts) if parts else ""

    def _get_output_fields(self):
        """Get output fields list."""
        if self._output_fields:
            return self._output_fields
        return self.model.get_field_names()

    def _ensure_loaded(self, client, collection):
        """Ensure collection indexes exist and collection is loaded."""
        try:
            _create_indexes_for_model(client, self.model)
        except Exception as e:
            # Ignore "already exists" errors, re-raise others
            if "already exists" not in str(e).lower() and "Duplicate" not in str(e):
                raise
        client.load_collection(collection)

    def _execute_query(self):
        """Execute a query (non-search) against Milvus."""
        client = self.model.get_client()
        collection = self.model.get_collection_name()
        filter_expr = self._build_filter()

        kwargs = {
            "collection_name": collection,
            "output_fields": self._get_output_fields(),
        }

        if filter_expr:
            kwargs["filter"] = filter_expr
        else:
            # Milvus requires a filter for query; use a match-all expression
            kwargs["filter"] = ""

        if self._limit is not None:
            kwargs["limit"] = self._limit
        else:
            kwargs["limit"] = 16384  # Milvus default max

        if self._offset:
            kwargs["offset"] = self._offset

        if self._partition_names:
            kwargs["partition_names"] = self._partition_names

        if self._consistency_level:
            kwargs["consistency_level"] = self._consistency_level

        try:
            results = client.query(**kwargs)
        except MilvusException as e:
            if "not loaded" in str(e):
                self._ensure_loaded(client, collection)
                results = client.query(**kwargs)
            else:
                raise
        return results

    def _execute_search(self):
        """Execute a vector search against Milvus."""
        client = self.model.get_client()
        collection = self.model.get_collection_name()
        filter_expr = self._build_filter()

        params = self._search_params.copy()
        params["collection_name"] = collection

        if filter_expr:
            params["filter"] = filter_expr

        if params.get("output_fields") is None:
            params["output_fields"] = self._get_output_fields()

        if self._partition_names:
            params["partition_names"] = self._partition_names

        if self._consistency_level:
            params["consistency_level"] = self._consistency_level

        try:
            results = client.search(**params)
        except MilvusException as e:
            if "not loaded" in str(e):
                self._ensure_loaded(client, collection)
                results = client.search(**params)
            else:
                raise
        return results

    def _execute_hybrid_search(self):
        """Execute a hybrid vector search against Milvus."""
        client = self.model.get_client()
        collection = self.model.get_collection_name()

        params = self._search_params.copy()
        params["collection_name"] = collection

        if params.get("output_fields") is None:
            params["output_fields"] = self._get_output_fields()

        if self._partition_names:
            params["partition_names"] = self._partition_names

        collection = self.model.get_collection_name()
        try:
            results = client.hybrid_search(**params)
        except MilvusException as e:
            if "not loaded" in str(e):
                self._ensure_loaded(client, collection)
                results = client.hybrid_search(**params)
            else:
                raise
        return results

    def _fetch_all(self):
        """Fetch results and populate cache."""
        if self._result_cache is not None:
            return

        if self._is_hybrid:
            raw = self._execute_hybrid_search()
            self._result_cache = self._parse_search_results(raw)
        elif self._is_search:
            raw = self._execute_search()
            self._result_cache = self._parse_search_results(raw)
        else:
            raw = self._execute_query()
            self._result_cache = [
                self.model.from_dict(item) for item in raw
            ]

    def _parse_search_results(self, raw_results):
        """Parse search results into MilvusSearchResult objects."""
        results = []
        for query_result in raw_results:
            for hit in query_result:
                entity_data = hit.get("entity", hit)
                if isinstance(entity_data, dict):
                    instance = self.model.from_dict(entity_data)
                else:
                    instance = entity_data

                distance = hit.get("distance", None)
                pk = hit.get("id", None)

                result = MilvusSearchResult(
                    entity=instance,
                    distance=distance,
                    score=distance,
                    id=pk,
                )
                results.append(result)
        return results

    # ─────────────────────────────────────────────────────
    # Terminal methods (trigger evaluation)
    # ─────────────────────────────────────────────────────

    def get(self, **kwargs):
        """Get a single object matching the filters.

        Raises:
            ObjectDoesNotExist: If no object matches.
            MultipleObjectsReturned: If more than one object matches.
        """
        qs = self.filter(**kwargs).limit(2)
        qs._fetch_all()

        if len(qs._result_cache) == 0:
            raise ObjectDoesNotExist(
                f"{self.model.__name__} matching query does not exist."
            )
        if len(qs._result_cache) > 1:
            raise MultipleObjectsReturned(
                f"get() returned more than one {self.model.__name__}."
            )
        return qs._result_cache[0]

    def get_or_none(self, **kwargs):
        """Get a single object or None if not found."""
        try:
            return self.get(**kwargs)
        except ObjectDoesNotExist:
            return None

    def first(self):
        """Return the first result, or None."""
        qs = self.limit(1)
        qs._fetch_all()
        return qs._result_cache[0] if qs._result_cache else None

    def count(self):
        """Return the count of matching entities."""
        client = self.model.get_client()
        collection = self.model.get_collection_name()
        filter_expr = self._build_filter()

        # Use query with count(*)
        kwargs = {"collection_name": collection}
        if filter_expr:
            kwargs["filter"] = filter_expr
        else:
            kwargs["filter"] = ""

        kwargs["output_fields"] = ["count(*)"]

        results = client.query(**kwargs)
        if results and "count(*)" in results[0]:
            return results[0]["count(*)"]
        return len(results)

    def exists(self):
        """Check if any results exist."""
        return self.limit(1).count() > 0

    def create(self, **kwargs):
        """Create and save a new instance."""
        instance = self.model(**kwargs)
        instance.save()
        return instance

    def bulk_create(self, instances=None, data=None, batch_size=1000):
        """Bulk insert multiple instances or raw data dicts.

        Args:
            instances: List of model instances.
            data: List of dicts (alternative to instances).
            batch_size: Number of records per batch.

        Returns:
            Insert result from Milvus.
        """
        client = self.model.get_client()
        collection = self.model.get_collection_name()

        if instances:
            all_data = [inst.to_dict() for inst in instances]
        elif data:
            all_data = data
        else:
            raise ValidationError("Provide either instances or data")

        results = []
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i + batch_size]
            result = client.insert(
                collection_name=collection,
                data=batch,
            )
            results.append(result)

            # Set PKs on instances
            if instances:
                pk_field = self.model._pk_field
                if pk_field and pk_field.auto_id:
                    ids = []
                    if hasattr(result, "primary_keys"):
                        ids = result.primary_keys
                    elif isinstance(result, dict) and "ids" in result:
                        ids = result["ids"]

                    for idx, inst in enumerate(instances[i:i + batch_size]):
                        if idx < len(ids):
                            inst.pk = ids[idx]
                        inst._is_new = False

        return results

    def upsert(self, instances=None, data=None, batch_size=1000):
        """Upsert (insert or update) multiple instances or raw data dicts.

        Args:
            instances: List of model instances.
            data: List of dicts (alternative to instances).
            batch_size: Number of records per batch.

        Returns:
            Upsert result from Milvus.
        """
        client = self.model.get_client()
        collection = self.model.get_collection_name()

        if instances:
            all_data = [inst.to_dict() for inst in instances]
        elif data:
            all_data = data
        else:
            raise ValidationError("Provide either instances or data")

        results = []
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i + batch_size]
            result = client.upsert(
                collection_name=collection,
                data=batch,
            )
            results.append(result)
        return results

    def delete(self, **kwargs):
        """Delete entities matching the current filters.

        Args:
            **kwargs: Additional filters.
        """
        if kwargs:
            qs = self.filter(**kwargs)
        else:
            qs = self

        client = self.model.get_client()
        collection = self.model.get_collection_name()
        filter_expr = qs._build_filter()

        if not filter_expr:
            raise ValidationError(
                "Cannot delete without filters. Use drop_collection() to "
                "remove all data."
            )

        extra = {}
        if qs._partition_names:
            extra["partition_name"] = qs._partition_names[0]

        return client.delete(
            collection_name=collection,
            filter=filter_expr,
            **extra,
        )

    def delete_by_ids(self, ids):
        """Delete entities by their primary key IDs.

        Args:
            ids: List of primary key values.
        """
        client = self.model.get_client()
        collection = self.model.get_collection_name()
        return client.delete(
            collection_name=collection,
            ids=ids,
        )

    # ─────────────────────────────────────────────────────
    # Iteration / Evaluation
    # ─────────────────────────────────────────────────────

    def __iter__(self):
        self._fetch_all()
        return iter(self._result_cache)

    def __len__(self):
        self._fetch_all()
        return len(self._result_cache)

    def __bool__(self):
        self._fetch_all()
        return bool(self._result_cache)

    def __getitem__(self, index):
        self._fetch_all()
        return self._result_cache[index]

    def __repr__(self):
        self._fetch_all()
        return f"<MilvusQuerySet [{self._result_cache[:5]}{'...' if len(self._result_cache) > 5 else ''}]>"


class MilvusSearchResult:
    """Wraps a single search result with distance/score info."""

    def __init__(self, entity, distance=None, score=None, id=None):
        self.entity = entity
        self.distance = distance
        self.score = score
        self.id = id

    def __repr__(self):
        return (
            f"<MilvusSearchResult(id={self.id}, "
            f"distance={self.distance})>"
        )

    def __getattr__(self, name):
        """Proxy attribute access to the entity."""
        if name in ("entity", "distance", "score", "id"):
            raise AttributeError(name)
        return getattr(self.entity, name)


class MilvusManager:
    """
    Manager class providing access to Milvus operations.

    Attached to model classes as `Model.objects`.
    Provides both queryset methods and direct Milvus operations.
    """

    def __init__(self):
        self.model = None

    def __get__(self, obj, objtype=None):
        if obj is not None:
            raise AttributeError(
                "Manager isn't accessible via model instances"
            )
        return self

    def _get_queryset(self):
        return MilvusQuerySet(self.model)

    # ─────────────────────────────────────────────────────
    # QuerySet proxy methods
    # ─────────────────────────────────────────────────────

    def all(self):
        return self._get_queryset().all()

    def filter(self, expr=None, **kwargs):
        return self._get_queryset().filter(expr=expr, **kwargs)

    def exclude(self, **kwargs):
        return self._get_queryset().exclude(**kwargs)

    def get(self, **kwargs):
        return self._get_queryset().get(**kwargs)

    def get_or_none(self, **kwargs):
        return self._get_queryset().get_or_none(**kwargs)

    def first(self):
        return self._get_queryset().first()

    def limit(self, count):
        return self._get_queryset().limit(count)

    def offset(self, count):
        return self._get_queryset().offset(count)

    def only(self, *fields):
        return self._get_queryset().only(*fields)

    def values(self, *fields):
        return self._get_queryset().values(*fields)

    def count(self):
        return self._get_queryset().count()

    def exists(self):
        return self._get_queryset().exists()

    def create(self, **kwargs):
        return self._get_queryset().create(**kwargs)

    def bulk_create(self, instances=None, data=None, batch_size=1000):
        return self._get_queryset().bulk_create(
            instances=instances, data=data, batch_size=batch_size
        )

    def upsert(self, instances=None, data=None, batch_size=1000):
        return self._get_queryset().upsert(
            instances=instances, data=data, batch_size=batch_size
        )

    def delete(self, **kwargs):
        return self._get_queryset().delete(**kwargs)

    def delete_by_ids(self, ids):
        return self._get_queryset().delete_by_ids(ids)

    def using_partitions(self, *partition_names):
        return self._get_queryset().using_partitions(*partition_names)

    def consistency(self, level):
        return self._get_queryset().consistency(level)

    # ─────────────────────────────────────────────────────
    # Search methods
    # ─────────────────────────────────────────────────────

    def search(self, vector, vector_field=None, limit=10,
               metric_type=None, search_params=None,
               output_fields=None, **kwargs):
        return self._get_queryset().search(
            vector=vector, vector_field=vector_field, limit=limit,
            metric_type=metric_type, search_params=search_params,
            output_fields=output_fields, **kwargs,
        )

    def hybrid_search(self, requests, ranker, limit=10,
                      output_fields=None, **kwargs):
        return self._get_queryset().hybrid_search(
            requests=requests, ranker=ranker, limit=limit,
            output_fields=output_fields, **kwargs,
        )

    # ─────────────────────────────────────────────────────
    # Direct Milvus operations
    # ─────────────────────────────────────────────────────

    def get_client(self):
        """Get the underlying MilvusClient."""
        return self.model.get_client()

    def insert_raw(self, data):
        """Insert raw data dicts directly into the collection."""
        client = self.model.get_client()
        return client.insert(
            collection_name=self.model.get_collection_name(),
            data=data,
        )

    def query_raw(self, filter_expr="", output_fields=None, limit=100,
                  offset=0, **kwargs):
        """Execute a raw Milvus query with filter expressions."""
        client = self.model.get_client()
        return client.query(
            collection_name=self.model.get_collection_name(),
            filter=filter_expr,
            output_fields=output_fields or self.model.get_field_names(),
            limit=limit,
            offset=offset,
            **kwargs,
        )

    def search_raw(self, data, anns_field, limit=10, output_fields=None,
                   search_params=None, filter_expr="", **kwargs):
        """Execute a raw Milvus search."""
        client = self.model.get_client()
        params = {
            "collection_name": self.model.get_collection_name(),
            "data": data,
            "anns_field": anns_field,
            "limit": limit,
            "output_fields": output_fields or self.model.get_field_names(),
        }
        if search_params:
            params["search_params"] = search_params
        if filter_expr:
            params["filter"] = filter_expr
        params.update(kwargs)
        return client.search(**params)

    def get_collection_stats(self):
        """Get collection statistics."""
        client = self.model.get_client()
        return client.get_collection_stats(
            self.model.get_collection_name()
        )

    def describe_collection(self):
        """Describe the collection."""
        client = self.model.get_client()
        return client.describe_collection(
            self.model.get_collection_name()
        )

    def load_collection(self):
        """Load collection into memory."""
        client = self.model.get_client()
        client.load_collection(self.model.get_collection_name())

    def release_collection(self):
        """Release collection from memory."""
        client = self.model.get_client()
        client.release_collection(self.model.get_collection_name())

    def get_load_state(self):
        """Get collection load state."""
        client = self.model.get_client()
        return client.get_load_state(self.model.get_collection_name())

    def flush(self):
        """Flush the collection to persist data."""
        client = self.model.get_client()
        collection = self.model.get_collection_name()
        # MilvusClient doesn't have explicit flush; use refresh_load
        try:
            client.refresh_load(collection)
        except Exception:
            pass
