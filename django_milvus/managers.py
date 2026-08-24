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

# Operation tags used to keep cache keys for different read shapes apart.
_OP_QUERY = "q"
_OP_SEARCH = "s"
_OP_HYBRID = "h"
_OP_COUNT = "c"


def normalize_query_results(results):
    """Convert a Milvus query response into plain dicts.

    pymilvus returns objects that behave like dicts but are not, and that
    do not necessarily survive a round trip through pickle or msgpack.
    Anything destined for the cache is flattened first, so the stored
    payload is ordinary data with no library types embedded in it.
    """
    if results is None:
        return []
    return [dict(row) for row in results]


def normalize_search_results(results):
    """Convert a Milvus search response into plain nested dicts.

    The grouping is preserved - one inner list per query vector - so
    :meth:`MilvusQuerySet._parse_search_results` works identically on a
    live response and a cached one.
    """
    if results is None:
        return []
    normalized = []
    for group in results:
        rows = []
        for hit in group:
            if isinstance(hit, dict):
                row = dict(hit)
            else:
                row = {
                    "id": hit.get("id") if hasattr(hit, "get") else None,
                    "distance": (
                        hit.get("distance") if hasattr(hit, "get") else None
                    ),
                    "entity": hit.get("entity") if hasattr(hit, "get") else hit,
                }
            entity = row.get("entity")
            if entity is not None and not isinstance(entity, dict):
                try:
                    entity = dict(entity)
                except (TypeError, ValueError):
                    entity = {}
            if entity is not None:
                row["entity"] = entity
            rows.append(row)
        normalized.append(rows)
    return normalized


class _CachePlan:
    """Bridges one queryset evaluation to the cache orchestrator.

    Gathers the query's identity (collection, filter, fields, limit,
    vectors) into the shape :class:`~.cache.registry.MilvusCache` expects,
    and dispatches to the right entry point for the operation.

    Constructing a plan is cheap and side-effect free, so ``cache_key()``
    can build one purely to report the key it would use.
    """

    __slots__ = ("qs", "cache", "options", "op")

    def __init__(self, queryset, cache, options, op=None):
        self.qs = queryset
        self.cache = cache
        self.options = options
        self.op = op or self._infer_op()

    def _infer_op(self):
        if self.qs._is_hybrid:
            return _OP_HYBRID
        if self.qs._is_search:
            return _OP_SEARCH
        return _OP_QUERY

    # ── query identity ───────────────────────────────────

    def _context(self):
        qs = self.qs
        return {
            "collection": qs.model.get_collection_name(),
            "filter_expr": qs._build_filter(),
            "output_fields": qs._get_output_fields(),
            "offset": qs._offset,
            "partitions": qs._partition_names,
            "consistency": qs._consistency_level,
        }

    def _search_context(self):
        """Search identity, or None when it cannot be pinned down.

        A hybrid search is only cacheable if its requests can be
        fingerprinted stably; ranker and request objects come from
        pymilvus and need not have deterministic reprs. When they cannot
        be fingerprinted the plan declines to cache rather than risk two
        different searches colliding on one key.
        """
        qs = self.qs
        params = qs._search_params or {}
        context = self._context()

        if self.op == _OP_HYBRID:
            fingerprint = _fingerprint_hybrid(
                params.get("requests"), params.get("ranker")
            )
            if fingerprint is None:
                return None
            context["extra"] = fingerprint
            context["limit"] = params.get("limit")
            return context

        context["limit"] = params.get("limit")
        context["vectors"] = params.get("data")
        context["anns_field"] = params.get("anns_field")
        context["search_params"] = params.get("search_params")
        return context

    @property
    def key(self):
        """The cache key this query would use, or None."""
        from .cache.keys import query_key, search_key, vector_fingerprint

        alias = self.cache.alias
        context = self._context()
        collection = context.pop("collection")
        version = self.cache.version(collection)

        if self.op == _OP_QUERY:
            return query_key(
                alias, collection, version, limit=self.qs._limit,
                op=_OP_QUERY, **context
            )

        search_context = self._search_context()
        if search_context is None:
            return None
        search_context.pop("collection")

        if self.op == _OP_HYBRID:
            return query_key(
                alias, collection, version, op=_OP_HYBRID, **search_context
            )

        vectors = search_context.pop("vectors")
        return search_key(
            alias, collection, version,
            vector_fp=vector_fingerprint(vectors), op=_OP_SEARCH,
            **search_context
        )

    # ── dispatch ─────────────────────────────────────────

    def run(self, producer):
        """Serve this query from cache, or run ``producer`` and cache it."""
        sender = self.qs.model

        if self.op == _OP_QUERY:
            context = self._context()
            collection = context.pop("collection")
            result = self.cache.query(
                collection=collection, producer=producer,
                options=self.options, sender=sender,
                limit=self.qs._limit, op=_OP_QUERY, **context
            )
            return result.value

        context = self._search_context()
        if context is None:
            # Not safely identifiable: run it uncached rather than risk a
            # key collision between two different searches.
            return producer(self.qs._search_params.get("limit"))
        collection = context.pop("collection")

        if self.op == _OP_HYBRID:
            result = self.cache.query(
                collection=collection, producer=producer,
                options=self.options, sender=sender, op=_OP_HYBRID, **context
            )
            return result.value

        result = self.cache.search(
            collection=collection, producer=producer,
            options=self.options, sender=sender, op=_OP_SEARCH,
            metric=(context.get("search_params") or {}).get("metric_type"),
            **context
        )
        return result.value


def _fingerprint_hybrid(requests, ranker):
    """Stable identity for a hybrid search, or None if there isn't one.

    ``AnnSearchRequest`` keeps its fields on private attributes with
    public properties; both spellings are tried. Anything we cannot read
    means we cannot promise two searches differ, so the caller declines
    to cache.
    """
    if not requests:
        return None

    from .cache.keys import vector_fingerprint

    parts = []
    for request in requests:
        if isinstance(request, dict):
            data = request.get("data")
            field = request.get("anns_field")
            param = request.get("param")
            limit = request.get("limit")
            expr = request.get("expr")
        else:
            def read(*names):
                for name in names:
                    if hasattr(request, name):
                        return getattr(request, name)
                return _UNREADABLE

            data = read("data", "_data")
            field = read("anns_field", "_anns_field")
            param = read("param", "_param")
            limit = read("limit", "_limit")
            expr = read("expr", "_expr")
            if _UNREADABLE in (data, field):
                return None

        try:
            parts.append(
                (vector_fingerprint(data), field, repr(param), limit, expr)
            )
        except Exception:
            return None

    ranker_id = getattr(ranker, "name", None) or repr(ranker)
    return ("hybrid", tuple(parts), ranker_id)


class _Unreadable:
    """Marker for a request attribute we could not read."""

    def __repr__(self):
        return "<unreadable>"


_UNREADABLE = _Unreadable()


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

        # Cross-request caching. None means "use whatever the model's
        # MilvusMeta.cache says"; a dict is an explicit per-query override.
        self._cache_opts = None

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
        qs._cache_opts = (
            dict(self._cache_opts) if self._cache_opts is not None else None
        )
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
    # Caching
    # ─────────────────────────────────────────────────────

    def cache(self, ttl=None, semantic=None, alias=None, store_vectors=None,
              keep_vectors=None, refresh=None, **extra):
        """Cache this query's results across requests.

        Requires ``MILVUS_CACHE`` in settings. Enables caching for this
        query even when the model has not opted in, and overrides the
        model's settings when it has.

        Args:
            ttl: Seconds to keep the result. Defaults to the alias ``TTL``.
            semantic: Nearest-vector matching for searches. ``False``
                disables it, ``True`` enables it with configured defaults,
                and a float is shorthand for the similarity threshold::

                    .cache(semantic=0.99)
                    .cache(semantic={"threshold": 0.99, "rerank": False})

            alias: Which ``MILVUS_CACHE`` alias to use.
            store_vectors: Fetch the vector field alongside results so
                embeddings are available for reranking. The vectors are
                hoisted into the shared vector cache and stripped from the
                returned entities, so this costs a wider Milvus response
                but not a wider cached payload.
            keep_vectors: Leave embeddings in the returned entities
                instead of stripping them.
            refresh: Skip the lookup and repopulate from Milvus. See also
                :meth:`refresh_cache`.

        Examples::

            Document.objects.search(v, limit=5).cache(ttl=60)
            Document.objects.filter(category="x").cache(ttl=300)
        """
        qs = self._clone()
        options = dict(qs._cache_opts or {})
        options["enabled"] = True
        if ttl is not None:
            options["ttl"] = ttl
        if semantic is not None:
            options["semantic"] = semantic
        if alias is not None:
            options["alias"] = alias
        if store_vectors is not None:
            options["store_vectors"] = store_vectors
        if keep_vectors is not None:
            options["keep_vectors"] = keep_vectors
        if refresh is not None:
            options["refresh"] = refresh
        options.update(extra)
        qs._cache_opts = options
        return qs

    def no_cache(self):
        """Bypass the cache for this query, whatever the model says."""
        qs = self._clone()
        qs._cache_opts = {"enabled": False}
        return qs

    def refresh_cache(self):
        """Query Milvus and overwrite the cached entry.

        Use after a write made outside the ORM, or to warm an entry ahead
        of traffic.
        """
        return self.cache(refresh=True)

    def cache_key(self):
        """The key this query would use, or None if it will not be cached.

        Handy when debugging a cache that seems not to be working: two
        queries you expect to share a result must produce the same key.
        """
        plan = self._cache_plan()
        return plan.key if plan is not None else None

    # ── cache plumbing ───────────────────────────────────

    def _cache_options(self):
        """Effective options for this query, or None if it is not cached."""
        from .cache import resolve_cache_options
        return resolve_cache_options(self.model, self._cache_opts)

    def _cache_for(self, options):
        """The cache instance for these options, or None."""
        if options is None:
            return None
        # Strong consistency is an explicit request for the freshest data
        # Milvus can give. Serving that from cache would defeat the point,
        # so it opts out unless the config says otherwise.
        if (
            self._consistency_level
            and str(self._consistency_level).lower() == "strong"
        ):
            from .cache import get_cache
            cache = get_cache(options.get("alias", "default"))
            if cache is None or not cache.config.cache_strong_consistency:
                return None
            return cache
        from .cache import get_cache
        return get_cache(options.get("alias", "default"))

    def _cache_plan(self, op=None):
        """Resolve everything needed to cache this query, or None."""
        options = self._cache_options()
        cache = self._cache_for(options)
        if cache is None:
            return None
        return _CachePlan(self, cache, options, op=op)

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
        """Ensure collection indexes exist and collection is loaded.

        Remembers what it has already done, so a cold collection costs the
        error-index-load-retry sequence once rather than on every miss.
        The memory has a TTL and is cleared on any Milvus error, so a
        collection released out from under us recovers on the next query.
        """
        from .cache.loadstate import load_state

        alias = self.model.get_database_alias()
        try:
            if not load_state.has_indexes(collection, alias):
                try:
                    _create_indexes_for_model(client, self.model)
                except Exception as e:
                    # Ignore "already exists" errors, re-raise others
                    if ("already exists" not in str(e).lower()
                            and "Duplicate" not in str(e)):
                        raise
                load_state.mark_indexed(collection, alias)

            client.load_collection(collection)
            load_state.mark_loaded(collection, alias)
        except Exception:
            load_state.invalidate(collection, alias)
            raise

    # ─────────────────────────────────────────────────────
    # Execution
    #
    # Each operation is split in two: a `_*_uncached` method that talks to
    # Milvus and normalises the response, and an `_execute_*` wrapper that
    # routes it through the cache when one is configured. The cache
    # therefore sits *below* result parsing and stores the raw wire shape
    # (plain lists of dicts), never model instances - which keeps it
    # serializer-agnostic and lets _parse_search_results stay untouched.
    # ─────────────────────────────────────────────────────

    def _execute_query(self):
        """Execute a query (non-search), through the cache when enabled."""
        plan = self._cache_plan(_OP_QUERY)
        if plan is None:
            return self._query_uncached(self._limit)
        return plan.run(self._query_uncached)

    def _query_uncached(self, limit=None):
        """Query Milvus directly and normalise the result."""
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

        if limit is not None:
            kwargs["limit"] = limit
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
        return normalize_query_results(results)

    def _execute_search(self):
        """Execute a vector search, through the cache when enabled."""
        plan = self._cache_plan(_OP_SEARCH)
        if plan is None:
            return self._search_uncached(self._search_params.get("limit"))
        return plan.run(self._search_uncached)

    def _search_uncached(self, limit=None):
        """Search Milvus directly and normalise the result."""
        client = self.model.get_client()
        collection = self.model.get_collection_name()
        filter_expr = self._build_filter()

        params = self._search_params.copy()
        params["collection_name"] = collection
        if limit is not None:
            params["limit"] = limit

        if filter_expr:
            params["filter"] = filter_expr

        if params.get("output_fields") is None:
            params["output_fields"] = self._get_output_fields()
        params["output_fields"] = self._with_stored_vectors(
            params["output_fields"]
        )

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
        return normalize_search_results(results)

    def _execute_hybrid_search(self):
        """Execute a hybrid search, through the cache when enabled."""
        plan = self._cache_plan(_OP_HYBRID)
        if plan is None:
            return self._hybrid_uncached(self._search_params.get("limit"))
        return plan.run(self._hybrid_uncached)

    def _hybrid_uncached(self, limit=None):
        """Run a hybrid search directly and normalise the result."""
        client = self.model.get_client()
        collection = self.model.get_collection_name()

        params = self._search_params.copy()
        params["collection_name"] = collection
        if limit is not None:
            params["limit"] = limit

        if params.get("output_fields") is None:
            params["output_fields"] = self._get_output_fields()

        if self._partition_names:
            params["partition_names"] = self._partition_names

        try:
            results = client.hybrid_search(**params)
        except MilvusException as e:
            if "not loaded" in str(e):
                self._ensure_loaded(client, collection)
                results = client.hybrid_search(**params)
            else:
                raise
        return normalize_search_results(results)

    def _with_stored_vectors(self, output_fields):
        """Add the vector field to a search's output when asked.

        ``.cache(store_vectors=True)`` needs embeddings back from Milvus so
        the cache can file them for later reranking. They are hoisted into
        the shared vector cache and stripped from the entities afterwards,
        so this widens the Milvus response but not the cached payload.
        """
        options = self._cache_opts
        if not options or not options.get("store_vectors"):
            return output_fields
        field = (self._search_params or {}).get("anns_field")
        if not field or not output_fields:
            return output_fields
        if field in output_fields:
            return output_fields
        return list(output_fields) + [field]

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
        plan = self._cache_plan(_OP_COUNT)
        if plan is not None and not plan.cache.config.cache_count:
            plan = None

        if plan is None:
            results = self._count_uncached()
        else:
            results = plan.cache.count(
                collection=self.model.get_collection_name(),
                producer=self._count_uncached,
                options=plan.options,
                sender=self.model,
                filter_expr=self._build_filter(),
                output_fields=["count(*)"],
                consistency=self._consistency_level,
                partitions=self._partition_names,
            ).value

        if results and "count(*)" in results[0]:
            return results[0]["count(*)"]
        return len(results)

    def _count_uncached(self, limit=None):
        """Run ``count(*)`` against Milvus."""
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

        try:
            results = client.query(**kwargs)
        except MilvusException as e:
            # count() reached Milvus without this retry, so a cold
            # collection made it fail where every other read recovered.
            if "not loaded" in str(e):
                self._ensure_loaded(client, collection)
                results = client.query(**kwargs)
            else:
                raise
        return normalize_query_results(results)

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

        self.model.invalidate_cache()
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

        self.model.invalidate_cache()
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

        result = client.delete(
            collection_name=collection,
            filter=filter_expr,
            **extra,
        )
        self.model.invalidate_cache()
        return result

    def delete_by_ids(self, ids):
        """Delete entities by their primary key IDs.

        Args:
            ids: List of primary key values.
        """
        client = self.model.get_client()
        collection = self.model.get_collection_name()
        result = client.delete(
            collection_name=collection,
            ids=ids,
        )
        self.model.invalidate_cache()
        return result

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
        """Proxy attribute access to the entity.

        ``__getattr__`` runs only for attributes not found normally, which
        during unpickling includes ``entity`` itself - the instance dict is
        still empty when pickle probes for ``__setstate__``. Reading
        ``self.entity`` there would recurse until the stack ran out, so
        both dunder lookups and the wrapper's own fields short-circuit,
        and the entity is read from ``__dict__`` rather than by attribute.
        """
        if name.startswith("__") or name in (
            "entity", "distance", "score", "id"
        ):
            raise AttributeError(name)
        entity = self.__dict__.get("entity")
        if entity is None:
            raise AttributeError(name)
        return getattr(entity, name)


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
    # Caching
    # ─────────────────────────────────────────────────────

    def cache(self, **kwargs):
        """Start a cached queryset. See :meth:`MilvusQuerySet.cache`."""
        return self._get_queryset().cache(**kwargs)

    def no_cache(self):
        """Start a queryset that bypasses the cache."""
        return self._get_queryset().no_cache()

    def cache_stats(self):
        """Statistics for the cache this model uses.

        Returns an empty dict when caching is not configured.
        """
        from .cache import get_cache, resolve_cache_options
        options = resolve_cache_options(self.model, None) or {}
        cache = get_cache(options.get("alias", "default"))
        if cache is None:
            return {}
        data = cache.stats_dict()
        data["collection"] = self.model.get_collection_name()
        data["collection_version"] = cache.version(
            self.model.get_collection_name()
        )
        return data

    def cache_clear(self):
        """Drop every cached entry for this model's collection.

        Returns the number of entries removed. Prefer this to clearing the
        whole cache: other collections are unaffected.
        """
        from .cache import get_cache, resolve_cache_options
        options = resolve_cache_options(self.model, None) or {}
        cache = get_cache(options.get("alias", "default"))
        if cache is None:
            return 0
        return cache.clear(self.model.get_collection_name())

    def cache_warm(self, queries=None, vectors=None, limit=10, **kwargs):
        """Populate the cache ahead of traffic.

        Run at deploy time so the first real users do not each pay for a
        cold cache.

        Args:
            queries: Querysets or callables to evaluate. Each is forced
                through the cache.
            vectors: Query vectors to search and cache, as an alternative
                to building querysets yourself.
            limit: Result limit for ``vectors``.
            **kwargs: Passed to ``.cache()`` for the warmed queries.

        Returns the number of entries warmed.
        """
        from .cache.warmup import warm

        return warm(
            self.model, queries=queries, vectors=vectors, limit=limit, **kwargs
        )

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
        result = client.insert(
            collection_name=self.model.get_collection_name(),
            data=data,
        )
        self.model.invalidate_cache()
        return result

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
        from .cache.loadstate import load_state
        client = self.model.get_client()
        collection = self.model.get_collection_name()
        client.release_collection(collection)
        # Forget the cached load state, or the next read would assume the
        # collection is still loaded and skip reloading it.
        load_state.invalidate(collection, self.model.get_database_alias())

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
