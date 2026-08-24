"""
Cache warm-up.

A cold cache means the first user of every query pays full price. After a
deploy or a restart that is every user at once, which is exactly when the
system is least able to absorb it - and with an in-process L1, every
worker starts cold independently.

Warming runs the queries you already know are coming, before the traffic
arrives::

    Document.objects.cache_warm(vectors=common_embeddings, limit=10)
    Document.objects.cache_warm(queries=[
        Document.objects.filter(status="published").limit(50),
    ])

Failures are collected rather than raised: warming is best-effort, and a
single bad query should not abort a deploy step or leave the rest of the
cache unwarmed.
"""

import logging

logger = logging.getLogger("django_milvus.cache")


class WarmupResult:
    """What a warm-up run accomplished."""

    def __init__(self):
        self.warmed = 0
        self.skipped = 0
        self.errors = []

    def record_success(self):
        self.warmed += 1

    def record_skip(self):
        self.skipped += 1

    def record_error(self, target, exc):
        self.errors.append((repr(target), str(exc)))

    def as_dict(self):
        return {
            "warmed": self.warmed,
            "skipped": self.skipped,
            "errors": self.errors,
        }

    def __int__(self):
        return self.warmed

    def __bool__(self):
        return self.warmed > 0

    def __repr__(self):
        return (
            f"<WarmupResult warmed={self.warmed} skipped={self.skipped} "
            f"errors={len(self.errors)}>"
        )


def warm(model, queries=None, vectors=None, limit=10, vector_field=None,
         **cache_kwargs):
    """Populate the cache for ``model``.

    Args:
        model: The :class:`~django_milvus.models.MilvusModel` to warm.
        queries: Querysets or zero-argument callables returning one.
        vectors: Query vectors to search and cache.
        limit: Result limit used for ``vectors``.
        vector_field: Vector field to search; auto-detected when the model
            has exactly one.
        **cache_kwargs: Forwarded to ``.cache()`` (``ttl``, ``semantic``,
            ``alias``, ``store_vectors``).

    Returns:
        WarmupResult
    """
    result = WarmupResult()

    from . import get_cache, resolve_cache_options
    options = resolve_cache_options(model, cache_kwargs or None) or {}
    if get_cache(options.get("alias", "default")) is None:
        logger.info(
            "django-milvus cache: nothing to warm - no cache configured for "
            "alias %r", options.get("alias", "default"),
        )
        return result

    for target in (queries or []):
        try:
            queryset = target() if callable(target) else target
            if queryset is None:
                result.record_skip()
                continue
            # Force evaluation so the result lands in the cache.
            list(queryset.cache(**cache_kwargs))
            result.record_success()
        except Exception as exc:
            logger.warning(
                "django-milvus cache: could not warm %r", target, exc_info=True
            )
            result.record_error(target, exc)

    for vector in (vectors or []):
        try:
            queryset = model.objects.search(
                vector, vector_field=vector_field, limit=limit
            ).cache(**cache_kwargs)
            list(queryset)
            result.record_success()
        except Exception as exc:
            logger.warning(
                "django-milvus cache: could not warm a query vector",
                exc_info=True,
            )
            result.record_error(vector, exc)

    logger.info(
        "django-milvus cache: warmed %d entries for %s (%d errors)",
        result.warmed, model.__name__, len(result.errors),
    )
    return result


def warm_models(models, **kwargs):
    """Warm several models, returning a result per model."""
    return {model.__name__: warm(model, **kwargs) for model in models}
