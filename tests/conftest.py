"""
Shared pytest configuration and fixtures.

Django setup lived at the top of every test module before this file
existed; it now happens once here.
"""

import os

import django
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.settings")
django.setup()

from django.test import override_settings  # noqa: E402

from django_milvus.cache import caches  # noqa: E402
from django_milvus.cache.loadstate import load_state  # noqa: E402


def cache_settings(**overrides):
    """Build a ``MILVUS_CACHE`` block with test-friendly defaults.

    TTL jitter is off and the janitor thread is disabled, so tests are
    deterministic and do not leave threads behind.
    """
    config = {
        "TTL": 300,
        "TTL_JITTER": 0.0,
        "L1": {
            "MAX_MEMORY": "16MB",
            "MAX_ENTRIES": 1000,
            "SHARDS": 2,
            "JANITOR": False,
        },
        "SEMANTIC": {"enabled": True, "threshold": 0.95},
        "STATS": {"enabled": True, "window": 60},
    }
    config.update(overrides)
    return {"default": config}


def with_cache(**overrides):
    """Context manager enabling the cache for one block of a test."""
    return override_settings(MILVUS_CACHE=cache_settings(**overrides))


@pytest.fixture(autouse=True)
def reset_cache_state():
    """Give every test a clean cache.

    The registry, load-state cache and version stamps are process-wide
    singletons, so without this a test would inherit whatever the last one
    left behind.
    """
    caches.reset()
    load_state.clear()
    yield
    caches.reset()
    load_state.clear()


@pytest.fixture
def enabled_cache():
    """Enable a default cache configuration for the test."""
    with override_settings(MILVUS_CACHE=cache_settings()):
        caches.reset()
        yield
        caches.reset()
