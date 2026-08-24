"""
Configuration parsing for the django-milvus caching layer.

Reads the ``MILVUS_CACHE`` Django setting, which is alias-keyed exactly
like the existing ``MILVUS`` setting::

    MILVUS_CACHE = {
        "default": {
            "ENABLED": True,
            "TTL": 300,
            "L1": {"MAX_MEMORY": "256MB", "ALGORITHM": "w-tinylfu"},
            "L2": {"BACKEND": "...RedisBackend", "LOCATION": "redis://..."},
        }
    }

Every key has a default in :data:`DEFAULTS`; user settings are deep-merged
on top, so a config need only specify what it overrides.
"""

import copy
import re

from django.conf import settings

from ..exceptions import CacheConfigurationError


# ─────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────

DEFAULTS = {
    # Master switch for this alias.
    "ENABLED": True,

    # Seconds a cached entry stays fresh. None = never expires (evicted
    # by capacity only).
    "TTL": 300,
    # Fraction of TTL used as random spread, so entries written together
    # do not all expire together. 0.1 => actual TTL in [270, 330] for 300.
    "TTL_JITTER": 0.1,
    # Separate (shorter) TTL for empty result sets.
    "NEGATIVE_TTL": 30,
    # Seconds past expiry an entry may still be served while a refresh
    # happens in the background. 0 disables.
    "STALE_WHILE_REVALIDATE": 60,
    # Serve an expired entry rather than raising when Milvus errors.
    "STALE_IF_ERROR": True,

    # Payloads larger than this are never admitted, so a single huge
    # query cannot flush the whole cache.
    "MAX_ENTRY_BYTES": "1MB",

    # Whether .count() results participate in caching.
    "CACHE_COUNT": True,
    # Whether querysets pinned to Strong consistency may be cached.
    # Off by default: asking for Strong means asking to bypass caches.
    "CACHE_STRONG_CONSISTENCY": False,

    # Log level used when a backend fails and we fall through to Milvus.
    "FAIL_OPEN": True,

    "L1": {
        "BACKEND": "django_milvus.cache.backends.local.LocalRAMBackend",
        # lru | lfu | fifo | random | ttl | slru | 2q | arc | w-tinylfu
        "ALGORITHM": "w-tinylfu",
        "MAX_MEMORY": "256MB",
        "MAX_ENTRIES": 100_000,
        # Lock striping: more shards = less contention, more overhead.
        "SHARDS": 8,
        "WINDOW": {
            # Fraction of capacity given to the admission window segment.
            "admission_ratio": 0.01,
            # Let the hill climber tune admission_ratio automatically.
            "adaptive": True,
            # Fraction of the main segment held as probation (SLRU).
            "probation_ratio": 0.2,
            # Seconds between adaptive samples / janitor passes.
            "sample_interval": 60,
            # Hill-climb step size for admission_ratio.
            "step": 0.05,
        },
        "WATERMARK": {
            # Start a batch eviction once usage crosses high * MAX_MEMORY.
            "high": 0.95,
            # ...and evict down to low * MAX_MEMORY in one pass.
            "low": 0.80,
        },
        "MEMORY_PRESSURE": {
            "enabled": True,
            # Shrink the cache when process RSS exceeds this. Needs psutil.
            "process_rss_limit": "2GB",
            # Never shrink effective capacity below this fraction.
            "floor_ratio": 0.25,
        },
        # Run the background janitor thread (expiry sweep, sketch aging,
        # adaptive window ticks, memory-pressure checks).
        "JANITOR": True,
    },

    # L2 is optional. Omit the key entirely for an L1-only cache.
    "L2": None,

    "SEMANTIC": {
        "enabled": True,
        # Minimum similarity for a near-neighbour query to serve a hit.
        "threshold": 0.97,
        # COSINE | IP | L2 - must match the collection's index metric.
        "metric": "COSINE",
        # Max cached query vectors per bucket.
        "max_vectors": 20_000,
        # Fetch limit*overfetch rows on a miss so rerank has candidates.
        "overfetch": 3,
        # Re-score cached candidates against the real query vector.
        "rerank": True,
        # auto | numpy | hnswlib
        "index": "auto",
    },

    "STAMPEDE": {
        "enabled": True,
        # Seconds a waiter blocks for the in-flight leader before giving
        # up and querying Milvus itself.
        "timeout": 5,
    },

    "VERSIONING": {
        "enabled": True,
        # Mirror version stamps in L2 so all workers agree.
        "shared": True,
        # Seconds a shared stamp is trusted locally before re-reading.
        "refresh_interval": 5,
    },

    "STATS": {
        "enabled": True,
        # Sliding window (seconds) for hit-rate and latency percentiles.
        "window": 300,
    },
}


L2_DEFAULTS = {
    "BACKEND": "django_milvus.cache.backends.redis.RedisBackend",
    "LOCATION": "redis://localhost:6379/0",
    "PREFIX": "dmv",
    # pickle | json | msgpack
    "SERIALIZER": "pickle",
    "COMPRESS": {
        # none | zlib | lz4
        "algorithm": "none",
        "min_bytes": 2048,
        "level": 1,
    },
    "SOCKET_TIMEOUT": 0.2,
    "CIRCUIT_BREAKER": {
        # Consecutive failures before the tier is skipped entirely.
        "failures": 5,
        # Seconds before probing the tier again.
        "reset_after": 30,
    },
    # Extra kwargs passed straight to the backend constructor.
    "OPTIONS": {},
}


VALID_ALGORITHMS = {
    "lru", "lfu", "fifo", "random", "ttl", "slru", "2q", "arc", "w-tinylfu",
}
VALID_SERIALIZERS = {"pickle", "json", "msgpack"}
VALID_COMPRESSORS = {"none", "zlib", "lz4"}
VALID_METRICS = {"COSINE", "IP", "L2"}
VALID_SEMANTIC_INDEXES = {"auto", "numpy", "hnswlib"}


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGT]?I?B?)\s*$", re.I)
_SIZE_UNITS = {
    "": 1, "B": 1,
    "K": 1024, "KB": 1024, "KIB": 1024,
    "M": 1024 ** 2, "MB": 1024 ** 2, "MIB": 1024 ** 2,
    "G": 1024 ** 3, "GB": 1024 ** 3, "GIB": 1024 ** 3,
    "T": 1024 ** 4, "TB": 1024 ** 4, "TIB": 1024 ** 4,
}


def parse_size(value, name="size"):
    """Parse a human byte size into an int.

    Accepts ``"256MB"``, ``"1.5 GiB"``, ``1048576`` or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise CacheConfigurationError(f"{name}: expected a size, got a bool")
    if isinstance(value, (int, float)):
        if value < 0:
            raise CacheConfigurationError(f"{name}: must not be negative")
        return int(value)
    if not isinstance(value, str):
        raise CacheConfigurationError(
            f"{name}: expected a size string or number, got {type(value).__name__}"
        )

    match = _SIZE_RE.match(value)
    if not match:
        raise CacheConfigurationError(
            f"{name}: cannot parse size {value!r}. Use e.g. '256MB' or 268435456."
        )
    number, unit = match.groups()
    unit = unit.upper()
    if unit not in _SIZE_UNITS:
        raise CacheConfigurationError(f"{name}: unknown size unit {unit!r}")
    return int(float(number) * _SIZE_UNITS[unit])


def deep_merge(base, override):
    """Recursively merge ``override`` onto a copy of ``base``."""
    result = copy.deepcopy(base)
    if not override:
        return result
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _require(mapping, key, types, name, allowed=None):
    """Fetch ``mapping[key]``, checking its type and allowed values."""
    value = mapping.get(key)
    expected = types if isinstance(types, tuple) else (types,)

    if not isinstance(value, expected):
        names = "/".join(t.__name__ for t in expected)
        raise CacheConfigurationError(
            f"{name}.{key}: expected {names}, got {value!r}"
        )
    # bool is a subclass of int; reject it unless explicitly expected.
    if isinstance(value, bool) and bool not in expected:
        raise CacheConfigurationError(
            f"{name}.{key}: expected a non-boolean value, got {value!r}"
        )
    if allowed is not None and value not in allowed:
        raise CacheConfigurationError(
            f"{name}.{key}: {value!r} is not one of {sorted(allowed)}"
        )
    return value


def _ratio(mapping, key, name):
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CacheConfigurationError(
            f"{name}.{key}: expected a number in [0, 1], got {value!r}"
        )
    if not 0.0 <= float(value) <= 1.0:
        raise CacheConfigurationError(
            f"{name}.{key}: must be between 0 and 1, got {value!r}"
        )
    return float(value)


# ─────────────────────────────────────────────────────────
# Config objects
# ─────────────────────────────────────────────────────────

class L1Config:
    """Parsed and validated L1 (in-process RAM) configuration."""

    __slots__ = (
        "backend", "algorithm", "max_memory", "max_entries", "shards",
        "admission_ratio", "adaptive", "probation_ratio", "sample_interval",
        "step", "high_watermark", "low_watermark", "pressure_enabled",
        "process_rss_limit", "floor_ratio", "janitor", "raw",
    )

    def __init__(self, raw, name="MILVUS_CACHE.L1"):
        self.raw = raw
        self.backend = _require(raw, "BACKEND", str, name)
        self.algorithm = _require(
            raw, "ALGORITHM", str, name, allowed=VALID_ALGORITHMS
        )
        self.max_memory = parse_size(raw.get("MAX_MEMORY"), f"{name}.MAX_MEMORY")
        self.max_entries = raw.get("MAX_ENTRIES")
        if self.max_entries is not None and (
            not isinstance(self.max_entries, int)
            or isinstance(self.max_entries, bool)
            or self.max_entries <= 0
        ):
            raise CacheConfigurationError(
                f"{name}.MAX_ENTRIES: expected a positive int or None, "
                f"got {self.max_entries!r}"
            )
        if self.max_memory is None and self.max_entries is None:
            raise CacheConfigurationError(
                f"{name}: set at least one of MAX_MEMORY or MAX_ENTRIES, "
                f"otherwise the cache is unbounded."
            )

        shards = raw.get("SHARDS", 8)
        if not isinstance(shards, int) or isinstance(shards, bool) or shards < 1:
            raise CacheConfigurationError(
                f"{name}.SHARDS: expected a positive int, got {shards!r}"
            )
        self.shards = shards

        window = raw.get("WINDOW") or {}
        self.admission_ratio = _ratio(window, "admission_ratio", f"{name}.WINDOW")
        self.probation_ratio = _ratio(window, "probation_ratio", f"{name}.WINDOW")
        self.adaptive = bool(window.get("adaptive", True))
        interval = window.get("sample_interval", 60)
        if not isinstance(interval, (int, float)) or interval <= 0:
            raise CacheConfigurationError(
                f"{name}.WINDOW.sample_interval: expected a positive number, "
                f"got {interval!r}"
            )
        self.sample_interval = float(interval)
        self.step = _ratio(window, "step", f"{name}.WINDOW")

        marks = raw.get("WATERMARK") or {}
        self.high_watermark = _ratio(marks, "high", f"{name}.WATERMARK")
        self.low_watermark = _ratio(marks, "low", f"{name}.WATERMARK")
        if self.low_watermark >= self.high_watermark:
            raise CacheConfigurationError(
                f"{name}.WATERMARK: low ({self.low_watermark}) must be less "
                f"than high ({self.high_watermark})"
            )

        pressure = raw.get("MEMORY_PRESSURE") or {}
        self.pressure_enabled = bool(pressure.get("enabled", False))
        self.process_rss_limit = parse_size(
            pressure.get("process_rss_limit"), f"{name}.MEMORY_PRESSURE"
        )
        self.floor_ratio = _ratio(
            {"floor_ratio": pressure.get("floor_ratio", 0.25)},
            "floor_ratio", f"{name}.MEMORY_PRESSURE",
        )
        self.janitor = bool(raw.get("JANITOR", True))


class L2Config:
    """Parsed and validated L2 (shared) configuration."""

    __slots__ = (
        "backend", "location", "prefix", "serializer", "compress_algorithm",
        "compress_min_bytes", "compress_level", "socket_timeout",
        "breaker_failures", "breaker_reset_after", "options", "raw",
    )

    def __init__(self, raw, name="MILVUS_CACHE.L2"):
        self.raw = raw
        self.backend = _require(raw, "BACKEND", str, name)
        self.location = raw.get("LOCATION")
        self.prefix = raw.get("PREFIX", "dmv")
        self.serializer = _require(
            raw, "SERIALIZER", str, name, allowed=VALID_SERIALIZERS
        )

        compress = raw.get("COMPRESS") or {}
        self.compress_algorithm = compress.get("algorithm", "none")
        if self.compress_algorithm not in VALID_COMPRESSORS:
            raise CacheConfigurationError(
                f"{name}.COMPRESS.algorithm: {self.compress_algorithm!r} is not "
                f"one of {sorted(VALID_COMPRESSORS)}"
            )
        self.compress_min_bytes = parse_size(
            compress.get("min_bytes", 2048), f"{name}.COMPRESS.min_bytes"
        )
        self.compress_level = int(compress.get("level", 1))

        timeout = raw.get("SOCKET_TIMEOUT", 0.2)
        if timeout is not None and (
            not isinstance(timeout, (int, float)) or timeout <= 0
        ):
            raise CacheConfigurationError(
                f"{name}.SOCKET_TIMEOUT: expected a positive number or None, "
                f"got {timeout!r}"
            )
        self.socket_timeout = timeout

        breaker = raw.get("CIRCUIT_BREAKER") or {}
        self.breaker_failures = int(breaker.get("failures", 5))
        self.breaker_reset_after = float(breaker.get("reset_after", 30))
        self.options = dict(raw.get("OPTIONS") or {})


class SemanticConfig:
    """Parsed and validated semantic-cache configuration."""

    __slots__ = (
        "enabled", "threshold", "metric", "max_vectors", "overfetch",
        "rerank", "index", "raw",
    )

    def __init__(self, raw, name="MILVUS_CACHE.SEMANTIC"):
        self.raw = raw
        self.enabled = bool(raw.get("enabled", False))
        threshold = raw.get("threshold", 0.97)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise CacheConfigurationError(
                f"{name}.threshold: expected a number, got {threshold!r}"
            )
        self.threshold = float(threshold)
        self.metric = str(raw.get("metric", "COSINE")).upper()
        if self.metric not in VALID_METRICS:
            raise CacheConfigurationError(
                f"{name}.metric: {self.metric!r} is not one of "
                f"{sorted(VALID_METRICS)}"
            )
        if self.metric in ("COSINE", "IP") and not 0.0 <= self.threshold <= 1.0:
            raise CacheConfigurationError(
                f"{name}.threshold: for metric {self.metric} must be in [0, 1], "
                f"got {self.threshold}"
            )
        max_vectors = raw.get("max_vectors", 20_000)
        if not isinstance(max_vectors, int) or max_vectors < 1:
            raise CacheConfigurationError(
                f"{name}.max_vectors: expected a positive int, got {max_vectors!r}"
            )
        self.max_vectors = max_vectors
        overfetch = raw.get("overfetch", 3)
        if not isinstance(overfetch, (int, float)) or overfetch < 1:
            raise CacheConfigurationError(
                f"{name}.overfetch: expected a number >= 1, got {overfetch!r}"
            )
        self.overfetch = float(overfetch)
        self.rerank = bool(raw.get("rerank", True))
        self.index = str(raw.get("index", "auto")).lower()
        if self.index not in VALID_SEMANTIC_INDEXES:
            raise CacheConfigurationError(
                f"{name}.index: {self.index!r} is not one of "
                f"{sorted(VALID_SEMANTIC_INDEXES)}"
            )

    def replace(self, **overrides):
        """Return a copy with ``overrides`` applied (per-query tuning)."""
        return SemanticConfig(deep_merge(self.raw, overrides))


class CacheConfig:
    """Fully parsed configuration for one cache alias."""

    __slots__ = (
        "alias", "enabled", "ttl", "ttl_jitter", "negative_ttl",
        "stale_while_revalidate", "stale_if_error", "max_entry_bytes",
        "cache_count", "cache_strong_consistency", "fail_open",
        "l1", "l2", "semantic", "stampede_enabled", "stampede_timeout",
        "versioning_enabled", "versioning_shared", "versioning_refresh",
        "stats_enabled", "stats_window", "raw",
    )

    def __init__(self, alias, raw):
        self.alias = alias
        self.raw = raw
        name = f"MILVUS_CACHE[{alias!r}]"

        self.enabled = bool(raw.get("ENABLED", True))
        self.ttl = raw.get("TTL")
        if self.ttl is not None and (
            not isinstance(self.ttl, (int, float)) or self.ttl <= 0
        ):
            raise CacheConfigurationError(
                f"{name}.TTL: expected a positive number or None, got {self.ttl!r}"
            )
        self.ttl_jitter = _ratio(raw, "TTL_JITTER", name)
        self.negative_ttl = raw.get("NEGATIVE_TTL")
        self.stale_while_revalidate = float(raw.get("STALE_WHILE_REVALIDATE") or 0)
        self.stale_if_error = bool(raw.get("STALE_IF_ERROR", True))
        self.max_entry_bytes = parse_size(
            raw.get("MAX_ENTRY_BYTES"), f"{name}.MAX_ENTRY_BYTES"
        )
        self.cache_count = bool(raw.get("CACHE_COUNT", True))
        self.cache_strong_consistency = bool(
            raw.get("CACHE_STRONG_CONSISTENCY", False)
        )
        self.fail_open = bool(raw.get("FAIL_OPEN", True))

        self.l1 = L1Config(raw.get("L1") or {}, f"{name}.L1")

        l2_raw = raw.get("L2")
        self.l2 = L2Config(deep_merge(L2_DEFAULTS, l2_raw), f"{name}.L2") \
            if l2_raw else None

        self.semantic = SemanticConfig(
            raw.get("SEMANTIC") or {}, f"{name}.SEMANTIC"
        )

        stampede = raw.get("STAMPEDE") or {}
        self.stampede_enabled = bool(stampede.get("enabled", True))
        self.stampede_timeout = float(stampede.get("timeout", 5))

        versioning = raw.get("VERSIONING") or {}
        self.versioning_enabled = bool(versioning.get("enabled", True))
        self.versioning_shared = bool(versioning.get("shared", True))
        self.versioning_refresh = float(versioning.get("refresh_interval", 5))

        stats = raw.get("STATS") or {}
        self.stats_enabled = bool(stats.get("enabled", True))
        self.stats_window = float(stats.get("window", 300))

    def __repr__(self):
        tiers = "L1" + ("+L2" if self.l2 else "")
        return (
            f"<CacheConfig alias={self.alias!r} enabled={self.enabled} "
            f"tiers={tiers} algorithm={self.l1.algorithm!r}>"
        )


# ─────────────────────────────────────────────────────────
# Settings access
# ─────────────────────────────────────────────────────────

def get_raw_settings():
    """Return the raw ``MILVUS_CACHE`` dict (empty if unset)."""
    raw = getattr(settings, "MILVUS_CACHE", None)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise CacheConfigurationError(
            f"MILVUS_CACHE must be a dict of alias -> config, "
            f"got {type(raw).__name__}"
        )
    return raw


def is_configured():
    """True when the project has any ``MILVUS_CACHE`` configuration."""
    return bool(get_raw_settings())


def load_config(alias="default"):
    """Build a :class:`CacheConfig` for ``alias``.

    Raises :class:`CacheConfigurationError` when the alias is unknown, so
    a typo surfaces loudly rather than silently disabling caching.
    """
    raw_all = get_raw_settings()
    if alias not in raw_all:
        raise CacheConfigurationError(
            f"No MILVUS_CACHE configuration for alias {alias!r}. "
            f"Known aliases: {sorted(raw_all) or 'none'}."
        )
    user = raw_all[alias]
    if not isinstance(user, dict):
        raise CacheConfigurationError(
            f"MILVUS_CACHE[{alias!r}] must be a dict, got {type(user).__name__}"
        )
    # L2 stays None unless the user supplied it; deep_merge would not
    # invent it, but be explicit for readability.
    merged = deep_merge(DEFAULTS, user)
    return CacheConfig(alias, merged)


def validate_all():
    """Parse every configured alias, raising on the first bad one.

    Called from ``AppConfig.ready()`` so misconfiguration fails at start-up
    instead of at the first query.
    """
    return {alias: load_config(alias) for alias in get_raw_settings()}
