"""
Management command to show django-milvus cache statistics.

Usage:
    python manage.py milvus_cache_stats
    python manage.py milvus_cache_stats --alias default
    python manage.py milvus_cache_stats --json
    python manage.py milvus_cache_stats --prometheus
"""

import json

from django.core.management.base import BaseCommand

from django_milvus.cache import cache_stats, is_configured
from django_milvus.cache.stats import prometheus_metrics


class Command(BaseCommand):
    help = "Show django-milvus cache statistics (hit rate, memory, evictions)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--alias",
            default=None,
            help="Cache alias to inspect (default: all configured aliases)",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            default=False,
            dest="as_json",
            help="Emit raw JSON instead of a formatted report",
        )
        parser.add_argument(
            "--prometheus",
            action="store_true",
            default=False,
            help="Emit Prometheus text exposition format",
        )

    def handle(self, *args, **options):
        if not is_configured():
            self.stdout.write(
                self.style.WARNING(
                    "No MILVUS_CACHE configured. Caching is off; every read "
                    "goes to Milvus."
                )
            )
            return

        alias = options.get("alias")
        data = cache_stats(alias)
        if alias:
            data = {alias: data}

        if not data or not any(data.values()):
            self.stdout.write(
                "No cache activity yet. Statistics appear once a cached "
                "query has run in this process."
            )
            return

        if options["prometheus"]:
            self.stdout.write(prometheus_metrics(data))
            return

        if options["as_json"]:
            self.stdout.write(json.dumps(data, indent=2, default=str))
            return

        for name, stats in data.items():
            self._report(name, stats)

    # ── formatted report ─────────────────────────────────

    def _report(self, alias, stats):
        if not stats:
            return

        self.stdout.write(self.style.SUCCESS(f"\nCache alias: {alias}"))
        self.stdout.write("=" * 60)

        lookups = stats.get("lookups", 0)
        hit_rate = stats.get("hit_rate", 0.0)
        style = (
            self.style.SUCCESS if hit_rate >= 0.5
            else self.style.WARNING if hit_rate >= 0.2
            else self.style.ERROR
        )
        self.stdout.write("\n  Effectiveness")
        self.stdout.write(f"    Lookups:          {lookups:,}")
        self.stdout.write(f"    Hit rate:         {style(f'{hit_rate:.1%}')}")
        self.stdout.write(
            f"    Recent hit rate:  {stats.get('recent_hit_rate', 0):.1%}"
        )
        self.stdout.write(
            f"    Hits:             {stats.get('hits', 0):,} "
            f"(local {stats.get('l1_hits', 0):,}, "
            f"shared {stats.get('l2_hits', 0):,}, "
            f"semantic {stats.get('semantic_hits', 0):,})"
        )
        self.stdout.write(f"    Misses:           {stats.get('misses', 0):,}")

        if lookups:
            # Built in two steps: nested same-quote f-strings only parse
            # on Python 3.12+, and this package supports 3.9.
            avoided = self.style.SUCCESS(format(stats.get("hits", 0), ","))
            self.stdout.write(f"    Milvus queries avoided: {avoided}")

        self.stdout.write("\n  Latency")
        for label, field in (
            ("p50", "latency_p50_ms"),
            ("p95", "latency_p95_ms"),
            ("p99", "latency_p99_ms"),
        ):
            value = stats.get(field)
            shown = "n/a" if value is None else f"{value} ms"
            self.stdout.write(f"    {label}:              {shown}")

        backend = stats.get("backend") or {}
        self.stdout.write("\n  Memory")
        self.stdout.write(
            f"    Entries:          {backend.get('entries', 0):,}"
        )
        self.stdout.write(
            f"    Bytes used:       {self._bytes(backend.get('bytes', 0))}"
        )
        limit = backend.get("max_memory")
        if limit:
            self.stdout.write(f"    Limit:            {self._bytes(limit)}")
            utilization = backend.get("utilization")
            if utilization is not None:
                self.stdout.write(f"    Utilization:      {utilization:.1%}")
        self.stdout.write(
            f"    Evictions:        {stats.get('evictions', 0):,}"
        )
        self.stdout.write(
            f"    Rejected (too large): {stats.get('rejected', 0):,}"
        )

        policy = backend.get("policy") or {}
        window = backend.get("window") or {}
        if policy or window:
            self.stdout.write("\n  Policy")
            if policy:
                self.stdout.write(
                    f"    Algorithm:        {policy.get('algorithm', 'n/a')}"
                )
            if window:
                self.stdout.write(
                    f"    Admission ratio:  "
                    f"{window.get('admission_ratio', 'n/a')}"
                    f"{' (adaptive)' if window.get('adaptive') else ''}"
                )
                self.stdout.write(
                    f"    Window ticks:     {window.get('ticks', 0)} "
                    f"({window.get('adjustments', 0)} adjustments)"
                )

        semantic = stats.get("semantic") or {}
        if semantic.get("enabled"):
            self.stdout.write("\n  Semantic cache")
            self.stdout.write(
                f"    Threshold:        {semantic.get('threshold')} "
                f"({semantic.get('metric')})"
            )
            self.stdout.write(f"    Buckets:          {semantic.get('buckets', 0)}")
            self.stdout.write(f"    Query vectors:    {semantic.get('vectors', 0):,}")
            self.stdout.write(
                f"    Hits / misses:    {semantic.get('hits', 0):,} / "
                f"{semantic.get('misses', 0):,}"
            )
            self.stdout.write(f"    Reranks:          {semantic.get('reranks', 0):,}")
            self.stdout.write(
                f"    Index memory:     {self._bytes(semantic.get('bytes', 0))}"
            )

        vectors = stats.get("vectors") or {}
        if vectors.get("entries"):
            self.stdout.write("\n  Vector cache")
            self.stdout.write(f"    Embeddings:       {vectors.get('entries', 0):,}")
            self.stdout.write(
                f"    Memory:           {self._bytes(vectors.get('bytes', 0))}"
            )

        l2 = backend.get("l2")
        if l2:
            self.stdout.write("\n  Shared tier")
            self.stdout.write(f"    Backend:          {l2.get('backend')}")
            reachable = l2.get("reachable")
            if reachable is False:
                note = f" - {l2['note']}" if l2.get("note") else ""
                self.stdout.write(
                    self.style.ERROR(f"    Reachable:        no{note}")
                )
            else:
                self.stdout.write("    Reachable:        yes")
            self.stdout.write(
                f"    Promotions:       {backend.get('promotions', 0):,}"
            )
            breaker = backend.get("circuit_breaker") or {}
            if breaker:
                self.stdout.write(
                    f"    Circuit breaker:  {breaker.get('state')} "
                    f"({breaker.get('trips', 0)} trips)"
                )

        versions = stats.get("versions") or {}
        if versions.get("versions"):
            self.stdout.write("\n  Collection versions")
            for collection, version in sorted(versions["versions"].items()):
                self.stdout.write(f"    {collection}: v{version}")

        errors = stats.get("errors", 0)
        if errors:
            self.stdout.write(
                self.style.WARNING(
                    f"\n  {errors:,} backend errors (all failed open; queries "
                    f"fell through to Milvus)"
                )
            )
        self.stdout.write("")

    @staticmethod
    def _bytes(value):
        if not value:
            return "0 B"
        size = float(value)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
            size /= 1024
        return f"{size:.1f} TB"
