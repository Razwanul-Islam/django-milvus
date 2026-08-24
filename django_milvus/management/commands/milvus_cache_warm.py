"""
Management command to warm the django-milvus cache.

Usage:
    python manage.py milvus_cache_warm --model myapp.models.Document \\
        --file vectors.json
    python manage.py milvus_cache_warm --model myapp.models.Document \\
        --file vectors.json --limit 20 --ttl 3600

The file holds query vectors as JSON: either a bare list of vectors, or
an object with a "vectors" key and optional "limit" / "ttl".

Run it after a deploy so the first real users do not each pay for a cold
cache. Remember that an in-process L1 is per worker: warming from a
management command fills a *separate* process, so it only helps if a
shared L2 is configured. Without one, warm from inside each worker (for
example in ``AppConfig.ready``).
"""

import json

from django.core.management.base import BaseCommand, CommandError
from django.utils.module_loading import import_string

from django_milvus.cache import is_configured


class Command(BaseCommand):
    help = "Pre-populate the django-milvus cache with known query vectors"

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            required=True,
            help="Dotted path to the MilvusModel, e.g. myapp.models.Document",
        )
        parser.add_argument(
            "--file",
            default=None,
            help="JSON file of query vectors to warm",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=10,
            help="Result limit for each warmed search (default: 10)",
        )
        parser.add_argument(
            "--ttl",
            type=float,
            default=None,
            help="TTL for warmed entries (default: the alias TTL)",
        )
        parser.add_argument(
            "--vector-field",
            default=None,
            help="Vector field to search (auto-detected when unambiguous)",
        )
        parser.add_argument(
            "--alias",
            default=None,
            help="Cache alias to warm",
        )

    def handle(self, *args, **options):
        if not is_configured():
            self.stdout.write(
                self.style.WARNING(
                    "No MILVUS_CACHE configured; there is nothing to warm."
                )
            )
            return

        try:
            model = import_string(options["model"])
        except ImportError as exc:
            raise CommandError(
                f"Could not import model {options['model']!r}: {exc}"
            ) from exc

        vectors, file_limit, file_ttl = self._load(options.get("file"))
        if not vectors:
            raise CommandError(
                "No query vectors to warm. Pass --file with a JSON list of "
                "vectors."
            )

        limit = options["limit"] or file_limit or 10
        ttl = options["ttl"] if options["ttl"] is not None else file_ttl

        cache_kwargs = {}
        if ttl is not None:
            cache_kwargs["ttl"] = ttl
        if options.get("alias"):
            cache_kwargs["alias"] = options["alias"]

        self.stdout.write(
            f"Warming {len(vectors)} query vector(s) for "
            f"{model.__name__} (limit={limit})..."
        )

        from django_milvus.cache.warmup import warm

        result = warm(
            model,
            vectors=vectors,
            limit=limit,
            vector_field=options.get("vector_field"),
            **cache_kwargs,
        )

        self.stdout.write(
            self.style.SUCCESS(f"Warmed {result.warmed} cache entries.")
        )
        if result.skipped:
            self.stdout.write(f"Skipped {result.skipped}.")
        if result.errors:
            self.stdout.write(
                self.style.WARNING(f"{len(result.errors)} failed:")
            )
            for target, message in result.errors[:10]:
                self.stdout.write(f"  {target}: {message}")
            if len(result.errors) > 10:
                self.stdout.write(f"  ... and {len(result.errors) - 10} more")

    def _load(self, path):
        """Read vectors from a JSON file, tolerating both layouts."""
        if not path:
            return [], None, None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise CommandError(f"Could not read {path!r}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CommandError(f"{path!r} is not valid JSON: {exc}") from exc

        if isinstance(payload, list):
            return payload, None, None
        if isinstance(payload, dict):
            return (
                payload.get("vectors") or [],
                payload.get("limit"),
                payload.get("ttl"),
            )
        raise CommandError(
            f"{path!r} must contain a list of vectors or an object with a "
            f"'vectors' key."
        )
