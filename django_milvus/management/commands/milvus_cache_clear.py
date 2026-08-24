"""
Management command to clear the django-milvus cache.

Usage:
    python manage.py milvus_cache_clear --collection documents
    python manage.py milvus_cache_clear --all
    python manage.py milvus_cache_clear --all --yes
    python manage.py milvus_cache_clear --all --alias analytics

Note that with an in-process L1 this only clears *this* process. Other
workers keep their own copies until their entries expire, unless a shared
L2 is configured - in which case the version bump reaches all of them.
"""

from django.core.management.base import BaseCommand, CommandError

from django_milvus.cache import caches, get_cache, is_configured


class Command(BaseCommand):
    help = "Clear cached django-milvus query results"

    def add_arguments(self, parser):
        parser.add_argument(
            "--collection",
            default=None,
            help="Clear only this collection's cached entries",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            default=False,
            dest="clear_all",
            help="Clear every cached entry",
        )
        parser.add_argument(
            "--alias",
            default="default",
            help="Cache alias to clear (default: 'default')",
        )
        parser.add_argument(
            "--yes",
            action="store_true",
            default=False,
            help="Skip the confirmation prompt",
        )

    def handle(self, *args, **options):
        if not is_configured():
            self.stdout.write(
                self.style.WARNING("No MILVUS_CACHE configured; nothing to clear.")
            )
            return

        collection = options.get("collection")
        clear_all = options["clear_all"]
        alias = options["alias"]

        if not collection and not clear_all:
            raise CommandError(
                "Specify --collection <name> or --all. Refusing to guess "
                "which entries you meant."
            )
        if collection and clear_all:
            raise CommandError("Use either --collection or --all, not both.")

        cache = get_cache(alias)
        if cache is None:
            raise CommandError(
                f"No usable cache for alias {alias!r}. "
                f"Known aliases: {sorted(caches.all()) or 'none built yet'}."
            )

        target = f"collection '{collection}'" if collection else "ALL collections"
        if not options["yes"]:
            self.stdout.write(
                self.style.WARNING(
                    f"About to clear cached entries for {target} "
                    f"(alias '{alias}')."
                )
            )
            answer = input("Continue? [y/N]: ")
            if answer.strip().lower() not in ("y", "yes"):
                self.stdout.write("Aborted.")
                return

        removed = cache.clear(collection)

        self.stdout.write(
            self.style.SUCCESS(
                f"Cleared {removed:,} cached entries for {target} "
                f"(alias '{alias}')."
            )
        )
        if collection:
            self.stdout.write(
                f"Collection version is now v{cache.version(collection)}."
            )
        if cache.backend.shared:
            self.stdout.write(
                "A shared tier is configured, so other workers are "
                "invalidated too."
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    "No shared tier configured: this cleared only the current "
                    "process. Other workers keep their entries until those "
                    "expire."
                )
            )
