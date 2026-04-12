"""
Management command to show Milvus collection statistics.

Usage:
    python manage.py milvus_stats
    python manage.py milvus_stats --collection documents
    python manage.py milvus_stats --database milvus
"""

from django.core.management.base import BaseCommand

from django_milvus.connection import get_milvus_client


class Command(BaseCommand):
    help = "Show Milvus collection statistics and information"

    def add_arguments(self, parser):
        parser.add_argument(
            "--collection",
            default=None,
            help="Specific collection to inspect",
        )
        parser.add_argument(
            "--database",
            default="default",
            help="Database alias to use (default: 'default')",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Show detailed field and index information",
        )

    def handle(self, *args, **options):
        alias = options["database"]
        collection_name = options.get("collection")
        verbose = options["verbose"]

        try:
            client = get_milvus_client(alias)
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Cannot connect to Milvus: {e}")
            )
            return

        if collection_name:
            self._show_collection_stats(client, collection_name, verbose)
        else:
            self._show_all_collections(client, verbose)

    def _show_all_collections(self, client, verbose):
        collections = client.list_collections()

        if not collections:
            self.stdout.write("No collections found.")
            return

        self.stdout.write(
            self.style.SUCCESS(f"\nFound {len(collections)} collection(s):\n")
        )

        for name in sorted(collections):
            self._show_collection_stats(client, name, verbose)
            self.stdout.write("")

    def _show_collection_stats(self, client, name, verbose):
        try:
            desc = client.describe_collection(name)
            stats = client.get_collection_stats(name)
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Error inspecting '{name}': {e}")
            )
            return

        row_count = stats.get("row_count", "N/A")
        self.stdout.write(f"  Collection: {self.style.SUCCESS(name)}")
        self.stdout.write(f"    Rows: {row_count}")
        self.stdout.write(
            f"    Description: {desc.get('description', 'N/A')}"
        )
        self.stdout.write(
            f"    Consistency: {desc.get('consistency_level', 'N/A')}"
        )

        if verbose:
            fields = desc.get("fields", [])
            self.stdout.write(f"    Fields ({len(fields)}):")
            for f in fields:
                pk = " [PK]" if f.get("is_primary") else ""
                auto = " [AUTO]" if f.get("auto_id") else ""
                params = f.get("params", {})
                dim = f" dim={params['dim']}" if "dim" in params else ""
                max_len = (
                    f" max_length={params['max_length']}"
                    if "max_length" in params else ""
                )
                self.stdout.write(
                    f"      - {f['name']}: {f.get('type', 'N/A')}"
                    f"{pk}{auto}{dim}{max_len}"
                )

            try:
                indexes = client.list_indexes(name)
                if indexes:
                    self.stdout.write(f"    Indexes ({len(indexes)}):")
                    for idx_name in indexes:
                        try:
                            idx_desc = client.describe_index(name, idx_name)
                            self.stdout.write(
                                f"      - {idx_name}: {idx_desc}"
                            )
                        except Exception:
                            self.stdout.write(f"      - {idx_name}")
            except Exception:
                pass
