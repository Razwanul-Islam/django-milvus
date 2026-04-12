"""
Management command to drop Milvus collections.

Usage:
    python manage.py milvus_drop --collection documents
    python manage.py milvus_drop --all
"""

from django.core.management.base import BaseCommand

from django_milvus.connection import get_milvus_client


class Command(BaseCommand):
    help = "Drop Milvus collections"

    def add_arguments(self, parser):
        parser.add_argument(
            "--collection",
            default=None,
            help="Name of the collection to drop",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            default=False,
            help="Drop ALL collections (use with caution)",
        )
        parser.add_argument(
            "--database",
            default="default",
            help="Database alias (default: 'default')",
        )
        parser.add_argument(
            "--yes",
            action="store_true",
            default=False,
            help="Skip confirmation prompt",
        )

    def handle(self, *args, **options):
        alias = options["database"]
        collection_name = options.get("collection")
        drop_all = options["all"]
        skip_confirm = options["yes"]

        try:
            client = get_milvus_client(alias)
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Cannot connect to Milvus: {e}")
            )
            return

        if collection_name:
            if not skip_confirm:
                confirm = input(
                    f"Drop collection '{collection_name}'? [y/N] "
                )
                if confirm.lower() != "y":
                    self.stdout.write("Aborted.")
                    return

            try:
                client.drop_collection(collection_name)
                self.stdout.write(
                    self.style.SUCCESS(f"Dropped collection: {collection_name}")
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error dropping '{collection_name}': {e}")
                )

        elif drop_all:
            collections = client.list_collections()
            if not collections:
                self.stdout.write("No collections to drop.")
                return

            if not skip_confirm:
                self.stdout.write(
                    f"Collections to drop: {', '.join(collections)}"
                )
                confirm = input("Drop ALL collections? [y/N] ")
                if confirm.lower() != "y":
                    self.stdout.write("Aborted.")
                    return

            for name in collections:
                try:
                    client.drop_collection(name)
                    self.stdout.write(
                        self.style.SUCCESS(f"  Dropped: {name}")
                    )
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f"  Error dropping {name}: {e}")
                    )
        else:
            self.stdout.write(
                self.style.ERROR(
                    "Specify --collection <name> or --all"
                )
            )
