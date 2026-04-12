"""
Interactive Milvus shell management command.

Usage:
    python manage.py milvus_shell
    python manage.py milvus_shell --database milvus
"""

from django.core.management.base import BaseCommand

from django_milvus.connection import get_milvus_client


class Command(BaseCommand):
    help = "Launch an interactive Milvus shell with a connected client"

    def add_arguments(self, parser):
        parser.add_argument(
            "--database",
            default="default",
            help="Database alias (default: 'default')",
        )

    def handle(self, *args, **options):
        alias = options["database"]

        try:
            client = get_milvus_client(alias)
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Cannot connect to Milvus: {e}")
            )
            return

        self.stdout.write(
            self.style.SUCCESS(
                f"Connected to Milvus (alias='{alias}'). "
                f"Client available as 'client'."
            )
        )
        self.stdout.write("Type 'exit()' or Ctrl+D to quit.\n")
        self.stdout.write("Quick commands:")
        self.stdout.write("  client.list_collections()")
        self.stdout.write("  client.describe_collection('name')")
        self.stdout.write("  client.get_collection_stats('name')")
        self.stdout.write("")

        # Import useful things into the shell namespace
        import django_milvus
        from django_milvus import schema, fields, indexes

        namespace = {
            "client": client,
            "django_milvus": django_milvus,
            "schema": schema,
            "fields": fields,
            "indexes": indexes,
        }

        try:
            from IPython import embed
            embed(user_ns=namespace)
        except ImportError:
            import code
            code.interact(local=namespace)
