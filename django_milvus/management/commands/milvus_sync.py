"""
Management command to create Milvus collections for all registered MilvusModels.

Usage:
    python manage.py milvus_sync
    python manage.py milvus_sync --drop-existing
    python manage.py milvus_sync --models myapp.Document myapp.Product
"""

from django.core.management.base import BaseCommand, CommandError

from django_milvus.models import MilvusModel


class Command(BaseCommand):
    help = "Create Milvus collections for registered MilvusModel classes"

    def add_arguments(self, parser):
        parser.add_argument(
            "--drop-existing",
            action="store_true",
            default=False,
            help="Drop existing collections before creating them",
        )
        parser.add_argument(
            "--models",
            nargs="*",
            help="Specific model classes to sync (e.g., myapp.Document)",
        )
        parser.add_argument(
            "--database",
            default=None,
            help="Database alias to use",
        )

    def handle(self, *args, **options):
        drop_existing = options["drop_existing"]
        model_names = options.get("models")

        # Discover all MilvusModel subclasses
        models = self._discover_models(model_names)

        if not models:
            self.stdout.write(
                self.style.WARNING("No MilvusModel classes found.")
            )
            return

        self.stdout.write(f"Found {len(models)} MilvusModel class(es).")

        for model_class in models:
            collection_name = model_class.get_collection_name()
            try:
                created = model_class.create_collection(
                    drop_existing=drop_existing
                )
                if created:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"  Created collection: {collection_name}"
                        )
                    )
                else:
                    self.stdout.write(
                        f"  Collection already exists: {collection_name}"
                    )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(
                        f"  Error creating {collection_name}: {e}"
                    )
                )

    def _discover_models(self, model_names=None):
        """Discover MilvusModel subclasses from installed Django apps."""
        from django.apps import apps

        models = []

        if model_names:
            # Import specific models
            for name in model_names:
                try:
                    parts = name.rsplit(".", 1)
                    if len(parts) == 2:
                        module_path, class_name = parts
                        import importlib
                        module = importlib.import_module(module_path)
                        model_class = getattr(module, class_name)
                        if (
                            isinstance(model_class, type)
                            and issubclass(model_class, MilvusModel)
                            and model_class is not MilvusModel
                        ):
                            models.append(model_class)
                        else:
                            raise CommandError(
                                f"{name} is not a MilvusModel subclass"
                            )
                    else:
                        raise CommandError(
                            f"Invalid model name: {name}. Use 'module.ClassName'"
                        )
                except (ImportError, AttributeError) as e:
                    raise CommandError(f"Cannot import model '{name}': {e}")
        else:
            # Auto-discover from all installed apps
            for app_config in apps.get_app_configs():
                try:
                    import importlib
                    models_module = importlib.import_module(
                        f"{app_config.name}.models"
                    )
                    for attr_name in dir(models_module):
                        attr = getattr(models_module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, MilvusModel)
                            and attr is not MilvusModel
                            and attr.__module__ == models_module.__name__
                        ):
                            models.append(attr)
                except (ImportError, AttributeError):
                    continue

        return models
