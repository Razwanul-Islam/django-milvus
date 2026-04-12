from django.apps import AppConfig


class DjangoMilvusConfig(AppConfig):
    name = "django_milvus"
    verbose_name = "Django Milvus"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        pass
