import logging

from django.apps import AppConfig
from django.core.checks import Error, register

logger = logging.getLogger("django_milvus.cache")


class DjangoMilvusConfig(AppConfig):
    name = "django_milvus"
    verbose_name = "Django Milvus"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        register(check_cache_settings)


def check_cache_settings(app_configs, **kwargs):
    """Validate ``MILVUS_CACHE`` at start-up.

    Registered as a Django system check so a typo in the cache settings
    surfaces on ``runserver``/``check``/``migrate`` with a readable
    message, instead of raising from inside whichever view happens to run
    the first cached query.

    Reported as errors rather than raised, so ``manage.py check`` can list
    every problem at once.
    """
    from .cache.config import is_configured, validate_all
    from .exceptions import CacheConfigurationError

    if not is_configured():
        return []

    try:
        validate_all()
    except CacheConfigurationError as exc:
        return [
            Error(
                str(exc),
                hint=(
                    "See the Caching section of the django-milvus README "
                    "for the full MILVUS_CACHE reference."
                ),
                id="django_milvus.E001",
            )
        ]
    except Exception as exc:  # pragma: no cover - defensive
        return [
            Error(
                f"MILVUS_CACHE could not be loaded: {exc}",
                id="django_milvus.E002",
            )
        ]
    return []
