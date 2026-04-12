"""Django settings for tests."""

SECRET_KEY = "test-secret-key-do-not-use-in-production"

INSTALLED_APPS = [
    "django_milvus",
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    },
    "milvus": {
        "ENGINE": "django_milvus.backend",
        "HOST": "localhost",
        "PORT": 19530,
        "NAME": "default",
    },
}

MILVUS = {
    "default": {
        "URI": "http://localhost:19530",
        "TOKEN": "",
        "DB_NAME": "default",
    },
}

DATABASE_ROUTERS = ["django_milvus.routers.MilvusRouter"]
