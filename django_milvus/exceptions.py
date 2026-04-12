"""Custom exceptions for django-milvus."""


class DjangoMilvusError(Exception):
    """Base exception for django-milvus."""
    pass


class ConnectionError(DjangoMilvusError):
    """Raised when connection to Milvus fails."""
    pass


class CollectionError(DjangoMilvusError):
    """Raised for collection-related errors."""
    pass


class SchemaError(DjangoMilvusError):
    """Raised for schema-related errors."""
    pass


class FieldError(DjangoMilvusError):
    """Raised for field-related errors."""
    pass


class ValidationError(DjangoMilvusError):
    """Raised for validation errors."""
    pass


class IndexError(DjangoMilvusError):
    """Raised for index-related errors."""
    pass


class PartitionError(DjangoMilvusError):
    """Raised for partition-related errors."""
    pass


class SearchError(DjangoMilvusError):
    """Raised for search-related errors."""
    pass


class ObjectDoesNotExist(DjangoMilvusError):
    """Raised when a queried object does not exist."""
    pass


class MultipleObjectsReturned(DjangoMilvusError):
    """Raised when multiple objects are returned but one expected."""
    pass
