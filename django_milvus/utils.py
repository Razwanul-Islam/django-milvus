"""Utility functions for django-milvus."""

from pymilvus import DataType


PYTHON_TO_MILVUS_TYPE = {
    bool: DataType.BOOL,
    int: DataType.INT64,
    float: DataType.DOUBLE,
    str: DataType.VARCHAR,
    bytes: DataType.BINARY_VECTOR,
    list: DataType.FLOAT_VECTOR,
    dict: DataType.JSON,
}


def python_value_to_milvus(value):
    """Convert a Python value to a Milvus-compatible value."""
    if value is None:
        return None
    return value


def milvus_value_to_python(value, field_type=None):
    """Convert a Milvus value to a Python value."""
    if value is None:
        return None
    return value


def build_filter_expr(filters):
    """Build a Milvus filter expression from a dict of filter conditions.

    Supports Django-style lookups:
        field=value         -> field == value
        field__eq=value     -> field == value
        field__ne=value     -> field != value
        field__gt=value     -> field > value
        field__gte=value    -> field >= value
        field__lt=value     -> field < value
        field__lte=value    -> field <= value
        field__in=list      -> field in [values]
        field__nin=list     -> field not in [values]
        field__like=str     -> field like "str"
        field__exists=bool  -> exists(field) or not exists(field)

    Args:
        filters: dict of filter conditions

    Returns:
        str: Milvus filter expression
    """
    if not filters:
        return ""

    expressions = []
    for key, value in filters.items():
        parts = key.split("__")
        if len(parts) == 1:
            field = parts[0]
            op = "eq"
        else:
            field = "__".join(parts[:-1])
            op = parts[-1]

            # Check if the last part is actually a known operator
            known_ops = {
                "eq", "ne", "gt", "gte", "lt", "lte",
                "in", "nin", "like", "exists",
                "json_contains", "json_contains_all", "json_contains_any",
                "array_contains", "array_contains_all", "array_contains_any",
                "array_length",
            }
            if op not in known_ops:
                # Not a known operator, treat whole key as field name
                field = key
                op = "eq"

        expr = _build_single_expr(field, op, value)
        expressions.append(expr)

    return " and ".join(expressions)


def _build_single_expr(field, op, value):
    """Build a single filter expression."""
    formatted_value = _format_value(value)

    if op == "eq":
        return f"{field} == {formatted_value}"
    elif op == "ne":
        return f"{field} != {formatted_value}"
    elif op == "gt":
        return f"{field} > {formatted_value}"
    elif op == "gte":
        return f"{field} >= {formatted_value}"
    elif op == "lt":
        return f"{field} < {formatted_value}"
    elif op == "lte":
        return f"{field} <= {formatted_value}"
    elif op == "in":
        if not isinstance(value, (list, tuple, set)):
            raise ValueError(f"'in' lookup requires a list, got {type(value)}")
        items = ", ".join(_format_value(v) for v in value)
        return f"{field} in [{items}]"
    elif op == "nin":
        if not isinstance(value, (list, tuple, set)):
            raise ValueError(f"'nin' lookup requires a list, got {type(value)}")
        items = ", ".join(_format_value(v) for v in value)
        return f"{field} not in [{items}]"
    elif op == "like":
        return f'{field} like "{value}"'
    elif op == "exists":
        if value:
            return f"exists {field}"
        else:
            return f"not exists {field}"
    elif op == "json_contains":
        return f"json_contains({field}, {formatted_value})"
    elif op == "json_contains_all":
        items = ", ".join(_format_value(v) for v in value)
        return f"json_contains_all({field}, [{items}])"
    elif op == "json_contains_any":
        items = ", ".join(_format_value(v) for v in value)
        return f"json_contains_any({field}, [{items}])"
    elif op == "array_contains":
        return f"array_contains({field}, {formatted_value})"
    elif op == "array_contains_all":
        items = ", ".join(_format_value(v) for v in value)
        return f"array_contains_all({field}, [{items}])"
    elif op == "array_contains_any":
        items = ", ".join(_format_value(v) for v in value)
        return f"array_contains_any({field}, [{items}])"
    elif op == "array_length":
        return f"array_length({field}) == {formatted_value}"
    else:
        raise ValueError(f"Unknown filter operator: {op}")


def _format_value(value):
    """Format a Python value for use in a Milvus filter expression."""
    if isinstance(value, str):
        # Escape double quotes in strings
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif value is None:
        return "null"
    else:
        return str(value)


def get_collection_name(model_class):
    """Get the Milvus collection name for a model class."""
    meta = getattr(model_class, "MilvusMeta", None)
    if meta and hasattr(meta, "collection_name"):
        return meta.collection_name
    return model_class.__name__.lower()


def get_database_alias(model_class):
    """Get the database alias for a model class."""
    meta = getattr(model_class, "MilvusMeta", None)
    if meta and hasattr(meta, "database"):
        return meta.database
    return "default"
