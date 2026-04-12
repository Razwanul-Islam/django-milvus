"""Database features declaration for Milvus backend."""

from django.db.backends.base.features import BaseDatabaseFeatures


class DatabaseFeatures(BaseDatabaseFeatures):
    """Declares what SQL features Milvus supports (very few)."""

    # Milvus is a vector database, not a relational database
    supports_transactions = False
    supports_savepoints = False
    supports_select_for_update = False
    supports_select_for_update_with_limit = False
    supports_select_related = False
    supports_sequence_reset = False
    supports_aggregation = False
    supports_aggregate_filter_clause = False
    supports_column_check_constraints = False
    supports_table_check_constraints = False
    supports_expression_indexes = False
    supports_foreign_keys = False
    supports_index_on_text_field = False
    supports_paramstyle_pyformat = False
    supports_json_field = True  # Milvus has JSON field support
    supports_order_by_nulls_modifier = False
    supports_partial_indexes = False
    supports_regex_backreferencing = False
    supports_subqueries_in_group_by = False
    supports_temporal_subtraction = False
    supports_timezones = False
    supports_unspecified_pk = False
    uses_savepoints = False
    can_create_inline_fk = False
    can_distinct_on_fields = False
    can_introspect_json_field = True
    can_rollback_ddl = False
    has_bulk_insert = True
    has_native_json_field = True
    has_native_uuid_field = False
    has_real_datatype = True

    # Prevent Django from running SQL tests against Milvus
    supports_explaining_query_execution = False
