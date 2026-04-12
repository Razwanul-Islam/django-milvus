"""
Schema management for Milvus collections.

Provides functions to create, drop, and manage Milvus collections
based on MilvusModel definitions. Includes support for indexes,
partitions, aliases, and collection lifecycle management.
"""

from pymilvus import CollectionSchema, FieldSchema

from .connection import get_milvus_client
from .exceptions import CollectionError, SchemaError


def create_collection_for_model(model_class, drop_existing=False):
    """Create a Milvus collection based on a MilvusModel class.

    Args:
        model_class: MilvusModel subclass.
        drop_existing: If True, drop existing collection first.

    Returns:
        bool: True if collection was created.
    """
    client = model_class.get_client()
    collection_name = model_class.get_collection_name()
    options = model_class._options

    if client.has_collection(collection_name):
        if drop_existing:
            client.drop_collection(collection_name)
        else:
            return False

    # Build schema
    schema = model_class.get_schema()

    # Create collection kwargs
    create_kwargs = {
        "collection_name": collection_name,
        "schema": schema,
    }

    if options.shards_num is not None:
        create_kwargs["shards_num"] = options.shards_num

    if options.consistency_level:
        create_kwargs["consistency_level"] = options.consistency_level

    if options.num_partitions is not None:
        create_kwargs["num_partitions"] = options.num_partitions

    client.create_collection(**create_kwargs)

    # Create indexes
    _create_indexes_for_model(client, model_class)

    # Load collection if auto_load is enabled
    if options.auto_load:
        try:
            client.load_collection(collection_name)
        except Exception:
            pass  # Collection might auto-load after index creation

    return True


def _create_indexes_for_model(client, model_class):
    """Create indexes defined in MilvusIndexes."""
    collection_name = model_class.get_collection_name()
    options = model_class._options

    for index in options.indexes:
        index_dict = index.to_dict()
        try:
            index_params = {
                "collection_name": collection_name,
                "field_name": index_dict["field_name"],
                "index_params": {
                    "index_type": index_dict["index_type"],
                    "params": index_dict.get("params", {}),
                },
            }
            if index_dict.get("metric_type"):
                index_params["index_params"]["metric_type"] = index_dict["metric_type"]

            client.create_index(**index_params)
        except Exception as e:
            raise SchemaError(
                f"Failed to create index on {index_dict['field_name']}: {e}"
            ) from e


def drop_collection_for_model(model_class):
    """Drop the Milvus collection for a model.

    Args:
        model_class: MilvusModel subclass.

    Returns:
        bool: True if collection was dropped.
    """
    client = model_class.get_client()
    collection_name = model_class.get_collection_name()

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        return True
    return False


# ─────────────────────────────────────────────────────────
# Collection Management
# ─────────────────────────────────────────────────────────

def list_collections(alias="default"):
    """List all collections in the Milvus database."""
    client = get_milvus_client(alias)
    return client.list_collections()


def describe_collection(collection_name, alias="default"):
    """Describe a collection."""
    client = get_milvus_client(alias)
    return client.describe_collection(collection_name)


def has_collection(collection_name, alias="default"):
    """Check if a collection exists."""
    client = get_milvus_client(alias)
    return client.has_collection(collection_name)


def rename_collection(old_name, new_name, alias="default"):
    """Rename a collection."""
    client = get_milvus_client(alias)
    client.rename_collection(old_name, new_name)


def get_collection_stats(collection_name, alias="default"):
    """Get collection statistics."""
    client = get_milvus_client(alias)
    return client.get_collection_stats(collection_name)


def load_collection(collection_name, alias="default"):
    """Load a collection into memory."""
    client = get_milvus_client(alias)
    client.load_collection(collection_name)


def release_collection(collection_name, alias="default"):
    """Release a collection from memory."""
    client = get_milvus_client(alias)
    client.release_collection(collection_name)


def get_load_state(collection_name, alias="default"):
    """Get collection load state."""
    client = get_milvus_client(alias)
    return client.get_load_state(collection_name)


def drop_collection(collection_name, alias="default"):
    """Drop a collection."""
    client = get_milvus_client(alias)
    client.drop_collection(collection_name)


# ─────────────────────────────────────────────────────────
# Index Management
# ─────────────────────────────────────────────────────────

def create_index(collection_name, field_name, index_type="AUTOINDEX",
                 metric_type="COSINE", params=None, alias="default"):
    """Create an index on a collection field.

    Args:
        collection_name: Name of the collection.
        field_name: Name of the field to index.
        index_type: Index type (FLAT, IVF_FLAT, HNSW, etc.).
        metric_type: Distance metric (COSINE, L2, IP).
        params: Additional index parameters.
        alias: Database alias.
    """
    client = get_milvus_client(alias)
    index_params = {
        "index_type": index_type,
        "params": params or {},
    }
    if metric_type:
        index_params["metric_type"] = metric_type

    client.create_index(
        collection_name=collection_name,
        field_name=field_name,
        index_params=index_params,
    )


def list_indexes(collection_name, alias="default"):
    """List indexes on a collection."""
    client = get_milvus_client(alias)
    return client.list_indexes(collection_name)


def drop_index(collection_name, index_name, alias="default"):
    """Drop an index from a collection."""
    client = get_milvus_client(alias)
    client.drop_index(collection_name, index_name)


def describe_index(collection_name, index_name, alias="default"):
    """Describe an index."""
    client = get_milvus_client(alias)
    return client.describe_index(collection_name, index_name)


# ─────────────────────────────────────────────────────────
# Partition Management
# ─────────────────────────────────────────────────────────

def create_partition(collection_name, partition_name, alias="default"):
    """Create a partition in a collection."""
    client = get_milvus_client(alias)
    client.create_partition(collection_name, partition_name)


def drop_partition(collection_name, partition_name, alias="default"):
    """Drop a partition from a collection."""
    client = get_milvus_client(alias)
    client.drop_partition(collection_name, partition_name)


def has_partition(collection_name, partition_name, alias="default"):
    """Check if a partition exists."""
    client = get_milvus_client(alias)
    return client.has_partition(collection_name, partition_name)


def list_partitions(collection_name, alias="default"):
    """List partitions in a collection."""
    client = get_milvus_client(alias)
    return client.list_partitions(collection_name)


def load_partitions(collection_name, partition_names, alias="default"):
    """Load specific partitions into memory."""
    client = get_milvus_client(alias)
    client.load_partitions(collection_name, partition_names)


def release_partitions(collection_name, partition_names, alias="default"):
    """Release specific partitions from memory."""
    client = get_milvus_client(alias)
    client.release_partitions(collection_name, partition_names)


def get_partition_stats(collection_name, partition_name, alias="default"):
    """Get partition statistics."""
    client = get_milvus_client(alias)
    return client.get_partition_stats(collection_name, partition_name)


# ─────────────────────────────────────────────────────────
# Alias Management
# ─────────────────────────────────────────────────────────

def create_alias(collection_name, alias_name, db_alias="default"):
    """Create an alias for a collection."""
    client = get_milvus_client(db_alias)
    client.create_alias(collection_name, alias_name)


def drop_alias(alias_name, db_alias="default"):
    """Drop a collection alias."""
    client = get_milvus_client(db_alias)
    client.drop_alias(alias_name)


def alter_alias(collection_name, alias_name, db_alias="default"):
    """Alter (reassign) a collection alias."""
    client = get_milvus_client(db_alias)
    client.alter_alias(collection_name, alias_name)


def describe_alias(alias_name, db_alias="default"):
    """Describe a collection alias."""
    client = get_milvus_client(db_alias)
    return client.describe_alias(alias_name)


def list_aliases(collection_name=None, db_alias="default"):
    """List collection aliases."""
    client = get_milvus_client(db_alias)
    if collection_name:
        return client.list_aliases(collection_name)
    return client.list_aliases()


# ─────────────────────────────────────────────────────────
# User and RBAC Management
# ─────────────────────────────────────────────────────────

def create_user(username, password, alias="default"):
    """Create a Milvus user."""
    client = get_milvus_client(alias)
    client.create_user(username, password)


def drop_user(username, alias="default"):
    """Drop a Milvus user."""
    client = get_milvus_client(alias)
    client.drop_user(username)


def update_password(username, old_password, new_password, alias="default"):
    """Update a user's password."""
    client = get_milvus_client(alias)
    client.update_password(username, old_password, new_password)


def list_users(alias="default"):
    """List all users."""
    client = get_milvus_client(alias)
    return client.list_users()


def describe_user(username, alias="default"):
    """Describe a user (get roles)."""
    client = get_milvus_client(alias)
    return client.describe_user(username)


def create_role(role_name, alias="default"):
    """Create a role."""
    client = get_milvus_client(alias)
    client.create_role(role_name)


def drop_role(role_name, alias="default"):
    """Drop a role."""
    client = get_milvus_client(alias)
    client.drop_role(role_name)


def describe_role(role_name, alias="default"):
    """Describe a role (get privileges)."""
    client = get_milvus_client(alias)
    return client.describe_role(role_name)


def list_roles(alias="default"):
    """List all roles."""
    client = get_milvus_client(alias)
    return client.list_roles()


def grant_role(username, role_name, alias="default"):
    """Grant a role to a user."""
    client = get_milvus_client(alias)
    client.grant_role(username, role_name)


def revoke_role(username, role_name, alias="default"):
    """Revoke a role from a user."""
    client = get_milvus_client(alias)
    client.revoke_role(username, role_name)


def grant_privilege(role_name, object_type, object_name, privilege,
                    alias="default"):
    """Grant a privilege to a role.

    Args:
        role_name: Name of the role.
        object_type: Type of object ("Collection", "Global", "User").
        object_name: Name of the object.
        privilege: Privilege to grant.
        alias: Database alias.
    """
    client = get_milvus_client(alias)
    client.grant_privilege(role_name, object_type, privilege, object_name)


def revoke_privilege(role_name, object_type, object_name, privilege,
                     alias="default"):
    """Revoke a privilege from a role."""
    client = get_milvus_client(alias)
    client.revoke_privilege(role_name, object_type, privilege, object_name)
