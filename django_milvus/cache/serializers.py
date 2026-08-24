"""
Payload serialization and compression for shared cache tiers.

L1 stores live Python objects, so this module only matters for L2
(Redis and any other backend that needs bytes on the wire).

Cached payloads are plain ``list[dict]`` - the raw Milvus wire shape -
never model instances, so ``json`` and ``msgpack`` are viable choices
alongside ``pickle``.

The framing byte prepended to every payload records which compressor was
used, so a config change does not invalidate data written earlier:

    b"\\x00" + data   uncompressed
    b"\\x01" + data   zlib
    b"\\x02" + data   lz4
"""

import json
import pickle
import zlib

from ..exceptions import CacheConfigurationError

RAW = 0x00
ZLIB = 0x01
LZ4 = 0x02


# ─────────────────────────────────────────────────────────
# Serializers
# ─────────────────────────────────────────────────────────

class BaseSerializer:
    """Turn cache payloads into bytes and back."""

    name = "base"

    def dumps(self, value):
        raise NotImplementedError

    def loads(self, data):
        raise NotImplementedError


class PickleSerializer(BaseSerializer):
    """Fastest and most permissive; the default.

    Only ever reads data this application wrote, so the usual pickle
    caveats about untrusted input do not apply - but a shared Redis must
    not be writable by untrusted parties.
    """

    name = "pickle"

    def __init__(self, protocol=pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol

    def dumps(self, value):
        return pickle.dumps(value, protocol=self.protocol)

    def loads(self, data):
        return pickle.loads(data)


class JSONSerializer(BaseSerializer):
    """Portable and inspectable, at the cost of size and float precision."""

    name = "json"

    def dumps(self, value):
        return json.dumps(value, separators=(",", ":")).encode("utf-8")

    def loads(self, data):
        return json.loads(data.decode("utf-8"))


class MsgPackSerializer(BaseSerializer):
    """Compact binary encoding; needs the optional ``msgpack`` package."""

    name = "msgpack"

    def __init__(self):
        try:
            import msgpack
        except ImportError as exc:  # pragma: no cover - import guard
            raise CacheConfigurationError(
                "SERIALIZER='msgpack' requires the msgpack package. "
                "Install it with: pip install django-milvus[fast]"
            ) from exc
        self._msgpack = msgpack

    def dumps(self, value):
        return self._msgpack.packb(value, use_bin_type=True)

    def loads(self, data):
        return self._msgpack.unpackb(data, raw=False, strict_map_key=False)


SERIALIZERS = {
    "pickle": PickleSerializer,
    "json": JSONSerializer,
    "msgpack": MsgPackSerializer,
}


def get_serializer(name):
    """Instantiate a serializer by config name."""
    try:
        return SERIALIZERS[name]()
    except KeyError as exc:
        raise CacheConfigurationError(
            f"Unknown serializer {name!r}. Choose one of {sorted(SERIALIZERS)}."
        ) from exc


# ─────────────────────────────────────────────────────────
# Compression
# ─────────────────────────────────────────────────────────

class Compressor:
    """Optionally compress payloads above a size threshold.

    Small payloads skip compression entirely: below roughly 2 KB the CPU
    cost outweighs the bytes saved on a local network.
    """

    def __init__(self, algorithm="none", min_bytes=2048, level=1):
        self.algorithm = algorithm
        self.min_bytes = min_bytes or 0
        self.level = level
        self._lz4 = None

        if algorithm == "lz4":
            try:
                import lz4.frame
            except ImportError as exc:  # pragma: no cover - import guard
                raise CacheConfigurationError(
                    "COMPRESS.algorithm='lz4' requires the lz4 package. "
                    "Install it with: pip install django-milvus[fast]"
                ) from exc
            self._lz4 = lz4.frame
        elif algorithm not in ("none", "zlib"):
            raise CacheConfigurationError(
                f"Unknown compression algorithm {algorithm!r}. "
                f"Choose one of: none, zlib, lz4."
            )

    def compress(self, data):
        if self.algorithm == "none" or len(data) < self.min_bytes:
            return bytes([RAW]) + data
        if self.algorithm == "zlib":
            return bytes([ZLIB]) + zlib.compress(data, self.level)
        return bytes([LZ4]) + self._lz4.compress(data)

    def decompress(self, data):
        if not data:
            return data
        marker, body = data[0], data[1:]
        if marker == RAW:
            return body
        if marker == ZLIB:
            return zlib.decompress(body)
        if marker == LZ4:
            if self._lz4 is None:
                try:
                    import lz4.frame
                except ImportError as exc:  # pragma: no cover - import guard
                    raise CacheConfigurationError(
                        "Cached entry was lz4-compressed but the lz4 package "
                        "is not installed."
                    ) from exc
                self._lz4 = lz4.frame
            return self._lz4.decompress(body)
        raise ValueError(f"Unknown compression marker: {marker}")


class Codec:
    """Serializer plus compressor, as used by a shared backend."""

    def __init__(self, serializer, compressor=None):
        self.serializer = serializer
        self.compressor = compressor or Compressor()

    @classmethod
    def from_config(cls, l2_config):
        return cls(
            get_serializer(l2_config.serializer),
            Compressor(
                algorithm=l2_config.compress_algorithm,
                min_bytes=l2_config.compress_min_bytes,
                level=l2_config.compress_level,
            ),
        )

    def encode(self, value):
        return self.compressor.compress(self.serializer.dumps(value))

    def decode(self, data):
        return self.serializer.loads(self.compressor.decompress(data))
