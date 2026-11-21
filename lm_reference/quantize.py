"""Export the pinned TinyStories checkpoint as integer weights and lookup tables."""

import argparse
from array import array
from hashlib import sha256
import math
from pathlib import Path
import struct
import sys

from lm_reference.reference import MODEL_HASH, QUANTIZED_MODEL_HASH, read_pinned

CONFIG = (64, 172, 5, 8, 4, 512, 512)
MAGIC = b"PPQ16V1\0"
Q = 1 << 16
WEIGHT_SCALE = 1 << 24
EXP_STEPS = 64
EXP_MAX = 16


def nearest(value):
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def little_endian(typecode, values):
    data = array(typecode, values)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def export(model: Path) -> bytes:
    data = read_pinned(model, MODEL_HASH)
    config = struct.unpack_from("<7i", data)
    if config != CONFIG:
        raise ValueError(f"Unsupported TinyStories model shape {config}.")
    dim, hidden, layers, heads, kv_heads, vocab, seq = config
    head = dim // heads
    kv_dim = kv_heads * head
    weights = iter(struct.unpack_from(f"<{(len(data) - 28) // 4}f", data, 28))

    def take(count):
        return [next(weights) for _ in range(count)]

    def fixed(values):
        return little_endian("i", (nearest(value * Q) for value in values))

    def matrix(rows, columns):
        encoded = bytearray()
        scales = []
        for _ in range(rows):
            values = take(columns)
            peak = max(map(abs, values))
            scale = peak / 127 if peak else 1
            scales.append(max(1, nearest(scale * WEIGHT_SCALE)))
            encoded.extend(nearest(value / scale) & 255 for value in values)
        return little_endian("i", scales) + encoded

    embedding = take(vocab * dim)
    parts = [
        MAGIC,
        fixed(embedding),
        fixed(take(layers * dim)),
        matrix(layers * dim, dim),
        matrix(layers * kv_dim, dim),
        matrix(layers * kv_dim, dim),
        matrix(layers * dim, dim),
        fixed(take(layers * dim)),
        matrix(layers * hidden, dim),
        matrix(layers * dim, hidden),
        matrix(layers * hidden, dim),
        fixed(take(dim)),
    ]
    # The pinned runtime uses the formula below rather than the historical RoPE data.
    take(seq * head)
    if next(weights, None) is not None:
        raise ValueError("Unexpected weights after the tied checkpoint tensors.")
    # Match the original classifier encoder's two binary32 divisions.
    scale = f32(max(map(abs, embedding)) / 127)
    parts.append(bytes(nearest(f32(value / scale)) & 255 for value in embedding))
    parts.append(fixed(value for position in range(seq) for i in range(0, head, 2)
                       for value in (math.cos(position * 10000 ** (-i / head)),
                                     math.sin(position * 10000 ** (-i / head)))))
    parts.append(fixed(math.exp(-index / EXP_STEPS) for index in range(EXP_MAX * EXP_STEPS + 1)))
    return b"".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    quantized = export(args.model)
    digest = sha256(quantized).hexdigest()
    if digest != QUANTIZED_MODEL_HASH:
        raise ValueError(f"Exported checkpoint SHA-256 differs from the pinned integer model: {digest}")
    args.out.write_bytes(quantized)
    print(f"{digest}  {args.out} ({len(quantized):,} bytes)")


if __name__ == "__main__":
    main()
