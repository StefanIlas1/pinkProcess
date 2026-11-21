"""Reconstruct the deployed FLAN checkpoint from its pinned INT8 base."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zlib

import numpy as np

from lm_reference.export_flan import file_sha


BASE_SHA256 = "49c975a010352c432ad5245c0bb8074564083ef9869a09f16402a19c8ef0a7e1"
BLOCK = 1024 * 1024


def create(base, adapted, delta):
    if file_sha(base / "weights.bin") != BASE_SHA256:
        raise ValueError("Base weights SHA-256 mismatch.")
    if (base / "weights.bin").stat().st_size != (adapted / "weights.bin").stat().st_size:
        raise ValueError("The base and trained checkpoints must have equal dimensions.")
    delta.mkdir(parents=True, exist_ok=True)
    codec = zlib.compressobj(level=9)
    with ((base / "weights.bin").open("rb") as src, (adapted / "weights.bin").open("rb") as dst,
          (delta / "weights.delta.z").open("wb") as diff):
        while block := src.read(BLOCK):
            a = np.frombuffer(block, dtype="uint8")
            b = np.frombuffer(dst.read(len(block)), dtype="uint8")
            diff.write(codec.compress((b - a).tobytes()))
        diff.write(codec.flush())
    shutil.copyfile(adapted / "model.json", delta / "model.json")


def apply(base, delta, out):
    manifest = json.loads((delta / "model.json").read_text())
    if file_sha(base / "weights.bin") != BASE_SHA256:
        raise ValueError("Base weights SHA-256 mismatch.")
    if file_sha(base / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Base tokenizer SHA-256 mismatch.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        staging = Path(tmp) / "model"
        staging.mkdir()
        codec = zlib.decompressobj()
        with ((base / "weights.bin").open("rb") as src, (delta / "weights.delta.z").open("rb") as diff,
              (staging / "weights.bin").open("wb") as dst):
            while block := diff.read(64 * 1024):
                while block:
                    raw = codec.decompress(block, BLOCK)
                    block = codec.unconsumed_tail
                    original = src.read(len(raw))
                    if len(original) != len(raw):
                        raise ValueError("Delta exceeded the base checkpoint size.")
                    a = np.frombuffer(original, dtype="uint8")
                    b = np.frombuffer(raw, dtype="uint8")
                    dst.write((a + b).tobytes())
            if not codec.eof or codec.unused_data or src.read(1):
                raise ValueError("Delta stream is incomplete or has trailing data.")
        if file_sha(staging / "weights.bin") != manifest["weights_sha256"]:
            raise ValueError("Reconstructed weights SHA-256 mismatch.")
        shutil.copyfile(base / "tokenizer.json", staging / "tokenizer.json")
        shutil.copyfile(delta / "model.json", staging / "model.json")
        staging.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name, paths in (("create", ("base", "adapted", "delta")), ("apply", ("base", "delta", "out"))):
        p = commands.add_parser(name)
        for path in paths:
            p.add_argument(path, type=Path)
    args = parser.parse_args()
    if args.action == "apply":
        apply(args.base, args.delta, args.out)
    else:
        create(args.base, args.adapted, args.delta)
