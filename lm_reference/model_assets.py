"""Verify and install a pinned board model without loading it into memory."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory


FILES = ("model.json", "tokenizer.json", "weights.bin")


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(directory, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Supply a trusted lowercase SHA-256 for model.json.")
    if file_digest(directory / "model.json") != expected:
        raise ValueError("Model manifest SHA-256 does not match the trusted hash.")
    manifest = json.loads((directory / "model.json").read_bytes())
    for filename, key in (("weights.bin", "weights_sha256"), ("tokenizer.json", "tokenizer_sha256")):
        if file_digest(directory / filename) != manifest.get(key):
            raise ValueError(f"{filename} SHA-256 does not match the pinned model manifest.")


def install_model(source, expected, root):
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Supply a trusted lowercase SHA-256 for model.json.")
    root.mkdir(mode=0o755, parents=True, exist_ok=True)
    destination = root / expected
    if destination.exists():
        verify_model(destination, expected)
        return
    with TemporaryDirectory(dir=root) as parent:
        staged = Path(parent) / "model"
        staged.mkdir()
        for name in FILES:
            shutil.copyfile(source / name, staged / name)
        verify_model(staged, expected)
        for name in FILES:
            (staged / name).chmod(0o444)
        staged.rename(destination)
        destination.chmod(0o555)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("manifest_sha256")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    install_model(args.source, args.manifest_sha256, args.destination)
