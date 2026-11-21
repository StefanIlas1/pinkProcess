"""Export the pinned FLAN-T5-small safetensors and SentencePiece assets."""
import argparse
import hashlib
import json
from pathlib import Path


REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export(source, tokenizer, out, *, adaptation=None):
    import numpy as np
    import sentencepiece as sp
    from safetensors import safe_open

    source_digest = file_sha(source)
    if adaptation is None and source_digest != "495fa51e204676f1a857a9fc13c4c89f3f5ba9f480b898cebca02add25e6d749":
        raise ValueError("Unexpected FLAN checkpoint. Supply --adaptation for trained weights.")
    if file_sha(tokenizer) != "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86":
        raise ValueError("Unexpected FLAN SentencePiece model.")
    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    binary = out / "weights.bin"
    def put(f, name, array):
        tensors[name] = {"offset": f.tell(), "shape": list(array.shape), "dtype": array.dtype.str}
        f.write(array.tobytes())

    with safe_open(str(source), framework="numpy") as model, binary.open("wb") as f:
        for name in sorted(model.keys()):
            a = model.get_tensor(name).astype("<f4")
            if a.ndim == 2 and "relative_attention_bias" not in name:
                scale = np.maximum(abs(a).max(axis=1) / 127, 1e-30).astype("<f4")
                put(f, name + ".scale", scale)
                put(f, name, np.clip(np.rint(a / scale[:, None]), -127, 127).astype("i1"))
            else:
                put(f, name, a)
    proc = sp.SentencePieceProcessor(model_file=str(tokenizer))
    vocab = [[proc.id_to_piece(i), proc.get_score(i)] for i in range(proc.get_piece_size())]
    (out / "tokenizer.json").write_text(json.dumps(vocab, ensure_ascii=False))
    manifest = {"model": "google/flan-t5-small", "revision": REVISION, "tensors": tensors,
                "source_sha256": source_digest, "weights_sha256": file_sha(binary),
                "tokenizer_sha256": file_sha(out / "tokenizer.json")}
    if adaptation:
        manifest["adaptation"] = adaptation
    (out / "model.json").write_text(json.dumps(manifest, sort_keys=True))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "tokenizer", "out"):
        p.add_argument(key, type=Path)
    p.add_argument("--adaptation", help="Label the supervised adaptation of the pinned model.")
    a = p.parse_args()
    export(a.source, a.tokenizer, a.out, adaptation=a.adaptation)
