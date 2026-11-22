"""Board-local FLAN-T5-small inference with tiled fabric W8A8 projections."""
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np


class UnigramTokenizer:
    """Viterbi segmentation for the FLAN English SentencePiece vocabulary."""

    def __init__(self, path):
        self.pieces = json.loads(path.read_text())
        self.trie = {}
        for i, (piece, score) in enumerate(self.pieces[3:], 3):
            node = self.trie
            for char in piece:
                node = node.setdefault(char, {})
            node[None] = (i, score)
        self.unknown = min(s for _, s in self.pieces) - 10

    def encode(self, text):
        normalized = "▁" + "▁".join(unicodedata.normalize("NFKC", text).split())
        best = [float("-inf")] * (len(normalized) + 1)
        prev = [None] * len(best)
        best[0] = 0
        for start in range(len(normalized)):
            node = self.trie
            single = False
            for end in range(start, len(normalized)):
                node = node.get(normalized[end])
                if node is None:
                    break
                if None in node:
                    token, score = node[None]
                    single |= end == start
                    if best[start] + score > best[end + 1]:
                        best[end + 1] = best[start] + score
                        prev[end + 1] = (start, token)
            if not single and best[start] + self.unknown > best[start + 1]:
                best[start + 1] = best[start] + self.unknown
                prev[start + 1] = (start, 2)
        out = []
        pos = len(normalized)
        while pos:
            pos, token = prev[pos]
            if token != 2 or not out or out[-1] != 2:
                out.append(token)
        return out[::-1] + [1]

    def decode(self, tokens):
        return "".join(self.pieces[i][0] if i < len(self.pieces) else "" for i in tokens
                       if i > 2).replace("▁", " ").strip()


class Flan:
    """The fabric receives every learned dense projection, including lm_head."""

    def __init__(self, path, fabric, *, reference=False, query_projection=None):
        if fabric is None and not reference:
            raise ValueError("A fabric backend is required for board generation.")
        path = Path(path)
        manifest = json.loads((path / "model.json").read_text())
        self.storage = np.memmap(str(path / "weights.bin"), mode="c", dtype="uint8")
        self.t = {k: np.ndarray(v["shape"], v["dtype"], buffer=self.storage, offset=v["offset"])
                  for k, v in manifest["tensors"].items()}
        self.tokenizer = UnigramTokenizer(path / "tokenizer.json")
        self.fabric = fabric
        self.query_projection = query_projection
        self._original = None
        self.first_output_activation = None
        self.first_output_max = None
        self.fabric_projections = 0
        self.partial_answer = ""

    def restore_head(self):
        if self._original is not None:
            row, original = self._original
            self.t["lm_head.weight"][row] = original
            self._original = None

    def select_head_row(self, row, payload):
        head = self.t["lm_head.weight"]
        if not 0 <= row < head.shape[0] or len(payload) != head.shape[1]:
            raise ValueError("Invalid output-projection patch dimensions.")
        self.restore_head()
        self._original = row, head[row].copy()
        head[row] = np.frombuffer(payload, dtype="int8")

    def install_weight_version(self, version):
        """Use a process-private output projection and verify its exact digest."""
        if version.patch is None:
            self.restore_head()
        else:
            self.select_head_row(version.patch.row, version.patch.payload)
        digest = hashlib.sha256(memoryview(self.t["lm_head.weight"])).hexdigest()
        if digest != version.classifier_sha256:
            self.restore_head()
            raise ValueError("Loaded FLAN classifier differs from the selected version.")
        return digest

    def linear(self, name, x):
        w = self.t[name + ".weight"]
        scale = np.maximum(abs(x).max(axis=1, keepdims=True) / np.float32(127), np.float32(1e-30))
        q = np.rint(x / scale).clip(-127, 127).astype("int8")
        if name == "lm_head" and self.first_output_activation is None:
            self.first_output_activation = q[0].tolist()
        if self.fabric is None:
            # Float32 represents every integer sum exactly for these dimensions.
            dots = q.astype("float32") @ w.astype("float32").T
        else:
            dots = self.query_projection.project(name, w, q) if self.query_projection is not None else None
            if dots is None:
                project = getattr(self.fabric, "matmat_array", None) or self.fabric.matmat
                dots = project(memoryview(w), q, w.shape[0])
            dots = np.asarray(dots, dtype="float32")
            self.fabric_projections += 1
        row_scales = self.t[name + ".weight.scale"]
        if name == "lm_head" and self.first_output_max is None:
            self.first_output_max = float((dots[0] * row_scales).max())
        return dots * scale * row_scales

    def head_row_scale(self, index):
        return float(self.t["lm_head.weight.scale"][index])

    def norm(self, name, x):
        return (x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + np.float32(1e-6))
                * self.t[name + ".weight"])

    def bias(self, side, length, step):
        positions = np.arange(length)[None, :] - (np.arange(length)[:, None] if step is None else step)
        n = 32 if side == "decoder" else 16
        offset = 0 if side == "decoder" else (positions > 0) * n
        dist = np.abs(positions) if side == "encoder" else np.maximum(-positions, 0)
        half = n // 2
        large = half + (np.log(np.maximum(dist, half) / half) / np.log(128 / half)
                        * (n - half)).astype(int)
        bucket = offset + np.where(dist < half, dist, np.minimum(large, n - 1))
        return self.t[side + ".block.0.layer.0.SelfAttention.relative_attention_bias.weight"][bucket].transpose(2, 0, 1)

    def attention(self, name, x, source, bias, cache, incremental):
        def project(k, value):
            return self.linear(name + "." + k, value).reshape(-1, 6, 64).transpose(1, 0, 2)
        q = project("q", x)
        key = cache.get(name)
        if key is None or incremental:
            k, v = project("k", source), project("v", source)
            if key is not None:
                k = np.concatenate((key[0], k), axis=1)
                v = np.concatenate((key[1], v), axis=1)
            cache[name] = (k, v)
        else:
            k, v = key
        scores = q @ k.transpose(0, 2, 1) + bias
        p = np.exp(scores - scores.max(axis=2, keepdims=True))
        attended = ((p / p.sum(axis=2, keepdims=True)) @ v).transpose(1, 0, 2)
        return self.linear(name + ".o", attended.reshape(-1, 384))

    def stack(self, side, ids, cache, encoder=None, step=None):
        x = self.t["shared.weight"][ids].astype("float32") * self.t["shared.weight.scale"][ids, None]
        bias = self.bias(side, len(ids) if step is None else step + 1, step)
        for i in range(8):
            base = f"{side}.block.{i}.layer."
            n = self.norm(base + "0.layer_norm", x)
            x += self.attention(base + "0.SelfAttention", n, n, bias, cache, step is not None)
            ff = "1"
            if encoder is not None:
                n = self.norm(base + "1.layer_norm", x)
                x += self.attention(base + "1.EncDecAttention", n, encoder, 0, cache, False)
                ff = "2"
            n = self.norm(base + ff + ".layer_norm", x)
            dense = base + ff + ".DenseReluDense."
            a = self.linear(dense + "wi_0", n)
            gelu = 0.5 * a * (1 + np.tanh(0.79788456 * (a + 0.044715 * a**3)))
            x += self.linear(dense + "wo", gelu * self.linear(dense + "wi_1", n))
        return self.norm(side + ".final_layer_norm", x)

    def generate(self, prompt, max_tokens=64):
        self.first_output_activation = None
        self.first_output_max = None
        self.fabric_projections = 0
        self.partial_answer = ""
        ids = self.tokenizer.encode(prompt)
        if len(ids) > 512:
            raise ValueError("The evidence and question exceed the model's 512-token input.")
        encoder = self.stack("encoder", ids, {})
        cache = {}
        tokens = []
        previous = 0
        for step in range(max_tokens):
            hidden = self.stack("decoder", [previous], cache, encoder, step)
            previous = int(np.argmax(self.linear("lm_head", hidden)))
            if previous == 1:
                break
            tokens.append(previous)
            self.partial_answer = self.tokenizer.decode(tokens)
        return self.tokenizer.decode(tokens)
