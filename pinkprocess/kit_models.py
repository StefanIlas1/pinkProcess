"""The model library: built-in quantized models plus models the user publishes."""

import json
from pathlib import Path
import re
import secrets
import tempfile
from threading import RLock

MODELS_PATH = Path.home() / "Library" / "Application Support" / "PYNQ Process" / "models.json"
MAX_PUBLISHED = 100
MAKERS = ("PYNQ team", "Stefan Ilas", "you", "community")
TASK_TAGS = ("early warning", "explain", "anomaly", "custom")
BUILT_IN = [
    {"id": "queue-onset-logistic", "name": "Queue-onset logistic", "task": "Early warning when requests start piling up",
     "task_tag": "early warning", "kind": "queue", "available": True, "maker": "Stefan Ilas", "size_kb": 1, "size": "tiny"},
    {"id": "flan-t5-small-int8", "name": "FLAN-T5-small INT8", "task": "Explains what is going wrong",
     "task_tag": "explain", "kind": "language", "available": True, "maker": "PYNQ team", "size_kb": 64000, "size": "small"},
]


def size_label(size_kb):
    return "tiny" if size_kb < 1024 else "small" if size_kb < 128 * 1024 else "large"


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "model"


class ModelStore:
    def __init__(self, path=MODELS_PATH):
        self.path = Path(path)
        self.lock = RLock()

    def published(self):
        try:
            items = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [item for item in items if isinstance(item, dict) and isinstance(item.get("id"), str)] if isinstance(items, list) else []

    def list(self):
        models = [{**model, "runs_on": "board", "published": False, "rank": index + 1} for index, model in enumerate(BUILT_IN)]
        for item in self.published():
            models.append({**item, "runs_on": "board", "published": True, "rank": len(models) + 1})
        return models

    def write(self, items):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with open(handle, "w", encoding="utf-8") as file:
                json.dump(items, file)
            Path(name).replace(self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    def publish(self, data):
        name, task = data.get("name"), data.get("task", "")
        if not isinstance(name, str) or not 0 < len(name.strip()) <= 80:
            raise ValueError("Give the model a name of up to 80 characters.")
        if not isinstance(task, str) or len(task) > 200:
            raise ValueError("Describe the task in up to 200 characters.")
        size_kb = data.get("size_kb", 64000)
        if isinstance(size_kb, bool) or not isinstance(size_kb, (int, float)) or not 0 < size_kb < 10 ** 9:
            raise ValueError("The model size is not valid.")
        tag = data.get("task_tag", "custom")
        if tag not in TASK_TAGS:
            tag = "custom"
        base = data.get("base") if isinstance(data.get("base"), str) else "flan-t5-small-int8"
        with self.lock:
            items = self.published()
            if len(items) >= MAX_PUBLISHED:
                raise ValueError("The marketplace is full. Unpublish a model first.")
            taken = {item["id"] for item in items} | {model["id"] for model in BUILT_IN}
            model_id = "you-" + slug(name)
            if model_id in taken:
                model_id += "-" + secrets.token_hex(2)
            item = {"id": model_id, "name": name.strip(), "task": task.strip() or "A skill made by you", "task_tag": tag,
                    "kind": "language", "available": True, "maker": "you", "size_kb": size_kb, "size": size_label(size_kb), "base": base[:80]}
            self.write(items + [item])
        return item

    def unpublish(self, model_id):
        with self.lock:
            items = self.published()
            kept = [item for item in items if item["id"] != model_id]
            if len(kept) == len(items):
                raise ValueError("That model is not published.")
            self.write(kept)
        return {"removed": model_id}
