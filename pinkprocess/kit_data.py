"""Profile imported data, recommend a quantized model and answer questions about it on this Mac."""

import csv
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from threading import Lock
from statistics import fmean
from time import perf_counter

from .kit_analyst import build_answer
from .kit_dataset import pool_sample_csv

MAX_ROWS = 50000
MAX_COLUMNS = 40
MAX_CELL_CHARS = 5000
MAX_IMPORT_BYTES = 20 * 1024 * 1024
KEPT_IMPORTS = 10
DATA_DIR = Path.home() / "Library" / "Application Support" / "PYNQ Process" / "datasets"
MISSING = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?"}
STYLES = ("short", "summary", "transcript", "table")
SIGNALS = (
    ("queue", ("waiting", "queue", "backlog", "pending")),
    ("connections", ("busy", "connection", "pool", "active")),
    ("latency", ("latency", "response", "duration", "p95", "p99")),
    ("errors", ("error", "fail", "5xx", "exception")),
    ("temperature", ("temp",)),
    ("sales", ("sales", "revenue", "orders", "price", "amount")),
    ("load", ("cpu", "memory", "load", "util")),
)
TIME_NAMES = ("time", "timestamp", "ts", "date", "datetime", "t")
FOCUS_ORDER = ("queue", "latency", "errors", "load", "temperature", "sales", "connections")
READABLE = {"queue": "queue", "connections": "connection", "latency": "latency", "errors": "error",
            "temperature": "temperature", "sales": "sales", "load": "load"}


def parse_time(text):
    try:
        return datetime.fromisoformat(text.strip().replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def is_missing(cell):
    if cell is None:
        return True
    if isinstance(cell, float):
        return not math.isfinite(cell)
    return isinstance(cell, str) and cell.strip().lower() in MISSING


def pick_delimiter(text):
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    counts = {mark: first.count(mark) for mark in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] and counts[best] > counts[","] else ","


def parse_number(text):
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text) if math.isfinite(text) else None
    if not isinstance(text, str):
        return None
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_json_records(stripped):
    try:
        if stripped[0] == "[":
            records = json.loads(stripped)
        else:
            records = []
            for number, line in enumerate(stripped.splitlines(), 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except ValueError as exc:
                        raise ValueError(f"Line {number} is not valid JSON. Use one object per line.") from exc
    except RecursionError as exc:
        raise ValueError("This JSON file is nested too deeply to read.") from exc
    except ValueError as exc:
        if str(exc).startswith("Line "):
            raise
        raise ValueError("Could not read this JSON file. Use one object per line.") from exc
    if not isinstance(records, list) or not records or not all(isinstance(record, dict) for record in records):
        raise ValueError("Each JSON line must be an object like {\"time\": 1, \"value\": 2}.")
    columns = list(dict.fromkeys(str(key) for record in records for key in record))
    if len(columns) > MAX_COLUMNS:
        raise ValueError(f"Use at most {MAX_COLUMNS} columns.")
    return columns, [[record.get(column) for column in columns] for record in records], "JSON lines"


def read_records(text, name):
    text = text.removeprefix("\ufeff").replace("\x00", "")
    stripped = text.lstrip()
    if not stripped:
        raise ValueError("The file is empty.")
    if stripped[0] in "{[":
        return read_json_records(stripped)
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=pick_delimiter(text), strict=True))
    except csv.Error as exc:
        raise ValueError("Could not read this CSV file. Check for a stray quote mark or a very long cell.") from exc
    numbered = [(number, row) for number, row in enumerate(rows, 1) if any(cell.strip() for cell in row)]
    if len(numbered) < 2:
        raise ValueError("Add a header row and at least one row of data.")
    header = [column.strip() for column in numbered[0][1]]
    if not all(header) or len(set(header)) != len(header):
        raise ValueError("Every column needs a unique name in the first row.")
    if len(header) > MAX_COLUMNS:
        raise ValueError(f"Use at most {MAX_COLUMNS} columns.")
    body = []
    for number, row in numbered[1:]:
        while len(row) > len(header) and not row[-1].strip():
            row = row[:-1]
        if len(row) != len(header):
            raise ValueError(f"Line {number} has {len(row)} columns but the header has {len(header)}.")
        body.append(row)
    return header, body, "CSV"


def parse_table(text, name=""):
    if not isinstance(text, str):
        raise ValueError("Choose a CSV or JSON lines file.")
    columns, rows, kind = read_records(text, name)
    if len(rows) > MAX_ROWS:
        raise ValueError(f"Use at most {MAX_ROWS:,} rows.")
    if len(columns) > MAX_COLUMNS:
        raise ValueError(f"Use at most {MAX_COLUMNS} columns.")
    data, types, time_column = {}, {}, None
    for index, column in enumerate(columns):
        cells = [None if is_missing(row[index]) else row[index] for row in rows]
        if any(isinstance(cell, str) and len(cell) > MAX_CELL_CHARS for cell in cells):
            raise ValueError(f"A cell in the {column} column is too long to read.")
        numbers = [parse_number(cell) for cell in cells]
        present = [cell for cell in cells if cell is not None]
        if present and all(number is not None for cell, number in zip(cells, numbers) if cell is not None):
            data[column], types[column] = numbers, "number"
            continue
        stamps = [parse_time(cell) if isinstance(cell, str) and cell else None for cell in cells]
        if present and all(stamp is not None for cell, stamp in zip(cells, stamps) if cell is not None):
            data[column], types[column] = stamps, "time"
        else:
            data[column], types[column] = [None if cell is None else str(cell) for cell in cells], "text"
    for column in columns:
        lowered = column.lower()
        named = lowered in TIME_NAMES or lowered.endswith(("_ms", "_time", "_at")) and "latency" not in lowered
        if types[column] == "time" or types[column] == "number" and named and monotonic(data[column]):
            time_column = column
            break
    return dict(columns=columns, types=types, data=data, rows=len(rows), time=time_column, format=kind)


def monotonic(values):
    present = [value for value in values if value is not None]
    return len(present) > 1 and all(b >= a for a, b in zip(present, present[1:]))


def signals_for(table):
    found = {}
    for column in table["columns"]:
        if table["types"][column] != "number" or column == table["time"]:
            continue
        lowered = column.lower()
        for signal, words in SIGNALS:
            if any(word in lowered for word in words):
                found.setdefault(signal, column)
    return found


def recommend(table, found):
    numeric = [column for column in table["columns"] if table["types"][column] == "number" and column != table["time"]]
    if "queue" in found or {"connections", "queue"} <= set(found):
        return dict(model_id="queue-onset-logistic", reason="Looks like connection-pool telemetry, so Queue-onset INT8 fits",
                    alternatives=["flan-t5-small-int8"])
    if numeric:
        label = next((READABLE[signal] for signal in FOCUS_ORDER if signal in found), "numeric")
        return dict(model_id="flan-t5-small-int8", reason=f"Looks like {label} readings, so FLAN-T5-small INT8 can explain them",
                    alternatives=[])
    return dict(model_id="flan-t5-small-int8", reason="Mostly text, so FLAN-T5-small INT8 can explain what happened",
                alternatives=[])


def queue_ready(table):
    if table["columns"][:3] != ["time_ms", "busy", "waiting"] or table["rows"] < 50:
        return False
    return all(table["types"][column] == "number" and None not in table["data"][column] for column in table["columns"][:3])


def profile_table(table, name):
    found = signals_for(table)
    columns = []
    for column in table["columns"]:
        info = dict(name=column, type=table["types"][column], missing=sum(value is None for value in table["data"][column]))
        if info["type"] == "number":
            values = [value for value in table["data"][column] if value is not None]
            if values:
                info.update(min=min(values), max=max(values), mean=round(fmean(values), 3))
        columns.append(info)
    result = dict(name=name, format=table["format"], rows=table["rows"], columns=columns, time_column=table["time"],
                  signals=sorted(found), recommendation=recommend(table, found),
                  queue_ready=queue_ready(table))
    if result["queue_ready"]:
        result["capacity"] = max(int(max(table["data"]["busy"])), 1)
    return result


def answer_question(table, question, style):
    if style not in STYLES:
        raise ValueError("Choose a short answer, summary, transcript or table.")
    return build_answer(table, question, style)


DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def load_samples():
    try:
        entries = json.loads((DATASETS_DIR / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    return tuple(dict(id=entry["id"], name=entry["name"], description=entry["description"], file=entry["file"])
                 for entry in entries if (DATASETS_DIR / entry["file"]).is_file())


SAMPLES = load_samples()


def sample_csv(sample_id):
    sample = next((item for item in SAMPLES if item["id"] == sample_id), None)
    if sample is None:
        raise ValueError("Unknown sample.")
    return (DATASETS_DIR / sample["file"]).read_text(encoding="utf-8")


def timed(function, *args):
    started = perf_counter()
    result = function(*args)
    return result, round((perf_counter() - started) * 1000, 1)


def safe_dataset_name(name):
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(str(name)).name).strip(" .")[:80]
    return stem or "dataset.csv"


class DatasetStore:
    def __init__(self, directory=DATA_DIR, keep=KEPT_IMPORTS):
        self.directory = Path(directory)
        self.keep = keep
        self.lock = Lock()

    def save(self, name, text):
        safe = safe_dataset_name(name)
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=self.directory, prefix=".import.", suffix=".tmp")
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                    handle.write(text)
                os.replace(temporary, self.directory / safe)
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
            for stale in self.files()[self.keep:]:
                try:
                    stale.unlink()
                except OSError:
                    pass
        return safe

    def files(self):
        try:
            found = [path for path in self.directory.iterdir() if path.is_file() and not path.name.startswith(".")]
            return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)
        except OSError:
            return []

    def names(self):
        return [path.name for path in self.files()]

    def read(self, name):
        path = self.directory / safe_dataset_name(name)
        try:
            if path.parent != self.directory or path.stat().st_size > MAX_IMPORT_BYTES:
                raise ValueError("That dataset is not available.")
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise ValueError("That dataset is not available.") from None
