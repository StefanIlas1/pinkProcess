"""Board-owned inference jobs, immutable evidence capture and weight selection."""
from contextlib import closing, contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from threading import Lock, Thread
from time import time_ns
from uuid import uuid4

from lm_reference.flan_prompt import telemetry_prompt
from .kit_dataset import dataset_snapshot
from .kit_evidence import analyst_evidence

HISTORY_MODEL = "Queue history v2"


DATASET_STAGES = {
    'generalization': 'Distinguish observations from explanations and forecasts.',
    'short_answer': 'Answer briefly using the reported measurements.',
    'summary': 'Summarize the key observations and the forecast.',
    'final': '',
    'followup': '',
}


class BoardLanguage:
    def __init__(self, runtime, versions, state_dir, snapshot, dataset_score=None, history_runtime=None):
        self.base_runtime = runtime
        self.runtime = runtime
        self.history_runtime = history_runtime
        self.versions = versions
        self.capture_db = Path(state_dir) / 'lm-captures.sqlite'
        self.capture_db.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot = snapshot
        self.dataset_score = dataset_score
        self.lock = Lock()
        self.jobs = {}
        boot_file = Path("/proc/sys/kernel/random/boot_id")
        self.boot_id = boot_file.read_text().strip() if boot_file.exists() else None
        self.activation = None
        self.activation_job_id = None
        self.weight_state_ok = True
        self.active = versions.active()
        self._install(self.active)
        with self.open_captures() as db:
            db.execute('CREATE TABLE IF NOT EXISTS captures (id TEXT PRIMARY KEY, body BLOB NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS capture_baselines (id TEXT PRIMARY KEY, body BLOB NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, body BLOB NOT NULL)')

    @contextmanager
    def open_captures(self):
        try:
            with closing(sqlite3.connect(self.capture_db)) as db, db:
                db.execute("PRAGMA synchronous = FULL")
                yield db
        except sqlite3.Error as exc:
            raise RuntimeError(f"Evidence capture database failed: {exc}") from exc

    def _install(self, version, previous=None):
        target_runtime = self.history_runtime if self.is_history(version.name) else self.base_runtime
        if target_runtime is not self.runtime:
            self.runtime = target_runtime
            runner = getattr(self.runtime, "query_projection", None)
            if runner is not None:
                # Packed layouts and resident DMA buffers belong to one set of tensors.
                runner.bind_model(self.runtime.t)
        self.runtime.install_weight_version(version)
        fabric = getattr(self.runtime, "fabric", None)
        if fabric is None:
            return
        target = version.patch or (previous or self.active).patch
        index = target.row if target else 0
        payload = self.runtime.t["lm_head.weight"][index].tobytes()
        fabric.verify_row_bytes(payload)

    def is_history(self, name):
        return self.history_runtime is not None and name == HISTORY_MODEL

    def capture(self, question, capture_id):
        baseline = None
        with self.open_captures() as db:
            if capture_id is not None:
                if not isinstance(capture_id, str) or not re.fullmatch('[0-9a-f]{64}', capture_id):
                    raise ValueError("Invalid capture ID.")
                row = db.execute('SELECT body FROM captures WHERE id=?', (capture_id,)).fetchone()
                if not row or hashlib.sha256(row[0]).hexdigest() != capture_id:
                    raise ValueError("Saved capture unavailable or damaged.")
                record = json.loads(row[0])
                if record["question"] != question:
                    raise ValueError("Replay must use the original question.")
                row = db.execute('SELECT body FROM capture_baselines WHERE id=?', (capture_id,)).fetchone()
                if row:
                    baseline = json.loads(row[0])
            else:
                snapshot = self.snapshot()
                if not snapshot.get("fresh") or not snapshot.get("latest"):
                    raise ValueError("A fresh reading is required to capture a new incident.")
                record = dict(question=question, captured_at_ms=time_ns() // 1_000_000, snapshot=snapshot,
                              prompt=telemetry_prompt(question, analyst_evidence(snapshot)))
                if self.is_history(self.active.name):
                    record["history_prompt"] = telemetry_prompt(question, analyst_evidence(snapshot), history=True)
                body = json.dumps(record, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()
                capture_id = hashlib.sha256(body).hexdigest()
                db.execute('INSERT OR IGNORE INTO captures VALUES (?,?)', (capture_id, body))
                record = json.loads(body)
        return capture_id, record, baseline

    def list_captures(self, before):
        if before is not None and (type(before) is not int or not 0 < before < 2**63):
            raise ValueError("Invalid saved-capture page.")
        page_size = 20
        with self.open_captures() as db:
            rows = db.execute('''SELECT c.rowid, c.id, c.body, b.body FROM captures c
                                LEFT JOIN capture_baselines b ON c.id = b.id
                                WHERE (? IS NULL OR c.rowid < ?) ORDER BY c.rowid DESC LIMIT ?''',
                              (before, before, page_size + 1)).fetchall()
        captures = []
        for _, capture_id, body, baseline in rows[:page_size]:
            if hashlib.sha256(body).hexdigest() != capture_id:
                continue
            record = json.loads(body)
            captures.append(dict(id=capture_id, question=record["question"],
                                 captured_at_ms=record.get("captured_at_ms"),
                                 reading_time_ms=record["snapshot"]["latest"]["time_ms"],
                                 baseline=json.loads(baseline) if baseline else None))
        return dict(ok=True, captures=captures, before=rows[page_size - 1][0] if len(rows) > page_size else None)

    def run(self, job_id, prompt, initial, selected=None):
        try:
            answer = self.runtime.generate(prompt)
            if not answer.strip():
                raise RuntimeError("The board language model generated an empty answer.")
            activation = self.runtime.first_output_activation
            activation_max = getattr(self.runtime, "first_output_max", None)
            result = dict(state="complete", answer=answer, fabric_projections=self.runtime.fabric_projections,
                          accelerated_projections=getattr(self.runtime, "accelerated_projections", 0))
            profile = getattr(self.runtime, "projection_profile", None)
            if profile is not None:
                phases = profile()
                if phases:
                    result["projection_phases"] = phases
            if initial:
                with self.lock:
                    job = self.jobs[job_id]
                    baseline = dict(answer=answer, version=job["active"])
                    with self.open_captures() as db:
                        db.execute('INSERT OR IGNORE INTO capture_baselines VALUES (?,?)',
                                   (job["capture_id"], json.dumps(baseline, allow_nan=False).encode()))
                    result["baseline"] = baseline
        except Exception as exc:
            result = dict(state="failed", error=str(exc))
        with self.lock:
            job = self.jobs[job_id]
            if selected is not None:
                try:
                    self._install(self.active, previous=selected)
                except Exception:
                    self.weight_state_ok = False
                    result = dict(state="failed", error="Could not restore the live language-model weights.")
            if result["state"] == "complete":
                self.activation = activation
                self.activation_job_id = job_id
                self.activation_max = activation_max
                if job.get("stage") in ("final", "followup"):
                    try:
                        with self.open_captures() as db:
                            dataset = self.dataset(db, job["dataset_id"])
                            dataset["last_chat"] = [job["question"], answer]
                            self.store_dataset(db, dataset)
                    except RuntimeError as exc:
                        result = dict(state="failed", error=str(exc))
            job.update(result)

    def start_job(self, job, prompt, initial=False, selected=None):
        job_id = uuid4().hex
        job.update(ok=True, job_id=job_id, state="running", model="flan-t5-small")
        if len(self.jobs) == 32:
            del self.jobs[next(iter(self.jobs))]
        self.jobs[job_id] = job
        try:
            Thread(target=self.run, args=(job_id, prompt, initial, selected), daemon=True).start()
        except RuntimeError:
            del self.jobs[job_id]
            raise
        return job.copy()

    def dataset(self, db, dataset_id):
        if not isinstance(dataset_id, str) or not re.fullmatch("[0-9a-f]{32}", dataset_id):
            raise ValueError("Invalid dataset ID.")
        row = db.execute("SELECT body FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        if row is None:
            raise ValueError("This dataset is unavailable on the paired board.")
        return json.loads(row[0])

    def store_dataset(self, db, dataset):
        db.execute("INSERT OR REPLACE INTO datasets VALUES (?,?)",
                   (dataset["dataset_id"], json.dumps(dataset, allow_nan=False)))

    def dataset_metadata(self, dataset):
        return dict(ok=True, **{k: dataset[k] for k in ("dataset_id", "name", "snapshot", "selected_model")},
                    summary=dataset["snapshot"]["observations"], models=self.versions.list())

    def handle_dataset(self, command):
        if self.dataset_score is None:
            raise ValueError("This board does not support uploaded datasets.")
        op = command["op"]
        with self.open_captures() as db:
            if op == "lm_dataset_upload":
                name = command.get("name", "Uploaded CSV")
                if not isinstance(name, str) or not 0 < len(name.strip()) <= 160:
                    raise ValueError("Supply a dataset name of up to 160 characters.")
                dataset_id = uuid4().hex
                snapshot = dataset_snapshot(dataset_id, name.strip(), command.get("csv"),
                                            command.get("capacity"), self.dataset_score)
                dataset = dict(dataset_id=dataset_id, name=name.strip(), snapshot=snapshot, csv=command["csv"],
                               selected_model=self.active.name)
                self.store_dataset(db, dataset)
                return self.dataset_metadata(dataset)
            dataset = self.dataset(db, command.get("dataset_id"))
            if op == "lm_dataset_models":
                return self.dataset_metadata(dataset)
            if op == "lm_dataset_select":
                dataset["selected_model"] = self.versions.get(command.get("name")).name
                self.store_dataset(db, dataset)
                return self.dataset_metadata(dataset)
            if op in ("lm_dataset_calibration", "lm_dataset_calibrate", "lm_dataset_save") and self.is_history(dataset["selected_model"]):
                raise ValueError("Select the original checkpoint to create an output-row edit.")
            if op == "lm_dataset_calibration":
                job = self.jobs.get(self.activation_job_id)
                if (not job or job.get("state") != "complete" or job.get("dataset_id") != dataset["dataset_id"]
                        or job["active"]["name"] != "original" or not self.activation):
                    return dict(ok=True, ready=False)
                return dict(ok=True, ready=True, **self.row_context(command.get("target_row")),
                            selected_model=dataset["selected_model"])
            if op == "lm_dataset_save":
                if command.get("selected_model") != dataset["selected_model"]:
                    raise ValueError("The dataset model changed during synthesis.")
                if command.get("base_sha256") != self.versions.layout.checkpoint_sha256:
                    raise ValueError("Weight edit belongs to another base checkpoint.")
                version = self.versions.save_astra(command.get("name"), command.get("target_row"),
                                                   command.get("response"))
                dataset["selected_model"] = version.name
                self.store_dataset(db, dataset)
                return self.dataset_metadata(dataset)
        if op not in ("lm_dataset_ask", "lm_dataset_calibrate"):
            raise ValueError("Unsupported dataset operation.")
        if any(job["state"] == "running" for job in self.jobs.values()):
            raise ValueError("A board answer is in progress.")
        calibration = op == "lm_dataset_calibrate"
        question = ("Summarize the observed connection pool readings." if calibration else command.get("question"))
        if not isinstance(question, str) or not 0 < len(question.strip()) <= 500:
            raise ValueError("Supply a question of up to 500 characters.")
        stage = command.get("stage", "final")
        if stage not in DATASET_STAGES:
            raise ValueError("Unsupported dataset answer stage.")
        snapshot = dataset["snapshot"]
        version = self.versions.get("original" if calibration else dataset["selected_model"])
        evidence = analyst_evidence(snapshot)
        if self.is_history(version.name):
            prompt = telemetry_prompt(question, evidence, history=True)
        else:
            prompt = telemetry_prompt(question, evidence)
        if DATASET_STAGES[stage]:
            prompt = DATASET_STAGES[stage] + " " + prompt
        if stage == "followup" and dataset.get("last_chat"):
            previous_question, previous_answer = dataset["last_chat"]
            prompt = (f"Previous question: {previous_question[:160]}\n"
                      f"Previous answer: {previous_answer[:200]}\n" + prompt)
        selected = version if version.name != self.active.name else None
        try:
            if selected is not None:
                self._install(selected)
            job = dict(dataset_id=dataset["dataset_id"], question=question.strip(),
                       stage="calibration" if calibration else stage,
                       active=version.metadata(), prompt_variant="queue_history" if self.is_history(version.name) else "original")
            return {**self.start_job(job, prompt, selected=selected), "snapshot": snapshot}
        except Exception:
            if selected is not None:
                try:
                    self._install(self.active, previous=selected)
                except Exception:
                    self.weight_state_ok = False
            raise

    def row_context(self, target):
        index = self.token_id(target)
        layout = self.versions.layout
        if type(index) is not int or not 0 <= index < layout.rows:
            raise ValueError("Output row outside model vocabulary.")
        start = index * layout.width
        raw = self.versions.classifier[start:start + layout.width]
        return dict(row=list(raw.cast('b')), target_row=index, token=self.base_runtime.tokenizer.decode([index]),
                    activation=self.activation, calibration_job_id=self.activation_job_id,
                    base_sha256=layout.checkpoint_sha256, max_score=self.activation_max,
                    row_scale=self.base_runtime.head_row_scale(index) if self.activation_max is not None else None)

    def token_id(self, target):
        if isinstance(target, str) and 0 < len(target.strip()) <= 80:
            ids = self.base_runtime.tokenizer.encode(target.strip())
            if len(ids) != 2 or ids[0] <= 2:
                raise ValueError("Choose a word encoded as one token, such as Warning, or enter its token ID.")
            return ids[0]
        return target

    def handle(self, command):
        op = command["op"]
        with self.lock:
            if op == 'lm_result':
                job = self.jobs.get(command.get('job_id'))
                if job is None:
                    raise ValueError("Unknown inference job.")
                if job["state"] == "running":
                    return {**job, "partial_answer": getattr(self.runtime, "partial_answer", ""),
                            "fabric_projections": getattr(self.runtime, "fabric_projections", 0),
                            "accelerated_projections": getattr(self.runtime, "accelerated_projections", 0)}
                return {**job}
            if op == 'lm_captures':
                return self.list_captures(command.get('before'))
            if not self.weight_state_ok:
                raise RuntimeError("Language-model weights require recovery. Restart the board service.")
            if op.startswith("lm_dataset_"):
                return self.handle_dataset(command)
            if op == 'lm_versions':
                return self.status()
            if op in ('lm_row', 'lm_save') and self.is_history(self.active.name):
                raise ValueError("Select the original checkpoint to create an output-row edit.")
            if op == 'lm_save':
                if command.get("base_sha256", self.versions.layout.checkpoint_sha256) != self.versions.layout.checkpoint_sha256:
                    raise ValueError("Weight edit belongs to another base checkpoint.")
                self.versions.save_astra(command.get("name"), self.token_id(command.get("target_row")), command.get("response"))
                return self.status()
            if op == 'lm_row':
                last = self.jobs.get(self.activation_job_id)
                if last and self.is_history(last["active"]["name"]):
                    raise ValueError("Generate calibration with the original checkpoint before editing.")
                return dict(ok=True, **self.row_context(command.get('target_row')))
            if any(job["state"] == "running" for job in self.jobs.values()):
                raise ValueError("A board answer is in progress.")
            if op in ('lm_select', 'lm_restore'):
                selected = self.versions.get("original" if op == 'lm_restore' else command.get("name"))
                try:
                    self._install(selected)
                    self.versions.select(selected.name)
                except Exception:
                    try:
                        self._install(self.active)
                    except Exception as restore_error:
                        self.weight_state_ok = False
                        raise RuntimeError(
                            "Could not restore the previous language-model weights. Restart the board service."
                        ) from restore_error
                    raise
                self.activation = None
                self.activation_job_id = None
                self.activation_max = None
                self.active = selected
                return self.status()
            if op != 'lm_start':
                raise ValueError("Unsupported language-model operation.")
            question = command.get('question')
            if not isinstance(question, str) or not 0 < len(question.strip()) <= 500:
                raise ValueError("Supply a question of up to 500 characters.")
            replay_id = command.get('capture_id')
            capture_id, record, baseline = self.capture(question, replay_id)
            use_history_prompt = self.is_history(self.active.name) and "history_prompt" in record
            prompt = record["history_prompt"] if use_history_prompt else record["prompt"]
            job = dict(capture_id=capture_id, active=self.active.metadata(), baseline=baseline,
                       prompt_variant="queue_history" if use_history_prompt else "original",
                       captured_at_ms=record.get("captured_at_ms"))
            return {**self.start_job(job, prompt, initial=replay_id is None),
                    "snapshot": record["snapshot"]}

    def status(self):
        return dict(ok=True, model="flan-t5-small", boot_id=self.boot_id, active=self.active.metadata(),
                    versions=self.versions.list(), capture_listing_supported=True,
                    datasets_supported=self.dataset_score is not None)
