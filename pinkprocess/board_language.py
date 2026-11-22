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
from .kit_evidence import analyst_evidence


class BoardLanguage:
    def __init__(self, runtime, versions, state_dir, snapshot):
        self.runtime = runtime
        self.versions = versions
        self.capture_db = Path(state_dir) / 'lm-captures.sqlite'
        self.capture_db.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot = snapshot
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

    @contextmanager
    def open_captures(self):
        try:
            with closing(sqlite3.connect(self.capture_db)) as db, db:
                db.execute("PRAGMA synchronous = FULL")
                yield db
        except sqlite3.Error as exc:
            raise RuntimeError(f"Evidence capture database failed: {exc}") from exc

    def _install(self, version):
        self.runtime.install_weight_version(version)
        fabric = getattr(self.runtime, "fabric", None)
        if fabric is None:
            return
        target = version.patch or self.active.patch
        index = target.row if target else 0
        payload = self.runtime.t["lm_head.weight"][index].tobytes()
        fabric.verify_row_bytes(payload)

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

    def run(self, job_id, prompt, initial):
        try:
            answer = self.runtime.generate(prompt)
            if not answer.strip():
                raise RuntimeError("The board language model generated an empty answer.")
            activation = self.runtime.first_output_activation
            activation_max = getattr(self.runtime, "first_output_max", None)
            result = dict(state="complete", answer=answer, fabric_projections=self.runtime.fabric_projections,
                          accelerated_projections=getattr(self.runtime, "accelerated_projections", 0))
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
            if result["state"] == "complete":
                self.activation = activation
                self.activation_job_id = job_id
                self.activation_max = activation_max
            self.jobs[job_id].update(result)

    def token_id(self, target):
        if isinstance(target, str) and 0 < len(target.strip()) <= 80:
            ids = self.runtime.tokenizer.encode(target.strip())
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
            if op == 'lm_versions':
                return self.status()
            if op == 'lm_save':
                if command.get("base_sha256", self.versions.layout.checkpoint_sha256) != self.versions.layout.checkpoint_sha256:
                    raise ValueError("Weight edit belongs to another base checkpoint.")
                self.versions.save_astra(command.get("name"), self.token_id(command.get("target_row")), command.get("response"))
                return self.status()
            if op == 'lm_row':
                index = self.token_id(command.get('target_row'))
                layout = self.versions.layout
                if type(index) is not int or not 0 <= index < layout.rows:
                    raise ValueError("Output row outside model vocabulary.")
                start = index * layout.width
                raw = self.versions.classifier[start:start + layout.width]
                return dict(ok=True, row=list(raw.cast('b')), target_row=index,
                            token=self.runtime.tokenizer.decode([index]), activation=self.activation,
                            calibration_job_id=self.activation_job_id,
                            base_sha256=layout.checkpoint_sha256,
                            max_score=getattr(self, "activation_max", None),
                            row_scale=self.runtime.head_row_scale(index) if getattr(self, "activation_max", None) is not None else None)
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
            job_id = uuid4().hex
            job = dict(ok=True, job_id=job_id, state="running", capture_id=capture_id,
                       active=self.active.metadata(), model="flan-t5-small", baseline=baseline,
                       captured_at_ms=record.get("captured_at_ms"))
            if len(self.jobs) == 32:
                del self.jobs[next(iter(self.jobs))]
            self.jobs[job_id] = job
            try:
                Thread(target=self.run, args=(job_id, record["prompt"], replay_id is None), daemon=True).start()
            except RuntimeError:
                del self.jobs[job_id]
                raise
            return {**job, "snapshot": record["snapshot"]}

    def status(self):
        return dict(ok=True, model="flan-t5-small", boot_id=self.boot_id, active=self.active.metadata(),
                    versions=self.versions.list(), capture_listing_supported=True)
