"""Mac operator for the board-resident PYNQ Process telemetry model."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Lock, Thread
from time import monotonic, sleep, time
from uuid import uuid4
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .live_analyst import analyze as analyze_live_evidence
from .backlog import VERSION as BACKLOG_VERSION
from .kit_dataset import prepare_dataset
from .astra_edit import propose
from .board_sim import HOST as SIM_HOST, SimulatedBoard
from .kit_transport import BoardOperationError, BoardTransport
from .kit_data import DATA_DIR, MAX_IMPORT_BYTES, SAMPLES, STYLES, DatasetStore, answer_question, parse_table, profile_table, sample_csv, timed
from .kit_accounts import CALLBACK_PATH, Accounts
from .kit_config import load_env
from .kit_export import export_action
from .kit_evidence import analyst_evidence, live_answer
import secrets


PAGE = Path(__file__).with_name("kit.html")
LOGO = Path(__file__).with_name("static") / "logo.png"
SIGNED_IN_PAGE = b"<!doctype html><meta charset=utf-8><title>PYNQ Process</title><body style=\"font:16px -apple-system,Helvetica Neue,sans-serif;padding:48px\">You are logged in. Return to PYNQ Process.</body>"
QUESTIONS = {
    "latest": "Will requests start waiting in the next ten seconds?",
    "backlog": "Will the current queue persist or grow over the next ten seconds?",
    "warnings": "Which readings triggered a ten-second queue warning?",
}
MAX_BODY_BYTES = MAX_IMPORT_BYTES + 64 * 1024
REQUEST_TIMEOUT_SECONDS = 30
QUICK_READS = {
    "How many requests are waiting now?": ("stream_status", None, ["waiting"]),
    "What was the peak queue in the last 30 seconds?": ("peak", 30000, ["waiting"]),
    "Is the queue growing over the last 10 seconds?": ("trend", 10000, ["waiting"]),
    "What does the FPGA predict for the next ten seconds?": ("stream_forecast", None, []),
}


class Kit:
    def __init__(self, transport, prepare=prepare_dataset, store=None):
        self.prepare = prepare
        self.transport = transport
        self.store = store
        self.host = None
        self.evidence_supported = False
        self.lm_supported = False
        self.connection_epoch = 0
        self.syntheses = {}
        self.table = None
        self.table_name = ""
        self.table_profile = None
        self.lock = Lock()
        self.data_lock = Lock()
        self.jobs = {}
        self.jobs_lock = Lock()

    def import_data(self, name, text, remember=True):
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError("Choose a file to import.")
        if not isinstance(text, str):
            raise ValueError("Choose a CSV or JSON lines file.")
        if len(text) > MAX_IMPORT_BYTES:
            raise ValueError("This file is larger than 20 MB. Import a smaller file.")
        table, parse_ms = timed(parse_table, text, name)
        profile, profile_ms = timed(profile_table, table, name.strip())
        with self.data_lock:
            self.table, self.table_name, self.table_profile = table, name.strip(), profile
        if remember and self.store is not None:
            try:
                self.store.save(name, text)
            except (OSError, ValueError):
                pass
        return dict(profile=profile, timings=dict(data_ms=parse_ms, profile_ms=profile_ms))

    def ask_data(self, data):
        if set(data) - {"action", "question", "style"}:
            raise ValueError("Data questions accept only a question and an answer style.")
        question, style = data.get("question"), data.get("style", "short")
        if not isinstance(question, str) or not 0 < len(question.strip()) <= 500:
            raise ValueError("Enter a question of up to 500 characters.")
        if style not in STYLES:
            raise ValueError("Choose a short answer, summary, transcript or table.")
        with self.data_lock:
            table, profile = self.table, self.table_profile
        if table is None:
            raise ValueError("Import data first.")
        result, answer_ms = timed(answer_question, table, question.strip(), style)
        where, fabric_ms, verdict = "this Mac", 0.0, None
        if self.host is not None and profile["queue_ready"]:
            csv_text = "time_ms,busy,waiting\n" + "".join(
                f"{int(t)},{int(b)},{int(w)}\n" for t, b, w in zip(table["data"]["time_ms"], table["data"]["busy"], table["data"]["waiting"]))
            try:
                with self.lock:
                    forecast, fabric_ms = timed(self.analyze, dict(question="latest", csv=csv_text, capacity=profile["capacity"]))
                latest = forecast["rows"][-1]
                verdict = ("The board's forecast: queue warning at " if latest["alert"] else "The board's forecast: no queue warning at ") + f"{latest['time_ms'] / 1000:.1f} s."
                where = "PYNQ-Z2"
            except (ValueError, RuntimeError, OSError, KeyError, IndexError, TypeError):
                verdict = None
                fabric_ms = 0.0
        if verdict:
            result["answer"] += "\n" + verdict if result["style"] != "table" else ""
            result["readings"] += " " + verdict
        return dict(result, where=where, model=profile["recommendation"]["model_id"],
                    timings=dict(data_ms=0.0, profile_ms=0.0, model_ms=0.0, fabric_ms=fabric_ms, answer_ms=answer_ms))

    def begin_data_job(self, data):
        if set(data) - {"action", "question", "style", "local_only"}:
            raise ValueError("Data questions accept only a question and an answer style.")
        local_only = data.get("local_only", False)
        if not isinstance(local_only, bool):
            raise ValueError("local_only must be a boolean.")
        question = data.get("question")
        if not isinstance(question, str) or not 0 < len(question.strip()) <= 500:
            raise ValueError("Enter a question of up to 500 characters.")
        if data.get("style", "short") not in STYLES:
            raise ValueError("Choose a short answer, summary, transcript or table.")
        with self.data_lock:
            table, profile = self.table, self.table_profile
        if table is None:
            raise ValueError("Import data first.")
        job = dict(job_id=secrets.token_hex(8), state="running", ready={}, order=list(STYLES), where="this Mac",
                   model=profile["recommendation"]["model_id"], timings={}, started=time())
        with self.jobs_lock:
            for old in sorted(self.jobs, key=lambda key: self.jobs[key]["started"])[:-19]:
                del self.jobs[old]
            self.jobs[job["job_id"]] = job
        Thread(target=self.run_data_job, args=(job, table, profile, question.strip(), local_only), daemon=True).start()
        return self.data_job_view(job)

    def run_data_job(self, job, table, profile, question, local_only):
        try:
            timings = dict(data_ms=0.0, profile_ms=0.0, model_ms=0.0, fabric_ms=0.0, answer_ms=0.0)
            verdict = None
            for style in STYLES:
                result, answer_ms = timed(answer_question, table, question, style)
                timings["answer_ms"] += answer_ms
                if style == "short" and not local_only:
                    verdict, fabric_ms = self.board_verdict(table, profile)
                    timings["fabric_ms"] = fabric_ms
                    if verdict:
                        job["where"] = "PYNQ-Z2"
                if verdict:
                    if result["style"] != "table":
                        result["answer"] += "\n" + verdict
                    result["readings"] += " " + verdict
                job["ready"][style] = result
                job["timings"] = dict(timings)
            job["state"] = "complete"
        except (ValueError, RuntimeError, OSError) as exc:
            job["error"] = str(exc)
            job["state"] = "error"

    def board_verdict(self, table, profile):
        if self.host is None or not profile["queue_ready"]:
            return None, 0.0
        csv_text = "time_ms,busy,waiting\n" + "".join(
            f"{int(t)},{int(b)},{int(w)}\n" for t, b, w in zip(table["data"]["time_ms"], table["data"]["busy"], table["data"]["waiting"]))
        try:
            with self.lock:
                forecast, fabric_ms = timed(self.analyze, dict(question="latest", csv=csv_text, capacity=profile["capacity"]))
            latest = forecast["rows"][-1]
            return ("The board's forecast: queue warning at " if latest["alert"] else "The board's forecast: no queue warning at ") + f"{latest['time_ms'] / 1000:.1f} s.", fabric_ms
        except (ValueError, RuntimeError, OSError, KeyError, IndexError, TypeError):
            return None, 0.0

    def data_job_view(self, job):
        view = {key: job[key] for key in ("job_id", "state", "order", "where", "model", "timings")}
        view["ready"] = dict(job["ready"])
        if job["state"] == "error":
            view["error"] = job.get("error", "The answer could not be prepared.")
        return view

    def progress_data_job(self, data):
        if set(data) != {"action", "job_id"}:
            raise ValueError("Progress takes only a job id.")
        with self.jobs_lock:
            job = self.jobs.get(data["job_id"]) if isinstance(data["job_id"], str) else None
        if job is None:
            raise ValueError("That answer is no longer available. Ask again.")
        return self.data_job_view(job)

    def disconnect(self):
        self.host = None
        self.connection_epoch += 1
        self.syntheses.clear()
        self.evidence_supported = False
        self.lm_supported = False
        self.transport.close()

    def board(self, action, **params):
        if self.host is None:
            raise ValueError("Connect a board first.")
        try:
            result = self.transport.request(action, **params)
        except BoardOperationError:
            raise
        except (OSError, RuntimeError):
            self.disconnect()
            raise
        if result.get("ok") is False:
            raise ValueError(result.get("error", "The board rejected this request."))
        return result

    def analyze(self, data):
        question = data.get("question")
        if question not in QUESTIONS:
            raise ValueError("Select one of the questions supported by this model.")
        csv_text = data.get("csv")
        if not isinstance(csv_text, str):
            raise ValueError("Choose a telemetry CSV.")
        if question == "backlog":
            dataset = prepare_dataset(csv_text, data.get("capacity"), backlog=True)
            vector = dataset.pop("features")
            status = self.board("status")
            if status.get("backlog_version") != BACKLOG_VERSION:
                raise ValueError("This kit needs the queue-outlook firmware update.")
            answer = dict(host=self.host, question=question, dataset=dataset, backend=status["backend"],
                          version=status["backlog_version"], persists=None, grows=None)
            if dataset["waiting"]:
                forecast = self.board("backlog", features=vector)
                answer.update(version=forecast["version"], weights_sha256=forecast["weights_sha256"],
                              persists=forecast["scores"][0]>=0, grows=forecast["scores"][1]>=0)
            return answer
        dataset = self.prepare(csv_text, data.get("capacity"))
        features = dataset.pop("features")
        times = dataset.pop("times_ms")
        if not features:
            raise ValueError("The dataset has no eligible forecast windows.")
        if question == "latest":
            features, times = features[-1:], times[-1:]
        rows = []
        identity = None
        for start in range(0, len(features), 512):
            batch = features[start:start + 512]
            result = self.board("query_batch", features=batch)
            new_identity = (result["version"], result["weights_sha256"], result["threshold"], result["score_scale"])
            if identity is not None and new_identity != identity:
                raise RuntimeError("The active model changed during analysis. Run the query again.")
            identity = new_identity
            scores, alerts = result["scores"], result["alerts"]
            if len(scores) != len(batch) or len(alerts) != len(batch):
                raise RuntimeError("The board returned an incomplete set of forecasts.")
            rows.extend({"time_ms": timestamp, "score": score, "alert": alert}
                        for timestamp, score, alert in zip(times[start:start + len(batch)], scores, alerts))
        return {
            "host": self.host, "question": question, "dataset": dataset,
            "backend": result["backend"], "model": result["model"],
            "version": result["version"], "weights_sha256": result["weights_sha256"],
            "threshold": result["threshold"], "score_scale": result["score_scale"],
            "windows": len(rows), "warnings": sum(row["alert"] for row in rows),
            "rows": rows,
        }

    def query_live(self, data):
        if set(data) - {"action", "question", "capture_id"}:
            raise ValueError("Live questions accept only question text. The board supplies the telemetry.")
        question = data.get("question")
        if not isinstance(question, str) or not 0 < len(question.strip()) <= 500:
            raise ValueError("Enter a question of up to 500 characters.")
        question = question.strip()
        with self.lock:
            host = self.host
            if question in QUICK_READS and "capture_id" not in data:
                intent, window_ms, metrics = QUICK_READS[question]
                snapshot = (self.board(intent) if window_ms is None else
                            self.board("stream_observations", window_ms=window_ms))
                return dict(host=host, snapshot=snapshot, answer=live_answer(snapshot, intent, metrics))
            if getattr(self, "lm_supported", False):
                options = {"question": question}
                if "capture_id" in data:
                    options["capture_id"] = data["capture_id"]
                job = self.board("lm_start", **options)
            else:
                job = None
                if "capture_id" in data:
                    raise ValueError("This board does not support saved incident replay.")
            if job is not None:
                snapshot = job["snapshot"]
            elif self.evidence_supported:
                snapshot = self.board("stream_evidence", window_ms=60000)
            else:
                # Earlier kit services expose measurements and inference as two timestamped reads.
                snapshot = self.board("stream_observations", window_ms=60000)
                if snapshot.get("fresh") and snapshot.get("session"):
                    scored = self.board("stream_forecast", session=snapshot["session"])
                    if scored.get("session") != snapshot["session"] or not scored.get("fresh"):
                        raise ValueError("The live source changed during capture. Ask again.")
                    snapshot["forecast"] = scored.get("forecast")
                    snapshot["forecast_reading"] = scored.get("latest")
        if job is not None:
            if data["action"] == "ask_begin":
                return {**job, "host": host}
            while job["state"] == "running":
                sleep(1)
                with self.lock:
                    if self.host != host:
                        raise RuntimeError("The board changed during diagnosis.")
                    job = self.board("lm_result", job_id=job["job_id"])
            if job["state"] != "complete":
                raise RuntimeError(job.get("error", "Board inference did not complete."))
            return dict(host=host, snapshot=snapshot, answer=job["answer"],
                        analyst_model=job["model"], analyst_location="PYNQ-Z2",
                        capture_id=job["capture_id"], active=job["active"], baseline=job.get("baseline"),
                        captured_at_ms=job.get("captured_at_ms"))
        if not snapshot.get("latest") or not snapshot.get("fresh"):
            return dict(host=host, snapshot=snapshot, answer=live_answer(snapshot, "stream_status", []))
        diagnosis = analyze_live_evidence(question, analyst_evidence(snapshot))
        if diagnosis.get("errors"):
            raise RuntimeError(" ".join(diagnosis["errors"]))
        return dict(host=host, snapshot=snapshot, answer=diagnosis["answer"],
                    analyst_model=diagnosis["model"], analyst_location="Mac")

    def synthesize_dataset(self, synthesis_id, epoch, data, context):
        dataset_id = data["dataset_id"]
        try:
            if not context["ready"]:
                with self.lock:
                    self.require_synthesis_connection(epoch)
                    job = self.board("lm_dataset_calibrate", dataset_id=dataset_id)
                    self.syntheses[synthesis_id]["job_id"] = job["job_id"]
                deadline = monotonic() + 1200
                while monotonic() < deadline:
                    with self.lock:
                        self.require_synthesis_connection(epoch)
                        result = self.board("lm_result", job_id=job["job_id"])
                        if result["state"] == "complete":
                            context = self.board("lm_dataset_calibration", dataset_id=dataset_id,
                                                 target_row=data["target_row"])
                            break
                        if result["state"] == "failed":
                            raise RuntimeError(result.get("error", "Board calibration failed."))
                        self.syntheses[synthesis_id].update(
                            {key: result[key] for key in ("partial_answer", "fabric_projections",
                                                         "accelerated_projections") if key in result})
                    sleep(1)
                if not context["ready"]:
                    raise RuntimeError("No calibration activation was returned for this dataset.")
            with self.lock:
                self.require_synthesis_connection(epoch)
                self.syntheses[synthesis_id]["stage"] = "authoring"
            response = propose(data["instruction"], context)
            with self.lock:
                self.require_synthesis_connection(epoch)
                self.syntheses[synthesis_id]["stage"] = "saving"
            # Keep connection identity and the durable save in one critical section.
            with self.lock:
                self.require_synthesis_connection(epoch)
                dataset = self.board("lm_dataset_save", dataset_id=dataset_id, name=data["name"],
                                     selected_model=context["selected_model"], target_row=context["target_row"],
                                     response=response, base_sha256=context["base_sha256"])
                self.syntheses[synthesis_id].update(state="complete", stage="ready", dataset=dataset)
        except Exception as exc:
            with self.lock:
                if epoch == self.connection_epoch:
                    self.syntheses[synthesis_id].update(state="failed", stage="failed", error=str(exc))

    def require_synthesis_connection(self, epoch):
        if self.host is None or self.connection_epoch != epoch:
            raise RuntimeError("The paired board connection changed during synthesis.")

    def dataset_action(self, action, data):
        required, optional = {
            "dataset_upload": ({"csv", "capacity"}, {"name"}),
            "dataset_models": ({"dataset_id"}, set()),
            "dataset_select": ({"dataset_id", "name"}, set()),
            "dataset_ask": ({"dataset_id", "question"}, {"stage"}),
            "dataset_synthesize": ({"dataset_id", "name", "target_row", "instruction"}, set()),
            "dataset_synthesis_progress": ({"synthesis_id"}, set()),
        }[action]
        fields = {key: value for key, value in data.items() if key != "action"}
        if required - fields.keys() or fields.keys() - (required | optional):
            raise ValueError("Invalid dataset action fields.")
        if action == "dataset_synthesis_progress":
            synthesis_id = fields["synthesis_id"]
            if not isinstance(synthesis_id, str) or synthesis_id not in self.syntheses:
                raise ValueError("Unknown synthesis job.")
            return dict(host=self.host, **self.syntheses[synthesis_id])
        if action != "dataset_synthesize":
            return dict(host=self.host, **self.board("lm_" + action, **fields))
        if any(job["state"] == "running" for job in self.syntheses.values()):
            raise ValueError("A model synthesis is in progress.")
        instruction = fields["instruction"]
        name = fields["name"]
        if not isinstance(instruction, str) or not 0 < len(instruction.strip()) <= 1000:
            raise ValueError("Describe the model edit in 1–1000 characters.")
        if (not isinstance(name, str) or not 0 < len(name) <= 80 or name != name.strip()
                or not name.isprintable() or name.casefold() == "original"):
            raise ValueError("Choose a new model name of up to 80 characters.")
        context = self.board("lm_dataset_calibration", dataset_id=fields["dataset_id"],
                             target_row=fields["target_row"])
        if len(self.syntheses) >= 32:
            del self.syntheses[next(iter(self.syntheses))]
        synthesis_id = uuid4().hex
        job = dict(synthesis_id=synthesis_id, dataset_id=fields["dataset_id"], state="running",
                   stage="authoring" if context["ready"] else "calibrating")
        self.syntheses[synthesis_id] = job
        try:
            Thread(target=self.synthesize_dataset, args=(synthesis_id, self.connection_epoch, fields, context),
                   daemon=True).start()
        except RuntimeError:
            del self.syntheses[synthesis_id]
            raise
        return dict(host=self.host, **job)

    def action(self, data):
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        action = data.get("action")
        if not isinstance(action, str):
            raise ValueError("Choose an action.")
        if action == "lm_author":
            with self.lock:
                address = self.host
                context = self.board("lm_row", target_row=data.get("target_row"))
            response = propose(data.get("instruction"), context)
            with self.lock:
                if not address or self.host != address:
                    raise ValueError("Reconnect the original board to author this version.")
                return dict(host=self.host, **self.board("lm_save", name=data.get("name"),
                            target_row=context["target_row"], response=response,
                            base_sha256=context["base_sha256"]))
        if action in ("ask_live", "ask_begin"):
            return self.query_live(data)
        if action == "import_data":
            if set(data) != {"action", "name", "text"}:
                raise ValueError("Importing data takes a name and the file text.")
            return self.import_data(data["name"], data["text"])
        if action == "load_sample":
            sample = next((item for item in SAMPLES if item["id"] == data.get("id")), None)
            if sample is None:
                raise ValueError("Unknown sample.")
            text = sample_csv(sample["id"])
            result = self.import_data(sample["file"], text, remember=False)
            if result["profile"].get("queue_ready"):
                result["queue_csv"] = text
            return result
        if action == "data_ask_begin":
            return self.begin_data_job(data)
        if action == "data_ask_progress":
            return self.progress_data_job(data)
        if action == "data_ask":
            return self.ask_data(data)
        with self.lock:
            if action in ("dataset_upload", "dataset_models", "dataset_select", "dataset_ask",
                          "dataset_synthesize", "dataset_synthesis_progress"):
                return self.dataset_action(action, data)
            if action == "answer_progress":
                return {**self.board("lm_result", job_id=data.get("job_id")), "host": self.host}
            if action == "lm_captures":
                if set(data) - {"action", "before"}:
                    raise ValueError("Invalid saved-capture listing request.")
                fields = {"before": data["before"]} if "before" in data else {}
                return {**self.board("lm_captures", **fields), "host": self.host}
            if action == "discover":
                return {"boards": self.transport.discover()}
            if action == "usb_connect":
                if self.host is not None:
                    return {"host": self.host, "status": self.board("status")}
                reconnect = getattr(self.transport, "reconnect_usb", None)
                status = reconnect() if reconnect else None
                if status:
                    self.host = "127.0.0.1"
                    self.evidence_supported = bool(status.get("stream_evidence_supported"))
                    self.lm_supported = bool(status.get("lm_supported"))
                return {"host": self.host, "status": status}
            if action == "connect":
                host = data.get("host")
                if not isinstance(host, str) or not host.strip() or len(host) > 255:
                    raise ValueError("Choose a board or enter its network address.")
                self.disconnect()
                try:
                    status = self.transport.connect(host.strip())
                    if status.get("ok") is False:
                        raise ValueError(status.get("error", "The board rejected the connection."))
                except Exception:
                    self.disconnect()
                    raise
                self.evidence_supported = bool(status.get("stream_evidence_supported"))
                self.lm_supported = bool(status.get("lm_supported"))
                self.host = host.strip()
                return {"host": self.host, "status": status}
            if action == "disconnect":
                stop_usb = getattr(self.transport, "stop_usb", None)
                if stop_usb:
                    stop_usb()
                self.disconnect()
                return {"host": None, "status": None}
            if action == "status":
                return {"host": self.host, "status": self.board("status") if self.host else None}
            if action == "query":
                return self.analyze(data)
            if action == "stream_status":
                if set(data) != {"action"}:
                    raise ValueError("Live status takes no parameters.")
                return {"host": self.host, "snapshot": self.board("stream_status")}
            if action in ("lm_versions", "lm_save", "lm_select", "lm_restore"):
                fields = {
                    "lm_versions": (),
                    "lm_save": ("name", "target_row", "response"),
                    "lm_select": ("name",),
                    "lm_restore": (),
                }[action]
                if set(data) != {"action", *fields}:
                    raise ValueError("Invalid language-model version request.")
                return {"host": self.host, **self.board(action, **{field: data[field] for field in fields})}
            if action in ("apply", "rollback"):
                params = {"version": data.get("version")} if action == "apply" else {}
                return {"host": self.host, "status": self.board(action, **params)}
            raise ValueError("Unknown kit action.")


class KitHandler(BaseHTTPRequestHandler):
    timeout = REQUEST_TIMEOUT_SECONDS

    def log_message(self, format, *args):
        pass

    def reply(self, status, body, content_type="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("Host") not in self.server.allowed_hosts:
            return self.reply(403, {"error": "Open the app on this Mac."})
        if self.path == "/":
            return self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/samples":
            return self.reply(200, {"samples": SAMPLES})
        if self.path == "/api/mcp":
            from .mcp_server import mcp_config

            return self.reply(200, mcp_config())
        if self.path == "/api/models":
            return self.reply(200, {"models": self.server.accounts.models.list()})
        if self.path == "/static/logo.png":
            if not LOGO.is_file():
                return self.reply(404, {"error": "No logo."})
            return self.reply(200, LOGO.read_bytes(), "image/png")
        if self.path == "/auth/login":
            referer = self.headers.get("Referer")
            fetch_site = self.headers.get("Sec-Fetch-Site")
            if (referer and not referer.startswith(f"http://{self.headers.get('Host')}/")) or fetch_site in ("cross-site", "same-site"):
                return self.reply(403, {"error": "Start sign-in from the PYNQ Process app."})
            try:
                url = self.server.accounts.login_url(f"http://{self.headers.get('Host')}")
            except ValueError as exc:
                return self.reply(400, {"error": str(exc)})
            self.send_response(302)
            self.send_header("Location", url)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.split("?")[0] == CALLBACK_PATH:
            query = parse_qs(urlsplit(self.path).query)
            try:
                self.server.accounts.finish_login(query.get("code", [""])[0], query.get("state", [""])[0],
                                                  query.get("error", [""])[0])
            except ValueError as exc:
                return self.reply(400, {"error": str(exc)})
            except RuntimeError as exc:
                return self.reply(503, {"error": str(exc)})
            return self.reply(200, SIGNED_IN_PAGE, "text/html; charset=utf-8")
        return self.reply(404, {"error": "Page unavailable."})

    def do_POST(self):
        host = self.headers.get("Host")
        if (host not in self.server.allowed_hosts or self.headers.get("Origin") != f"http://{host}"
                or self.headers.get_content_type() != "application/json"):
            return self.reply(403, {"error": "Use the PYNQ Process app on this Mac."})
        if self.path not in ("/api/action", "/api/account"):
            return self.reply(404, {"error": "Unknown endpoint."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY_BYTES:
                return self.reply(413, {"error": "This request is empty or larger than 20 MB. Import a smaller file."})
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError("The request ended early. Try again.")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            if self.path == "/api/account":
                result = self.server.accounts.action(data)
            elif data.get("action") == "export":
                result = export_action(data, self.server.accounts)
            else:
                result = self.server.kit.action(data)
            self.reply(200, result)
        except (ValueError, TypeError, RecursionError) as exc:
            self.reply(400, {"error": str(exc) if isinstance(exc, ValueError) else "This request could not be read."})
        except (OSError, RuntimeError) as exc:
            self.reply(503, {"error": str(exc)})
        except Exception:
            self.reply(500, {"error": "Something went wrong on this Mac. Try again."})


class KitServer(ThreadingHTTPServer):
    def server_close(self):
        super().server_close()
        if hasattr(self, "kit"):
            with self.kit.lock:
                self.kit.disconnect()


DEFAULT_PORT = 53530
SIGN_IN_PORTS = 5


def make_server(host="127.0.0.1", port=DEFAULT_PORT, data_dir=DATA_DIR):
    if host != "127.0.0.1":
        raise ValueError("The Mac app must use the local loopback address.")
    simulated = os.environ.get("PYNQ_SIM_BOARD") == "1"
    transport = SimulatedBoard() if simulated else BoardTransport()
    server = None
    for candidate in ([port + offset for offset in range(SIGN_IN_PORTS)] if port else []) + [0]:
        try:
            server = KitServer((host, candidate), KitHandler)
            break
        except OSError:
            if not candidate:
                raise
    server.kit = Kit(transport, store=DatasetStore(data_dir))
    load_env()
    server.accounts = Accounts()
    if simulated:
        server.kit.action({"action": "connect", "host": SIM_HOST})
    server.allowed_hosts = {f"127.0.0.1:{server.server_port}", f"localhost:{server.server_port}"}
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open", action="store_true", help="Open the app in the default browser")
    args = parser.parse_args()
    with make_server(port=args.port) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"PYNQ Process at {url}", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
