"""Mac operator for the board-resident pinkProcess telemetry model."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
import webbrowser

from .live_analyst import analyze as analyze_live_evidence
from .backlog import VERSION as BACKLOG_VERSION
from .kit_dataset import DEMO_CAPACITY, demo_csv, prepare_dataset
from .astra_edit import propose
from .kit_transport import BoardTransport
from .kit_evidence import analyst_evidence, live_answer
from time import sleep


PAGE = Path(__file__).with_name("kit.html")
QUESTIONS = {
    "latest": "Will requests start waiting in the next ten seconds?",
    "backlog": "Will the current queue persist or grow over the next ten seconds?",
    "warnings": "Which readings triggered a ten-second queue warning?",
}
MAX_BODY_BYTES = 2 * 1024 * 1024
DIAGNOSIS_EXAMPLE = "What looks wrong with our system, and what should I check next?"
QUICK_READS = {
    "How many requests are waiting now?": ("stream_status", None, ["waiting"]),
    "What was the peak queue in the last 30 seconds?": ("peak", 30000, ["waiting"]),
    "Is the queue growing over the last 10 seconds?": ("trend", 10000, ["waiting"]),
    "What does the FPGA predict for the next ten seconds?": ("stream_forecast", None, []),
}
LIVE_EXAMPLES = (DIAGNOSIS_EXAMPLE, *QUICK_READS)


class Kit:
    def __init__(self, transport, prepare=prepare_dataset):
        self.prepare = prepare
        self.transport = transport
        self.host = None
        self.evidence_supported = False
        self.lock = Lock()

    def disconnect(self):
        self.host = None
        self.transport.close()

    def board(self, action, **params):
        if self.host is None:
            raise ValueError("Connect a board first.")
        try:
            result = self.transport.request(action, **params)
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

    def action(self, data):
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        action = data.get("action")
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
        with self.lock:
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
                if not isinstance(host, str) or not host.strip():
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
        if self.path == "/api/setup":
            return self.reply(200, {"questions": QUESTIONS, "capacity": DEMO_CAPACITY, "csv": demo_csv()})
        if self.path == "/api/live-setup":
            return self.reply(200, {"examples": LIVE_EXAMPLES})
        return self.reply(404, {"error": "Page unavailable."})

    def do_POST(self):
        host = self.headers.get("Host")
        if (host not in self.server.allowed_hosts or self.headers.get("Origin") != f"http://{host}"
                or self.headers.get_content_type() != "application/json"):
            return self.reply(403, {"error": "Use the pinkProcess app on this Mac."})
        if self.path != "/api/action":
            return self.reply(404, {"error": "Unknown endpoint."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY_BYTES:
                return self.reply(413, {"error": "The request exceeds the 2 MiB limit or is empty."})
            data = json.loads(self.rfile.read(size))
            result = self.server.kit.action(data)
            self.reply(200, result)
        except (ValueError, TypeError) as exc:
            self.reply(400, {"error": str(exc)})
        except (OSError, RuntimeError) as exc:
            self.reply(503, {"error": str(exc)})


class KitServer(ThreadingHTTPServer):
    def server_close(self):
        super().server_close()
        if hasattr(self, "kit"):
            with self.kit.lock:
                self.kit.disconnect()


def make_server(host="127.0.0.1", port=0):
    if host != "127.0.0.1":
        raise ValueError("The Mac app must use the local loopback address.")
    transport = BoardTransport()
    server = KitServer((host, port), KitHandler)
    server.kit = Kit(transport)
    server.allowed_hosts = {f"127.0.0.1:{server.server_port}", f"localhost:{server.server_port}"}
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--open", action="store_true", help="Open the app in the default browser")
    args = parser.parse_args()
    with make_server(port=args.port) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"pinkProcess at {url}", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
