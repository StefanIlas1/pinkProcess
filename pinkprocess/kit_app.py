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
from .kit_transport import BoardTransport


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


def live_answer(snapshot, intent, metrics):
    state = snapshot["state"]
    latest = snapshot.get("latest")
    if latest is None:
        return "The board has no telemetry readings for this stream."
    if not snapshot["fresh"]:
        return "The board's telemetry is stale. It last saw {} busy and {} waiting.".format(
            latest["busy"], latest["waiting"])
    if intent == "stream_status":
        if metrics == ["busy"]:
            return f'{latest["busy"]} of {snapshot["capacity"]} connections are busy.'
        if metrics == ["waiting"]:
            return f'Waiting requests: {latest["waiting"]}.'
        return f'{latest["busy"]} of {snapshot["capacity"]} connections busy, with {latest["waiting"]} requests waiting.'
    if intent in ("peak", "trend"):
        observations = snapshot["observations"]
        if intent == "trend" and observations["sample_count"] < 2:
            return "The board needs at least two readings to report a trend."
        labels = {"busy": "Busy connections", "waiting": "Waiting requests"}
        statements = []
        for metric in metrics:
            if intent == "peak":
                peak = observations[f"peak_{metric}"]
                statements.append(f'{labels[metric]} peaked at {peak[metric]} at {peak["time_ms"] / 1000:g} s.')
            else:
                first, last = observations["first"][metric], observations["latest"][metric]
                if first == last:
                    statements.append(f"{labels[metric]} started and ended at {first}.")
                else:
                    direction = "rose" if last > first else "fell"
                    statements.append(f"{labels[metric]} {direction} from {first} to {last}.")
        span = (observations["end_ms"] - observations["start_ms"]) / 1000
        statements.append(f'Based on {observations["sample_count"]} readings spanning {span:g} seconds.')
        return " ".join(statements)
    forecast = snapshot.get("forecast")
    if not forecast:
        if state == "warming":
            return "The board is collecting five seconds of continuous readings for a forecast."
        return "The FPGA forecast is unavailable for this stream."
    if forecast["model"] == "queue_outlook":
        persists, grows = [score >= 0 for score in forecast["scores"]]
        n = latest["waiting"]
        persistence = "remain nonempty" if persists else "drain at least once"
        growth = "more than" if grows else "at most"
        return (f"The FPGA predicts the queue will {persistence} over the next ten seconds. "
                f"It expects {growth} the current {n} waiting at the end of that period.")
    if forecast["model"] == "pool_wait_logistic":
        if forecast["alert"]:
            return "The FPGA warns that requests may start waiting within the next ten seconds."
        return "The FPGA issued no queue-onset warning for the next ten seconds."
    return "The board returned an unsupported forecast model."


def analyst_evidence(snapshot):
    """Give the analyst a readable view of the timestamped data used for the answer."""
    fields = ("origin", "source", "capacity", "state", "latest", "observations", "limitations",
              "forecast_reading")
    evidence = {key: snapshot[key] for key in fields if key in snapshot}
    latest = snapshot["latest"]
    capacity = snapshot.get("capacity")
    if capacity:
        evidence["observed_now"] = (
            f'{latest["busy"]} of {capacity} connections occupied '
            f'({100 * latest["busy"] / capacity:.1f}%); {latest["waiting"]} requests waiting.'
        )
    observations = snapshot.get("observations") or {}
    peak = observations.get("peak_waiting")
    if peak:
        evidence["observed_peak"] = (
            f'{peak["waiting"]} requests were waiting at the window maximum, '
            f'{(latest["time_ms"] - peak["time_ms"]) / 1000:g} seconds before the latest reading. '
            f'{peak["busy"]} connections were occupied at that instant.'
        )
    forecast = snapshot.get("forecast")
    if forecast:
        scored = {**snapshot, "latest": snapshot.get("forecast_reading") or latest}
        evidence["forecast"] = {key: forecast[key] for key in ("version", "time_ms", "horizon_ms")
                                if key in forecast}
        evidence["forecast_interpretation"] = live_answer(scored, "stream_forecast", [])
    return evidence


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
        if set(data) - {"action", "question"}:
            raise ValueError("Live questions accept only question text. The board supplies the telemetry.")
        question = data.get("question")
        if not isinstance(question, str) or not 0 < len(question.strip()) <= 500:
            raise ValueError("Enter a question of up to 500 characters.")
        question = question.strip()
        with self.lock:
            host = self.host
            if question in QUICK_READS:
                intent, window_ms, metrics = QUICK_READS[question]
                snapshot = (self.board(intent) if window_ms is None else
                            self.board("stream_observations", window_ms=window_ms))
                return dict(host=host, snapshot=snapshot, answer=live_answer(snapshot, intent, metrics))
            if self.evidence_supported:
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
        if action == "ask_live":
            return self.query_live(data)
        with self.lock:
            if action == "discover":
                return {"boards": self.transport.discover()}
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
                self.host = host.strip()
                return {"host": self.host, "status": status}
            if action == "disconnect":
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
