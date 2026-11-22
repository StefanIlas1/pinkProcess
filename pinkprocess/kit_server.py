"""Board service for querying the persistently installed FPGA model."""

import argparse
from contextlib import ExitStack
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import socket
import ssl
from threading import Condition, Event, Thread

from fpga.check_classifier import board_lock
from lm_reference.live_demo import PynqClassifier
from .kit_transport import DISCOVER, PORT, discovery_proof
from .telemetry_device import TelemetryDevice


class InferenceLock:
    """Let waiting telemetry run before inference can submit its next tile."""

    def __init__(self):
        self.condition = Condition()
        self.next_ticket = 0
        self.serving = 0

    def __enter__(self):
        with self.condition:
            ticket = self.next_ticket
            self.next_ticket += 1
            self.condition.wait_for(lambda: self.serving == ticket)

    def __exit__(self, *_):
        with self.condition:
            self.serving += 1
            self.condition.notify_all()


def advertise(token, stopped):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("", PORT))
        sock.settimeout(1)
        while not stopped.is_set():
            try:
                data, peer = sock.recvfrom(128)
            except TimeoutError:
                continue
            if data.startswith(DISCOVER) and len(data) == len(DISCOVER) + 32:
                sock.sendto(discovery_proof(token, data[len(DISCOVER):]).encode(), peer)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path not in ("/v1/query", "/v1/telemetry"):
            return self.reply(404, "Unknown endpoint.")
        if not hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                   self.server.authorization):
            return self.reply(403, "Kit pairing rejected.")
        try:
            length = int(self.headers.get("Content-Length", 0))
            if self.headers.get_content_type() != "application/json" or not 0 < length <= 65536:
                raise ValueError("Expected a JSON request of at most 64 KiB.")
            command = json.loads(self.rfile.read(length))
            if (self.path == "/v1/query" and isinstance(command, dict)
                    and isinstance(command.get("op"), str) and command["op"].startswith("lm_")):
                language = getattr(self.server, "language", None)
                if language is None:
                    raise ValueError("The kit has no installed language model.")
                return self.reply(200, language.handle(command))
            with self.server.model_lock:
                if self.path == "/v1/telemetry":
                    result = self.server.model.ingest(command)
                else:
                    result = self.server.model.handle(command)
                    if command.get("op") == "status":
                        result["lm_supported"] = getattr(self.server, "language", None) is not None
        except (UnicodeError, ValueError, TypeError, KeyError) as exc:
            return self.reply(422, str(exc))
        except (RuntimeError, OSError) as exc:
            return self.reply(503, str(exc))
        self.reply(200, result)

    def reply(self, code, data):
        if isinstance(data, str):
            data = {"ok": False, "error": data}
        body = json.dumps(data, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class KitServer(ThreadingHTTPServer):
    daemon_threads = False
    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(5)
        return self.tls.wrap_socket(connection, server_side=True, do_handshake_on_connect=False), address


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--bitstream", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--demo-source", choices=("sqlite",))
    parser.add_argument("--language-model", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--projection-overlay", action="store_true",
                        help="Enable the experimental FLAN DDR projection accelerator.")
    parser.add_argument("--projection-scope", choices=("query", "all"), default="query",
                        help="Offload the first self-attention queries (default) or all learned projections.")
    parser.add_argument("--projection-queue", action="store_true",
                        help="Use a projection overlay explicitly known to have the command queue.")
    args = parser.parse_args(argv)
    if bool(args.language_model) != bool(args.manifest_sha256):
        parser.error("Supply both the language model directory and its pinned manifest hash.")
    if args.projection_overlay and args.language_model is None:
        parser.error("Supply the language model when enabling the projection overlay.")
    if args.projection_scope == "all" and not args.projection_overlay:
        parser.error("--projection-scope all requires --projection-overlay.")
    if args.projection_queue and (not args.projection_overlay or args.projection_scope != "all"):
        parser.error("--projection-queue requires --projection-overlay --projection-scope all.")
    return args


def main():
    args = parse_args()
    token = (args.identity / "token").read_text().strip()
    if len(token) != 64:
        raise ValueError("Invalid kit token.")
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(args.identity / "board.crt", args.identity / "board.key")
    stopped = Event()
    with board_lock(), ExitStack() as resources:
        model = TelemetryDevice(PynqClassifier(args.bitstream), args.profiles, args.state_file)
        with KitServer(("", PORT), Handler) as server:
            server.tls = tls
            server.model = model
            server.model_lock = InferenceLock()
            if args.language_model:
                from lm_reference.fabric import TiledMatvec
                from lm_reference.flan import Flan
                from lm_reference.weight_versions import WeightVersionStore
                from .board_language import BoardLanguage

                state_dir = args.state_file.parent
                versions = WeightVersionStore.from_flan(state_dir / "lm-versions.sqlite", args.language_model,
                                                        manifest_sha256=args.manifest_sha256)
                query_projection = None
                if args.projection_overlay:
                    from lm_reference.projection_overlay import OverlayProjection, projection_mmio

                    mmio = projection_mmio(model.classifier.overlay, required=args.projection_scope == "all")
                    if mmio is not None:
                        from pynq import allocate

                        query_projection = OverlayProjection(mmio, allocate, scope=args.projection_scope,
                                                            queue=args.projection_queue)
                        resources.callback(query_projection.close)
                runtime = Flan(args.language_model, TiledMatvec(model.classifier, lock=server.model_lock),
                               query_projection=query_projection)

                def snapshot():
                    with server.model_lock:
                        return model.handle(dict(op="stream_evidence", window_ms=60000))

                server.language = BoardLanguage(runtime, versions, state_dir, snapshot)
            server.authorization = ("Bearer " + token).encode()
            discovery = Thread(target=advertise, args=(token, stopped), daemon=True)
            discovery.start()
            if args.demo_source:
                from .board_demo import run_demo

                def publish(payload):
                    with server.model_lock:
                        return model.ingest(payload)

                demo = Thread(target=run_demo, args=(publish, stopped), daemon=True)
                demo.start()
            def stop(*_):
                stopped.set()
                Thread(target=server.shutdown, daemon=True).start()
            signal.signal(signal.SIGTERM, stop)
            try:
                server.serve_forever()
            finally:
                stopped.set()


if __name__ == "__main__":
    main()
