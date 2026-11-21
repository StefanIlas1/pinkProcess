"""Board service for querying the persistently installed FPGA model."""

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import socket
import ssl
from threading import Event, Lock, Thread

from fpga.check_classifier import board_lock
from lm_reference.live_demo import PynqClassifier
from .kit_transport import DISCOVER, PORT, discovery_proof
from .telemetry_device import TelemetryDevice


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
            with self.server.model_lock:
                if self.path == "/v1/telemetry":
                    result = self.server.model.ingest(command)
                else:
                    result = self.server.model.handle(command)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--bitstream", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--demo-source", choices=("sqlite",))
    args = parser.parse_args()
    token = (args.identity / "token").read_text().strip()
    if len(token) != 64:
        raise ValueError("Invalid kit token.")
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(args.identity / "board.crt", args.identity / "board.key")
    stopped = Event()
    with board_lock():
        model = TelemetryDevice(PynqClassifier(args.bitstream), args.profiles, args.state_file)
        with KitServer(("", PORT), Handler) as server:
            server.tls = tls
            server.model = model
            server.model_lock = Lock()
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
