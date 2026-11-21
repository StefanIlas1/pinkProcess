"""Serve the local pinkProcess browser operator over a loopback connection."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from threading import Lock

from lm_reference.patch_operator import query_prompt, stop_session

PAGE = Path(__file__).with_name("operator.html")


class Operator:
    def __init__(self, worker, candidates):
        self.worker = worker
        self.candidates = candidates
        self.lock = Lock()
        self.ready = self.read()["result"]

    def read(self):
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError("The operator exited. Check the terminal and restart the session.")
        return json.loads(line)

    def send(self, command, candidate=None, prompt=None):
        words = command
        if command in ("preview", "apply"):
            if candidate not in self.candidates:
                raise ValueError("Select an available patch bundle.")
            words += " " + shlex.quote(str(self.candidates[candidate]))
        elif command == "generate" and prompt is not None:
            words += " " + shlex.quote(query_prompt(prompt))
        elif command not in ("status", "verify", "generate", "rollback", "quit"):
            raise ValueError("Unknown operator action.")
        with self.lock:
            self.worker.stdin.write(words + "\n")
            self.worker.stdin.flush()
            response = self.read()
            if response["command"] != command:
                raise RuntimeError("The operator returned an unexpected response.")
            return response


class Handler(BaseHTTPRequestHandler):
    def send_data(self, status, data, content_type="application/json"):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.headers.get("Host") not in self.server.allowed_hosts:
            self.send_data(403, {"error": "Open the operator using its loopback URL."})
        elif self.path == "/":
            self.send_data(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            operator = self.server.operator
            try:
                status = operator.send("status")
                self.send_data(200, {
                    "board": operator.ready,
                    "status": status["result"],
                    "candidates": list(operator.candidates),
                })
            except (OSError, RuntimeError) as exc:
                self.send_data(503, {"error": str(exc)})
        else:
            self.send_data(404, {"error": "Page unavailable."})

    def do_POST(self):
        host = self.headers.get("Host")
        if (host not in self.server.allowed_hosts or self.headers.get("Origin") != f"http://{host}"
                or self.headers.get_content_type() != "application/json"):
            self.send_data(403, {"error": "Use the local operator page."})
            return
        if self.path != "/api/action":
            self.send_data(404, {"error": "Action endpoint unavailable."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2048:
                raise ValueError("Invalid action request length.")
            request = json.loads(self.rfile.read(length))
            command = request["action"]
            if command not in ("preview", "apply", "generate", "verify", "rollback"):
                raise ValueError("Unknown operator action.")
            response = self.server.operator.send(command, request.get("candidate"), request.get("prompt"))
            self.send_data(200 if response["ok"] else 422, response)
        except (ValueError, KeyError, TypeError) as exc:
            self.send_data(400, {"error": str(exc)})
        except (OSError, RuntimeError) as exc:
            self.send_data(503, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "tokenizer"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--bitstream", type=Path, help="Use the physical PYNQ-Z2")
    backend.add_argument("--software", action="store_true", help="Run the classifier on this computer")
    parser.add_argument("--patch", type=Path, action="append", default=[])
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    candidates = {candidate.stem: candidate.resolve() for candidate in args.patch}
    if len(candidates) != len(args.patch) or any(not path.is_file() for path in candidates.values()):
        parser.error("Each --patch must be an existing file with a unique name.")

    signal.signal(signal.SIGTERM, stop_session)
    signal.signal(signal.SIGHUP, stop_session)
    command = [sys.executable, "-m", "lm_reference.patch_operator", "serve", "--fixed"]
    for name in ("source", "model", "tokenizer", "cc"):
        command.extend((f"--{name}", str(getattr(args, name))))
    command.extend(("--bitstream", str(args.bitstream)) if args.bitstream else ("--software",))
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server, subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        ) as worker:
            try:
                server.operator = Operator(worker, candidates)
                port = server.server_port
                server.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
                print(f"pinkProcess operator at http://127.0.0.1:{port}", flush=True)
                server.serve_forever()
            finally:
                # EOF lets the board process finish its current command and restore its row.
                worker.stdin.close()
                worker.stdout.close()
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
