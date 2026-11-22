"""Serve the pinkProcess app and the cartridge API from the laptop or from the board itself."""

import json
from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from . import board
from .resources import APP
from .runtime import Device

PAGE = APP / "index.html"


def make_handler(device: Device):
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, document: dict, status: int = 200) -> None:
            body = json.dumps(document).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/api/state":
                with lock:
                    state = device.describe()
                self.send_json(state)
            elif self.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self.path != "/api/ask":
                self.send_json({"error": "not found"}, 404)
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                question = str(request["question"])[:500]
                cartridge = request.get("cartridge", "auto")
                if cartridge != "auto" and cartridge not in device.library:
                    raise KeyError(cartridge)
            except (ValueError, KeyError, TypeError):
                self.send_json({"error": "bad request"}, 400)
                return
            with lock:
                reply = device.ask(question, cartridge)
                reply["switch_microseconds"] = round(device.switch_microseconds, 1)
            self.send_json(reply)

        def log_message(self, *args) -> None:
            pass

    return Handler


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 on the board")
    parser.add_argument("--port", type=int, default=5320)
    args = parser.parse_args()
    device = board.BoardDevice() if board.available() else Device()
    print(f"pinkProcess on http://{args.host}:{args.port}  backend: {device.backend}")
    ThreadingHTTPServer((args.host, args.port), make_handler(device)).serve_forever()


if __name__ == "__main__":
    main()
