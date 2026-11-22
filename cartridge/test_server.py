"""Concurrent HTTP requests must observe complete cartridge operations."""

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from .server import make_handler


class CartridgeServerTest(unittest.TestCase):
    def test_state_and_asks_serialize_around_weight_switch(self):
        entered = threading.Event()
        release = threading.Event()
        observed = threading.Event()

        class Device:
            library = {"health": None, "wild": None}
            switch_microseconds = 0.0

            def ask(self, question, cartridge):
                if cartridge == "health":
                    entered.set()
                    if not release.wait(3):
                        raise RuntimeError("Timed out waiting for test cartridge")
                else:
                    observed.set()
                self.switch_microseconds = {"health": 12, "wild": 34}[cartridge]
                return {"cartridge": cartridge}

            def describe(self):
                observed.set()
                return {"active": "health"}

        with ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Device())) as server:
            worker = threading.Thread(target=server.serve_forever)
            worker.start()
            base = f"http://127.0.0.1:{server.server_port}"

            def request(path, body=None):
                data = json.dumps(body).encode() if body is not None else None
                with urlopen(Request(base + path, data), timeout=3) as response:
                    return json.load(response)

            try:
                with ThreadPoolExecutor(max_workers=3) as pool:
                    first = pool.submit(request, "/api/ask", {"question": "question", "cartridge": "health"})
                    try:
                        self.assertTrue(entered.wait(2))
                        second = pool.submit(request, "/api/ask", {"question": "question", "cartridge": "wild"})
                        state = pool.submit(request, "/api/state")
                        self.assertFalse(observed.wait(0.1))
                    finally:
                        release.set()
                    self.assertEqual(first.result()["switch_microseconds"], 12)
                    self.assertEqual(second.result()["switch_microseconds"], 34)
                    self.assertEqual(state.result(), {"active": "health"})
            finally:
                server.shutdown()
                worker.join()


if __name__ == "__main__":
    unittest.main()
