"""Loopback client for physical tests through the installed Mac app."""
import json
from time import monotonic, sleep
import urllib.request


class MacBackend:
    def __init__(self, port):
        self.origin = f"http://127.0.0.1:{port}"
        self.client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def action(self, action_name, **fields):
        request = urllib.request.Request(
            self.origin + "/api/action", data=json.dumps({"action": action_name, **fields}).encode(),
            headers={"Content-Type": "application/json", "Origin": self.origin},
        )
        with self.client.open(request, timeout=20) as response:
            result = json.load(response)
        if result.get("host") != "127.0.0.1":
            raise RuntimeError("The Mac backend is not paired through the USB loopback.")
        return result

    def ready_stream(self):
        deadline = monotonic() + 30
        while monotonic() < deadline:
            snapshot = self.action("stream_status")["snapshot"]
            if snapshot["state"] == "ready" and snapshot["fresh"]:
                return snapshot
            sleep(0.3)
        raise RuntimeError("The Mac app did not see a fresh forecasting window.")
