import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, build_opener, HTTPRedirectHandler, urlopen

from .kit_accounts import Accounts
from .kit_app import DEFAULT_PORT, Kit, make_server
from .kit_data import SAMPLES
from .kit_export import export_session
from .kit_models import BUILT_IN, ModelStore


class NoBoard:
    def close(self):
        pass


class ModelStoreTest(unittest.TestCase):
    def test_library_fields(self):
        with TemporaryDirectory() as directory:
            models = ModelStore(Path(directory) / "models.json").list()
        self.assertEqual(len(models), len(BUILT_IN))
        for model in models:
            self.assertEqual(model["runs_on"], "board")
            for key in ("maker", "task", "task_tag", "size", "size_kb", "rank"):
                self.assertIn(key, model)

    def test_publish_and_unpublish(self):
        with TemporaryDirectory() as directory:
            store = ModelStore(Path(directory) / "models.json")
            item = store.publish({"name": "Tighter warnings", "task": "Warns earlier", "task_tag": "early warning", "size_kb": 70})
            self.assertEqual(item["maker"], "you")
            again = store.publish({"name": "Tighter warnings"})
            self.assertNotEqual(item["id"], again["id"])
            listed = store.list()
            self.assertEqual(sum(model["published"] for model in listed), 2)
            self.assertEqual(listed[-1]["rank"], len(listed))
            store.unpublish(item["id"])
            self.assertEqual(sum(model["published"] for model in store.list()), 1)
            with self.assertRaises(ValueError):
                store.unpublish("flan-t5-small-int8")
            with self.assertRaises(ValueError):
                store.publish({"name": " "})
            with self.assertRaises(ValueError):
                store.publish({"name": "x", "size_kb": -1})

    def test_accounts_actions(self):
        with TemporaryDirectory() as directory:
            accounts = Accounts(Path(directory) / "history", environ={}, models_path=Path(directory) / "models.json")
            made = accounts.action({"action": "model_publish", "name": "My skill", "task": "Custom"})
            self.assertTrue(any(model["id"] == made["model"]["id"] for model in made["models"]))
            gone = accounts.action({"action": "model_unpublish", "id": made["model"]["id"]})
            self.assertFalse(any(model["published"] for model in gone["models"]))


class ProgressiveAnswerTest(unittest.TestCase):
    def test_styles_arrive_in_order_and_export_keeps_them(self):
        kit = Kit(NoBoard())
        kit.action({"action": "load_sample", "id": SAMPLES[1]["id"]})
        job = kit.action({"action": "data_ask_begin", "question": "what happened?", "style": "table"})
        self.assertEqual(job["order"], ["short", "summary", "transcript", "table"])
        for _ in range(100):
            view = kit.action({"action": "data_ask_progress", "job_id": job["job_id"]})
            if view["state"] != "running":
                break
            time.sleep(0.02)
        self.assertEqual(view["state"], "complete")
        self.assertEqual(list(view["ready"]), ["short", "summary", "transcript", "table"])
        self.assertIn("answer_ms", view["timings"])
        with self.assertRaises(ValueError):
            kit.action({"action": "data_ask_progress", "job_id": "missing"})
        with self.assertRaises(ValueError):
            kit.action({"action": "data_ask_begin", "question": "", "style": "short"})
        variants = {style: {"text": result["answer"], "table": result["table"]} for style, result in view["ready"].items()}
        session = {"id": "s", "title": "t", "dataset": "d", "model": "m", "notes": "", "updated_ms": 1, "timings": {},
                   "messages": [{"role": "user", "text": "what happened?", "time_ms": 1},
                                {"role": "answer", "text": "x", "readings": "", "time_ms": 1, "variants": variants}]}
        markdown = export_session(session, "md")["content"]
        for heading in ("#### Short", "#### Summary", "#### Transcript", "#### Table"):
            self.assertIn(heading, markdown)
        rows = list(csv.reader(io.StringIO(export_session(session, "csv")["content"])))
        self.assertEqual(len(rows), 5)


class NoFollow(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class ServerTest(unittest.TestCase):
    def test_port_falls_back_when_the_default_is_taken(self):
        import pinkprocess.kit_app as kit_app
        real = kit_app.KitServer
        attempts = []

        def refuse_default(address, handler):
            attempts.append(address[1])
            if address[1] == DEFAULT_PORT:
                raise OSError("Address already in use")
            return real(address, handler)

        with patch.object(kit_app, "KitServer", side_effect=refuse_default):
            with make_server() as server:
                self.assertEqual(attempts[0], DEFAULT_PORT)
                self.assertNotEqual(server.server_port, DEFAULT_PORT)

    def test_sign_in_start_rejects_foreign_referers(self):
        with patch.dict("os.environ", {"PYNQ_AUTH0_DOMAIN": "tenant.auth0.com", "PYNQ_AUTH0_CLIENT_ID": "abc"}):
            server = make_server(port=0)
        with server:
            Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_port}/auth/login"
            opener = build_opener(NoFollow)
            with self.assertRaises(HTTPError) as foreign:
                opener.open(Request(base, headers={"Referer": "http://evil.example/"}))
            self.assertEqual(foreign.exception.code, 403)
            with self.assertRaises(HTTPError) as ok:
                opener.open(Request(base, headers={"Referer": f"http://127.0.0.1:{server.server_port}/"}))
            self.assertEqual(ok.exception.code, 302)
            self.assertIn("nonce=", ok.exception.headers["Location"])
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
