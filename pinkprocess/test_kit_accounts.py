import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import patch

from . import kit_accounts
from .kit_accounts import MODEL_LIBRARY, Accounts, pkce_pair

CONFIGURED = {"PYNQ_AUTH0_DOMAIN": "tenant.auth0.com", "PYNQ_AUTH0_CLIENT_ID": "abc"}


def fake_token(claims):
    import base64

    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"h.{payload}.s"


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return self.body


class AccountsTest(unittest.TestCase):
    def test_history_requires_login_and_is_per_user(self):
        with TemporaryDirectory() as directory:
            accounts = Accounts(Path(directory), environ={})
            with self.assertRaises(ValueError):
                accounts.action({"action": "history_list"})
            accounts.action({"action": "local_login", "name": "Ana"})
            saved = accounts.action({"action": "history_save", "question": "Why?", "answer": "Busy."})
            accounts.action({"action": "history_save", "chat_id": saved["chat_id"], "question": "More?", "answer": "No."})
            chats = accounts.action({"action": "history_list"})["chats"]
            self.assertEqual([chat["title"] for chat in chats], ["Why?"])
            chat = accounts.action({"action": "history_get", "chat_id": saved["chat_id"]})["chat"]
            self.assertEqual(len(chat["messages"]), 4)
            accounts.action({"action": "local_login", "name": "Ben"})
            self.assertEqual(accounts.action({"action": "history_list"})["chats"], [])
            accounts.action({"action": "logout"})
            self.assertIsNone(accounts.session()["user"])

    def test_sessions_keep_dataset_model_and_notes(self):
        with TemporaryDirectory() as directory:
            accounts = Accounts(Path(directory), environ={})
            accounts.action({"action": "local_login", "name": "Ana"})
            made = accounts.action({"action": "session_save", "dataset": "pool.csv", "model": "queue-onset-logistic", "notes": "check pool size"})
            accounts.action({"action": "history_save", "chat_id": made["chat_id"], "question": "Peak?", "answer": "19",
                             "dataset": "pool.csv", "model": "queue-onset-logistic"})
            listed = accounts.action({"action": "history_list"})["chats"]
            self.assertEqual(len(listed), 1)
            self.assertEqual((listed[0]["title"], listed[0]["dataset"], listed[0]["questions"]), ("pool.csv", "pool.csv", 1))
            chat = accounts.action({"action": "history_get", "chat_id": made["chat_id"]})["chat"]
            self.assertEqual(chat["notes"], "check pool size")
            accounts.action({"action": "logout"})
            with self.assertRaises(ValueError):
                accounts.action({"action": "session_save", "notes": "x"})

    def test_auth0_requires_configuration(self):
        accounts = Accounts(environ={})
        self.assertFalse(accounts.session()["auth0"])
        with self.assertRaises(ValueError):
            accounts.login_url("http://127.0.0.1:1")
        configured = Accounts(environ={"PYNQ_AUTH0_DOMAIN": "tenant.auth0.com", "PYNQ_AUTH0_CLIENT_ID": "abc"})
        url = configured.login_url("http://127.0.0.1:1")
        self.assertIn("code_challenge_method=S256", url)
        self.assertIn("%2Fauth%2Fcallback", url)
        with self.assertRaises(ValueError):
            configured.finish_login("code", "wrong")

    def test_pkce_and_library(self):
        verifier, challenge = pkce_pair()
        self.assertNotEqual(verifier, challenge)
        self.assertTrue(any(not model["available"] for model in MODEL_LIBRARY))

    def state_of(self, accounts):
        url = accounts.login_url("http://127.0.0.1:1")
        return url.split("state=")[1].split("&")[0]

    def test_login_state_is_single_use_and_expires(self):
        accounts = Accounts(environ=CONFIGURED)
        reply = FakeResponse({"id_token": fake_token({"sub": "google|1", "name": "Ana", "aud": "abc"})})
        state = self.state_of(accounts)
        with patch.object(kit_accounts, "urlopen", return_value=reply) as call:
            accounts.finish_login("code", state)
            self.assertEqual(accounts.session()["user"]["name"], "Ana")
            self.assertIn(b"code_verifier=", call.call_args.args[0].data)
            with self.assertRaises(ValueError):
                accounts.finish_login("code", state)
        stale = self.state_of(accounts)
        with patch.object(kit_accounts, "time", return_value=kit_accounts.time() + kit_accounts.LOGIN_TTL_SECONDS + 1):
            with self.assertRaisesRegex(ValueError, "expired"):
                accounts.finish_login("code", stale)

    def test_login_error_paths(self):
        accounts = Accounts(environ=CONFIGURED)
        with self.assertRaisesRegex(ValueError, "cancelled"):
            accounts.finish_login("", self.state_of(accounts), "access_denied")
        with self.assertRaisesRegex(ValueError, "code"):
            accounts.finish_login("", self.state_of(accounts))
        wrong_audience = FakeResponse({"id_token": fake_token({"sub": "1", "aud": "someone-else"})})
        no_subject = FakeResponse({"id_token": fake_token({"name": "x"})})
        for reply in (wrong_audience, no_subject, FakeResponse({})):
            with patch.object(kit_accounts, "urlopen", return_value=reply):
                with self.assertRaises(RuntimeError):
                    accounts.finish_login("code", self.state_of(accounts))
        self.assertIsNone(accounts.session()["user"])

    def test_pending_logins_are_capped(self):
        accounts = Accounts(environ=CONFIGURED)
        for _ in range(30):
            accounts.login_url("http://127.0.0.1:1")
        self.assertLessEqual(len(accounts.pending), kit_accounts.MAX_PENDING_LOGINS)

    def test_user_names_cannot_escape_the_history_folder(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "history"
            accounts = Accounts(root, environ={})
            accounts.action({"action": "local_login", "name": "../../evil"})
            accounts.action({"action": "session_save", "notes": "x"})
            self.assertEqual([path.parent for path in root.rglob("*.json")], [root])
            self.assertEqual(list(Path(directory).glob("*.json")), [])
            for bad in (["x"], {"a": 1}, 5, "../../etc/passwd", None):
                with self.assertRaises(ValueError):
                    accounts.action({"action": "history_get", "chat_id": bad})
            with self.assertRaises(ValueError):
                accounts.action({"action": "history_save", "chat_id": ["x"], "question": "q", "answer": "a"})
            with self.assertRaises(ValueError):
                accounts.action({"action": "local_login", "name": "bad\x00name"})

    def test_damaged_history_does_not_crash_and_is_kept(self):
        with TemporaryDirectory() as directory:
            accounts = Accounts(Path(directory), environ={})
            accounts.action({"action": "local_login", "name": "Ana"})
            path = accounts.user_file()
            for broken in ("{not json", "{}", "[1, null, {\"id\": 3}, {\"id\": \"ok\", \"messages\": 7, \"updated_ms\": \"x\"}]"):
                path.write_text(broken)
                chats = accounts.action({"action": "history_list"})["chats"]
                self.assertTrue(all(chat["questions"] == 0 for chat in chats))
            path.write_text("{not json")
            accounts.action({"action": "history_save", "question": "Why?", "answer": "Busy."})
            self.assertEqual(path.with_suffix(".damaged").read_text(), "{not json")
            self.assertEqual(len(accounts.action({"action": "history_list"})["chats"]), 1)

    def test_concurrent_saves_are_atomic_and_none_are_lost(self):
        with TemporaryDirectory() as directory:
            accounts = Accounts(Path(directory), environ={})
            accounts.action({"action": "local_login", "name": "Ana"})
            made = accounts.action({"action": "session_save", "dataset": "d.csv"})["chat_id"]
            threads = [Thread(target=accounts.action, args=({"action": "history_save", "chat_id": made, "question": f"q{index}", "answer": "a"},))
                       for index in range(25)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            chat = accounts.action({"action": "history_get", "chat_id": made})["chat"]
            self.assertEqual(len(chat["messages"]), 50)
            self.assertEqual([path.name for path in Path(directory).iterdir() if path.suffix != ".json"], [])

    def test_save_validation(self):
        with TemporaryDirectory() as directory:
            accounts = Accounts(Path(directory), environ={})
            accounts.action({"action": "local_login", "name": "Ana"})
            base = {"action": "history_save", "question": "q", "answer": "a"}
            for extra in ({"table": "x"}, {"table": {"rows": ["x" * 300000]}}, {"mono": "yes"}, {"style": 4}, {"dataset": 4}):
                with self.assertRaises(ValueError):
                    accounts.action({**base, **extra})
            saved = accounts.action({**base, "style": "table", "table": {"columns": ["a"], "rows": [["1"]]}, "mono": False})
            session = accounts.action({"action": "session_save", "chat_id": saved["chat_id"], "timings": {"data_ms": 3, "bad": "x", "hot": True}})
            chat = accounts.action({"action": "history_get", "chat_id": session["chat_id"]})["chat"]
            self.assertEqual(chat["timings"], {"data_ms": 3.0})
            self.assertEqual(chat["messages"][1]["style"], "table")

    def test_logout_clears_the_session_and_pending_logins(self):
        accounts = Accounts(environ=CONFIGURED)
        accounts.action({"action": "local_login", "name": "Ana"})
        self.state_of(accounts)
        accounts.action({"action": "logout"})
        self.assertIsNone(accounts.user)
        self.assertEqual(accounts.pending, {})


if __name__ == "__main__":
    unittest.main()
