from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .kit_accounts import MODEL_LIBRARY, Accounts, pkce_pair


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


if __name__ == "__main__":
    unittest.main()
