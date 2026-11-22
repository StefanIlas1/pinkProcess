"""Sign-in, per-user chat history and the model library for the PYNQ Process app."""

import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
from threading import Lock
from time import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CALLBACK_PATH = "/auth/callback"
HISTORY_DIR = Path.home() / "Library" / "Application Support" / "PYNQ Process" / "history"
MAX_CHATS = 200
MAX_TEXT = 20000
MODEL_LIBRARY = [
    {"id": "flan-t5-small-int8", "name": "FLAN-T5-small INT8", "task": "Explains what is going wrong",
     "kind": "language", "available": True},
    {"id": "queue-onset-logistic", "name": "Queue-onset logistic", "task": "Early warning when requests start piling up",
     "kind": "queue", "available": True},
    {"id": "flan-t5-small-int8-runbook", "name": "FLAN-T5-small INT8 runbook", "task": "Suggests the next step to fix a problem",
     "kind": "language", "available": False},
    {"id": "timeseries-anomaly-int8", "name": "Anomaly spotter INT8", "task": "Spots unusual readings in any number stream",
     "kind": "anomaly", "available": False},
    {"id": "queue-onset-logistic-long", "name": "Queue-onset logistic long range", "task": "Early warning a full minute ahead",
     "kind": "queue", "available": False},
]


def pkce_pair():
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def decode_claims(token):
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


class Accounts:
    def __init__(self, history_dir=HISTORY_DIR, environ=None):
        environ = os.environ if environ is None else environ
        self.history_dir = Path(history_dir)
        self.domain = (environ.get("PYNQ_AUTH0_DOMAIN") or "").strip().removeprefix("https://").strip("/")
        self.client_id = (environ.get("PYNQ_AUTH0_CLIENT_ID") or "").strip()
        self.user = None
        self.pending = {}
        self.lock = Lock()

    @property
    def auth0_enabled(self):
        return bool(self.domain and self.client_id)

    def session(self):
        return {"user": self.user, "auth0": self.auth0_enabled}

    def user_file(self):
        if self.user is None:
            raise ValueError("Log in to keep your chats.")
        name = hashlib.sha256(self.user["sub"].encode()).hexdigest()[:32]
        return self.history_dir / f"{name}.json"

    def login_url(self, origin):
        if not self.auth0_enabled:
            raise ValueError("Sign-in is not configured.")
        verifier, challenge = pkce_pair()
        state = secrets.token_urlsafe(24)
        with self.lock:
            self.pending = {state: (verifier, origin + CALLBACK_PATH)}
        query = urlencode({
            "response_type": "code", "client_id": self.client_id, "redirect_uri": origin + CALLBACK_PATH,
            "scope": "openid profile", "connection": "google-oauth2", "state": state, "code_challenge": challenge, "code_challenge_method": "S256",
        })
        return f"https://{self.domain}/authorize?{query}"

    def finish_login(self, code, state):
        with self.lock:
            entry = self.pending.pop(state, None)
        if entry is None or not code:
            raise ValueError("This sign-in link has expired. Try logging in again.")
        verifier, redirect_uri = entry
        body = urlencode({
            "grant_type": "authorization_code", "client_id": self.client_id, "code": code,
            "code_verifier": verifier, "redirect_uri": redirect_uri,
        }).encode()
        request = Request(f"https://{self.domain}/oauth/token", body,
                          {"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urlopen(request, timeout=15) as response:
                tokens = json.load(response)
            claims = decode_claims(tokens["id_token"])
        except (URLError, OSError, KeyError, IndexError, ValueError):
            raise RuntimeError("Sign-in could not be completed. Try again.")
        name = claims.get("name") or claims.get("nickname") or claims.get("email") or "You"
        self.user = {"sub": str(claims["sub"]), "name": str(name)[:80]}

    def action(self, data):
        action = data.get("action")
        if action == "session":
            return self.session()
        if action == "local_login":
            name = data.get("name")
            if not isinstance(name, str) or not 0 < len(name.strip()) <= 60:
                raise ValueError("Enter a name of up to 60 characters.")
            name = name.strip()
            self.user = {"sub": "local:" + name.lower(), "name": name}
            return self.session()
        if action == "logout":
            self.user = None
            return self.session()
        if action == "history_list":
            return {"chats": [{"id": chat["id"], "title": chat.get("title", ""), "updated_ms": chat.get("updated_ms", 0),
                               "dataset": chat.get("dataset", ""), "model": chat.get("model", ""),
                               "questions": sum(message.get("role") == "user" for message in chat.get("messages", []))}
                              for chat in self.read_chats()]}
        if action == "history_get":
            chat_id = data.get("chat_id")
            for chat in self.read_chats():
                if chat["id"] == chat_id:
                    return {"chat": chat}
            raise ValueError("That chat is no longer available.")
        if action == "history_save":
            return self.save_exchange(data)
        if action == "session_save":
            return self.save_session(data)
        raise ValueError("Unknown account action.")

    def read_chats(self):
        path = self.user_file()
        try:
            chats = json.loads(path.read_text())
        except (OSError, ValueError):
            return []
        chats = [chat for chat in chats if isinstance(chat, dict) and "id" in chat] if isinstance(chats, list) else []
        return sorted(chats, key=lambda chat: chat.get("updated_ms", 0), reverse=True)

    def save_exchange(self, data):
        question, answer = data.get("question"), data.get("answer")
        readings = data.get("readings", "")
        if not all(isinstance(value, str) for value in (question, answer, readings)):
            raise ValueError("Nothing to save.")
        if not question.strip() or len(question) > MAX_TEXT or len(answer) > MAX_TEXT or len(readings) > MAX_TEXT:
            raise ValueError("This chat message is too long to save.")
        extra = {"text": answer, "readings": readings}
        for key in ("table", "mono"):
            if key in data:
                extra[key] = data[key]
        return self.update_session(data, question.strip()[:80], [{"role": "user", "text": question}, {"role": "answer", **extra}])

    def save_session(self, data):
        notes = data.get("notes", "")
        if not isinstance(notes, str) or len(notes) > MAX_TEXT:
            raise ValueError("These notes are too long to save.")
        return self.update_session(data, None, [], notes)

    def update_session(self, data, title, messages, notes=None):
        fields = {}
        for key in ("dataset", "model"):
            value = data.get(key, "")
            if not isinstance(value, str) or len(value) > 200:
                raise ValueError("Invalid session details.")
            fields[key] = value
        path = self.user_file()
        with self.lock:
            chats = self.read_chats()
            now = int(time() * 1000)
            chat = next((item for item in chats if item["id"] == data.get("chat_id")), None)
            if chat is None:
                chat = {"id": secrets.token_hex(8), "title": title or fields["dataset"] or "New session", "created_ms": now,
                        "messages": [], "notes": ""}
                chats.append(chat)
            elif title and not chat.get("dataset") and not chat["messages"]:
                chat["title"] = title
            for key, value in fields.items():
                if value:
                    chat[key] = value
            if fields["dataset"] and chat["title"] in ("New session", ""):
                chat["title"] = fields["dataset"]
            if notes is not None:
                chat["notes"] = notes
            for message in messages:
                chat["messages"].append({**message, "time_ms": now})
            chat["updated_ms"] = now
            chats = sorted(chats, key=lambda item: item["updated_ms"], reverse=True)[:MAX_CHATS]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(chats))
            temporary.replace(path)
        return {"chat_id": chat["id"]}
