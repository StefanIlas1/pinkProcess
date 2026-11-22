"""Sign-in, per-user chat history and the model library for the PYNQ Process app."""

import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
from threading import RLock
from time import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .kit_models import BUILT_IN as MODEL_LIBRARY, ModelStore


CALLBACK_PATH = "/auth/callback"
HISTORY_DIR = Path.home() / "Library" / "Application Support" / "PYNQ Process" / "history"
MAX_CHATS = 200
MAX_TEXT = 20000
MAX_EXTRA_BYTES = 200000
LOGIN_TTL_SECONDS = 600
MAX_PENDING_LOGINS = 8
NUMBER_TYPES = (int, float)


def pkce_pair():
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def decode_claims(token):
    payload = token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    if not isinstance(claims, dict):
        raise ValueError("Unreadable sign-in token.")
    return claims


def user_key(sub):
    return hashlib.sha256(sub.encode()).hexdigest()[:32]


def clean_text(value, limit=MAX_TEXT):
    return value[:limit] if isinstance(value, str) else ""


VARIANT_STYLES = ("short", "summary", "final", "transcript", "table")


def clean_variants(value):
    if not isinstance(value, dict):
        return {}
    cleaned = {}
    for style in VARIANT_STYLES:
        item = value.get(style)
        if not isinstance(item, dict):
            continue
        entry = {"text": clean_text(item.get("text"))}
        if isinstance(item.get("table"), dict) and len(json.dumps(item["table"])) <= MAX_EXTRA_BYTES:
            entry["table"] = item["table"]
        if item.get("mono") is True:
            entry["mono"] = True
        cleaned[style] = entry
    return cleaned


def clean_message(message):
    if not isinstance(message, dict):
        return None
    cleaned = {"role": "user" if message.get("role") == "user" else "answer", "text": clean_text(message.get("text")),
               "time_ms": message["time_ms"] if isinstance(message.get("time_ms"), NUMBER_TYPES) else 0}
    if cleaned["role"] == "answer":
        cleaned["readings"] = clean_text(message.get("readings"))
        if isinstance(message.get("table"), dict):
            cleaned["table"] = message["table"]
        if message.get("mono") is True:
            cleaned["mono"] = True
        if isinstance(message.get("style"), str):
            cleaned["style"] = message["style"][:20]
        variants = clean_variants(message.get("variants"))
        if variants:
            cleaned["variants"] = variants
    return cleaned


def clean_timings(value):
    if not isinstance(value, dict):
        return {}
    return {str(key)[:30]: round(float(number), 1) for key, number in list(value.items())[:12]
            if isinstance(number, NUMBER_TYPES) and not isinstance(number, bool) and abs(number) < 1e12}


def clean_chat(chat):
    if not isinstance(chat, dict) or not isinstance(chat.get("id"), str) or not chat["id"]:
        return None
    updated = chat.get("updated_ms")
    cleaned = {**chat, "id": chat["id"][:64], "title": clean_text(chat.get("title"), 200), "dataset": clean_text(chat.get("dataset"), 200),
               "model": clean_text(chat.get("model"), 200), "notes": clean_text(chat.get("notes")),
               "updated_ms": updated if isinstance(updated, NUMBER_TYPES) and not isinstance(updated, bool) else 0,
               "messages": [message for message in map(clean_message, chat["messages"]) if message]
               if isinstance(chat.get("messages"), list) else []}
    cleaned["timings"] = clean_timings(chat.get("timings"))
    return cleaned


def read_chat_file(path):
    try:
        chats = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(chats, list):
        return []
    chats = [chat for chat in map(clean_chat, chats) if chat]
    return sorted(chats, key=lambda chat: chat["updated_ms"], reverse=True)


class Accounts:
    def __init__(self, history_dir=HISTORY_DIR, environ=None, models_path=None):
        environ = os.environ if environ is None else environ
        self.history_dir = Path(history_dir)
        self.models = ModelStore(models_path or self.history_dir.parent / "models.json")
        self.domain = (environ.get("PYNQ_AUTH0_DOMAIN") or "").strip().removeprefix("https://").strip("/")
        self.client_id = (environ.get("PYNQ_AUTH0_CLIENT_ID") or "").strip()
        self.user = None
        self.pending = {}
        self.lock = RLock()

    @property
    def auth0_enabled(self):
        return bool(self.domain and self.client_id)

    def session(self):
        return {"user": self.user, "auth0": self.auth0_enabled}

    def user_file(self, user=None):
        user = user or self.user
        if user is None:
            raise ValueError("Log in to keep your chats.")
        return self.history_dir / f"{user_key(user['sub'])}.json"

    def latest_history_file(self):
        try:
            files = [path for path in self.history_dir.glob("*.json") if path.is_file()]
            return max(files, key=lambda path: path.stat().st_mtime, default=None)
        except OSError:
            return None

    def login_url(self, origin):
        if not self.auth0_enabled:
            raise ValueError("Sign-in is not configured.")
        verifier, challenge = pkce_pair()
        state = secrets.token_urlsafe(24)
        nonce = secrets.token_urlsafe(24)
        now = time()
        with self.lock:
            live = {key: entry for key, entry in self.pending.items() if now - entry[2] < LOGIN_TTL_SECONDS}
            newest = sorted(live.items(), key=lambda item: item[1][2])[-(MAX_PENDING_LOGINS - 1):]
            self.pending = dict(newest)
            self.pending[state] = (verifier, origin + CALLBACK_PATH, now, nonce)
        query = urlencode({
            "response_type": "code", "client_id": self.client_id, "redirect_uri": origin + CALLBACK_PATH,
            "scope": "openid profile", "connection": "google-oauth2", "state": state, "nonce": nonce, "code_challenge": challenge, "code_challenge_method": "S256",
        })
        return f"https://{self.domain}/authorize?{query}"

    def finish_login(self, code, state, error=""):
        with self.lock:
            entry = self.pending.pop(state, None) if isinstance(state, str) else None
        if entry is None or time() - entry[2] >= LOGIN_TTL_SECONDS:
            raise ValueError("This sign-in link has expired. Try logging in again.")
        if error:
            raise ValueError("Sign-in was cancelled. You can try again from the app.")
        if not code:
            raise ValueError("The sign-in did not return a code. Try logging in again.")
        verifier, redirect_uri, _, nonce = entry
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
            subject = claims["sub"]
            if not isinstance(subject, str) or not subject:
                raise ValueError("Unreadable sign-in token.")
        except (URLError, OSError, KeyError, IndexError, ValueError, TypeError, AttributeError):
            raise RuntimeError("Sign-in could not be completed. Try again.") from None
        audience = claims.get("aud")
        if audience != self.client_id and not (isinstance(audience, list) and self.client_id in audience):
            raise RuntimeError("Sign-in could not be completed. Try again.")
        if claims.get("nonce") != nonce:
            raise RuntimeError("Sign-in could not be completed. Try again.")
        name = claims.get("name") or claims.get("nickname") or claims.get("email") or "You"
        with self.lock:
            self.user = {"sub": subject, "name": str(name)[:80]}

    def action(self, data):
        action = data.get("action")
        if action == "session":
            return self.session()
        if action == "local_login":
            name = data.get("name")
            if not isinstance(name, str) or not 0 < len(name.strip()) <= 60:
                raise ValueError("Enter a name of up to 60 characters.")
            name = name.strip()
            if any(ord(char) < 32 for char in name):
                raise ValueError("Use letters and numbers in your name.")
            with self.lock:
                self.user = {"sub": "local:" + name.lower(), "name": name}
            return self.session()
        if action == "logout":
            with self.lock:
                self.user = None
                self.pending = {}
            return self.session()
        if action == "history_list":
            return {"chats": [{"id": chat["id"], "title": chat.get("title", ""), "updated_ms": chat.get("updated_ms", 0),
                               "dataset": chat.get("dataset", ""), "model": chat.get("model", ""),
                               "questions": sum(message.get("role") == "user" for message in chat.get("messages", []))}
                              for chat in self.read_chats()]}
        if action == "history_get":
            return {"chat": self.get_chat(data.get("chat_id"))}
        if action == "history_save":
            return self.save_exchange(data)
        if action == "session_save":
            return self.save_session(data)
        if action == "model_publish":
            return {"model": self.models.publish(data), "models": self.models.list()}
        if action == "model_unpublish":
            result = self.models.unpublish(data.get("id"))
            return {**result, "models": self.models.list()}
        raise ValueError("Unknown account action.")

    def get_chat(self, chat_id):
        if isinstance(chat_id, str):
            for chat in self.read_chats():
                if chat["id"] == chat_id:
                    return chat
        raise ValueError("That chat is no longer available.")

    def read_chats(self, user=None):
        path = self.user_file(user)
        with self.lock:
            return read_chat_file(path)

    def write_chats(self, path, chats):
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(chats, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def set_aside_damaged(self, path):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            try:
                os.replace(path, path.with_suffix(".damaged"))
            except OSError:
                pass

    def save_exchange(self, data):
        question, answer = data.get("question"), data.get("answer")
        readings = data.get("readings", "")
        if not all(isinstance(value, str) for value in (question, answer, readings)):
            raise ValueError("Nothing to save.")
        if not question.strip() or len(question) > MAX_TEXT or len(answer) > MAX_TEXT or len(readings) > MAX_TEXT:
            raise ValueError("This chat message is too long to save.")
        extra = {"text": answer, "readings": readings}
        for key in ("table", "mono", "style", "variants"):
            if key in data:
                extra[key] = data[key]
        if "table" in extra and (not isinstance(extra["table"], dict) or len(json.dumps(extra["table"])) > MAX_EXTRA_BYTES):
            raise ValueError("This table is too large to save.")
        if "mono" in extra and not isinstance(extra["mono"], bool):
            raise ValueError("Nothing to save.")
        if "style" in extra and not isinstance(extra["style"], str):
            raise ValueError("Nothing to save.")
        return self.update_session(data, question.strip()[:80], [{"role": "user", "text": question}, {"role": "answer", **extra}])

    def save_session(self, data):
        notes = data.get("notes", "")
        if not isinstance(notes, str) or len(notes) > MAX_TEXT:
            raise ValueError("These notes are too long to save.")
        return self.update_session(data, None, [], notes, clean_timings(data["timings"]) if "timings" in data else None)

    def update_session(self, data, title, messages, notes=None, timings=None):
        fields = {}
        for key in ("dataset", "model"):
            value = data.get(key, "")
            if not isinstance(value, str) or len(value) > 200:
                raise ValueError("Invalid session details.")
            fields[key] = value
        user = self.user
        path = self.user_file(user)
        chat_id = data.get("chat_id")
        if chat_id is not None and not isinstance(chat_id, str):
            raise ValueError("That chat is no longer available.")
        with self.lock:
            self.set_aside_damaged(path)
            chats = self.read_chats(user)
            now = int(time() * 1000)
            chat = next((item for item in chats if item["id"] == chat_id), None)
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
            if timings:
                chat["timings"] = timings
            for message in messages:
                chat["messages"].append({**message, "time_ms": now})
            chat["updated_ms"] = now
            chats = sorted(chats, key=lambda item: item["updated_ms"], reverse=True)[:MAX_CHATS]
            self.write_chats(path, chats)
        return {"chat_id": chat["id"]}
