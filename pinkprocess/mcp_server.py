"""Model Context Protocol server (stdio) exposing PYNQ Process sessions and datasets."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .kit_accounts import HISTORY_DIR, Accounts, read_chat_file, user_key
from .kit_app import Kit
from .kit_data import DATA_DIR, SAMPLES, STYLES, DatasetStore, parse_table, profile_table, sample_csv

SERVER_NAME = "pynq-process"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_LINE_BYTES = 30 * 1024 * 1024
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603

TOOLS = [
    dict(name="list_sessions", description="List the saved PYNQ Process sessions of a user, newest first.",
         inputSchema=dict(type="object", properties=dict(user=dict(type="string", description="Profile name. Defaults to the most recently used profile.")))),
    dict(name="get_session", description="Read one saved session with its questions, answers, readings and notes.",
         inputSchema=dict(type="object", properties=dict(id=dict(type="string"), user=dict(type="string")), required=["id"])),
    dict(name="list_datasets", description="List the sample datasets and the data files imported recently in the app.",
         inputSchema=dict(type="object", properties={})),
    dict(name="profile_dataset", description="Profile a dataset: columns, signals and the recommended quantized model.",
         inputSchema=dict(type="object", properties=dict(name=dict(type="string")), required=["name"])),
    dict(name="ask_data", description="Ask a question about a dataset on this Mac, the same way the app does.",
         inputSchema=dict(type="object", properties=dict(
             dataset=dict(type="string"), question=dict(type="string"),
             style=dict(type="string", enum=list(STYLES), default="short")), required=["dataset", "question"])),
]


class NoBoard:
    def close(self):
        pass


def frozen_app():
    return bool(getattr(sys, "frozen", False))


def repo_root():
    if not frozen_app():
        return Path(__file__).resolve().parent.parent
    for candidate in (os.environ.get("PYNQ_REPO_DIR"), Path.cwd()):
        if candidate and (Path(candidate) / "pinkprocess" / "mcp_server.py").is_file():
            return Path(candidate).resolve()
    return None


def interpreter_ok(path):
    try:
        done = subprocess.run([path, "-c", "import sys; print(int(sys.version_info >= (3, 10)))"],
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and done.stdout.strip() == "1"


def pick_python():
    candidates = ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", shutil.which("python3")]
    if not frozen_app():
        candidates.append(sys.executable)
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and interpreter_ok(candidate):
            return str(Path(candidate).resolve())
    fallback = shutil.which("python3") if frozen_app() else sys.executable
    return str(Path(fallback).resolve()) if fallback else "/usr/bin/python3"


def mcp_config(environ=None):
    environ = os.environ if environ is None else environ
    root = repo_root()
    python = pick_python()
    env = {"PYTHONPATH": str(root)} if root else {}
    if environ.get("PYNQ_USER"):
        env["PYNQ_USER"] = environ["PYNQ_USER"]
    args = ["-m", "pinkprocess.mcp_server"]
    entry = dict(command=python, args=args, env=env)
    config_json = json.dumps({"mcpServers": {SERVER_NAME: entry}}, indent=2)
    result = dict(command=python, args=args, env=env, config_json=config_json)
    if root is None:
        result["note"] = "Set PYTHONPATH to the folder that contains pinkprocess before pasting this into Claude Desktop."
    return result


class Connector:
    def __init__(self, history_dir=HISTORY_DIR, data_dir=DATA_DIR, environ=None):
        self.environ = os.environ if environ is None else environ
        self.accounts = Accounts(history_dir, environ={})
        self.store = DatasetStore(data_dir)

    def history_file(self, user):
        user = (user or self.environ.get("PYNQ_USER") or "").strip()
        if not user:
            latest = self.accounts.latest_history_file()
            if latest is None:
                raise ValueError("No saved sessions yet. Save a session in PYNQ Process first.")
            return latest
        for sub in ("local:" + user.lower(), user):
            path = self.accounts.history_dir / f"{user_key(sub)}.json"
            if path.is_file():
                return path
        raise ValueError(f"No saved sessions for {user}.")

    def list_sessions(self, args):
        chats = read_chat_file(self.history_file(args.get("user")))
        return dict(sessions=[dict(id=chat["id"], title=chat["title"], dataset=chat["dataset"], model=chat["model"],
                                   updated_ms=chat["updated_ms"],
                                   questions=sum(message["role"] == "user" for message in chat["messages"]))
                              for chat in chats])

    def get_session(self, args):
        chat_id = args.get("id")
        if not isinstance(chat_id, str) or not chat_id:
            raise ValueError("Give the id of a saved session.")
        first = self.history_file(args.get("user"))
        others = [path for path in self.accounts.history_dir.glob("*.json") if path != first] if not args.get("user") else []
        for path in [first, *others]:
            for chat in read_chat_file(path):
                if chat["id"] == chat_id:
                    return chat
        raise ValueError("That session is not available.")

    def list_datasets(self, args):
        return dict(samples=[dict(name=sample["name"], id=sample["id"], description=sample["description"]) for sample in SAMPLES],
                    imported=self.store.names())

    def dataset_text(self, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Give the name of a dataset. Use list_datasets to see the choices.")
        name = name.strip()
        for sample in SAMPLES:
            if name.lower() in (sample["id"], sample["name"].lower(), sample["name"].lower() + ".csv"):
                return sample["name"] + ".csv", sample_csv(sample["id"])
        return name, self.store.read(name)

    def profile_dataset(self, args):
        name, text = self.dataset_text(args.get("name"))
        return profile_table(parse_table(text, name), name)

    def ask_data(self, args):
        name, text = self.dataset_text(args.get("dataset"))
        kit = Kit(NoBoard())
        kit.import_data(name, text)
        result = kit.action(dict(action="data_ask", question=args.get("question"), style=args.get("style", "short")))
        return {key: result[key] for key in ("answer", "readings", "table", "style", "focus", "where", "model", "events")}


def respond(identifier, result=None, error=None):
    message = {"jsonrpc": "2.0", "id": identifier}
    if error:
        message["error"] = dict(code=error[0], message=error[1])
    else:
        message["result"] = result
    return message


def call_tool(connector, params):
    name, args = params.get("name"), params.get("arguments", {})
    args = {} if args is None else args
    handlers = {tool["name"]: getattr(connector, tool["name"]) for tool in TOOLS}
    if name not in handlers:
        raise KeyError(name)
    if not isinstance(args, dict):
        raise TypeError("arguments")
    try:
        payload = handlers[name](args)
        return dict(content=[dict(type="text", text=json.dumps(payload, indent=2, ensure_ascii=False))], isError=False)
    except (ValueError, OSError) as exc:
        return dict(content=[dict(type="text", text=str(exc))], isError=True)


def handle(connector, message):
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return respond(message.get("id") if isinstance(message, dict) else None, error=(INVALID_REQUEST, "Invalid request."))
    method, identifier = message["method"], message.get("id")
    params = message.get("params", {})
    params = {} if params is None else params
    if "id" not in message:
        return None
    if not isinstance(params, dict):
        return respond(identifier, error=(INVALID_PARAMS, "Params must be an object."))
    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return respond(identifier, dict(protocolVersion=version, capabilities=dict(tools=dict(listChanged=False)),
                                        serverInfo=dict(name=SERVER_NAME, version=SERVER_VERSION)))
    if method == "ping":
        return respond(identifier, {})
    if method == "tools/list":
        return respond(identifier, dict(tools=TOOLS))
    if method == "tools/call":
        try:
            return respond(identifier, call_tool(connector, params))
        except KeyError:
            return respond(identifier, error=(INVALID_PARAMS, f"Unknown tool: {params.get('name')}"))
        except TypeError:
            return respond(identifier, error=(INVALID_PARAMS, "Tool arguments must be an object."))
        except Exception as exc:
            return respond(identifier, dict(content=[dict(type="text", text=f"The tool failed: {type(exc).__name__}")], isError=True))
    return respond(identifier, error=(METHOD_NOT_FOUND, f"Method not found: {method}"))


def handle_line(connector, line):
    try:
        message = json.loads(line)
    except (ValueError, RecursionError):
        return respond(None, error=(PARSE_ERROR, "Parse error."))
    return handle(connector, message)


def serve(connector, source=None, sink=None):
    source = sys.stdin if source is None else source
    sink = sys.stdout if sink is None else sink
    for line in source:
        if not line.strip():
            continue
        reply = handle_line(connector, line[:MAX_LINE_BYTES])
        if reply is not None:
            sink.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sink.flush()


def main():
    history = os.environ.get("PYNQ_HISTORY_DIR") or HISTORY_DIR
    data = os.environ.get("PYNQ_DATA_DIR") or DATA_DIR
    serve(Connector(history, data))


if __name__ == "__main__":
    main()
