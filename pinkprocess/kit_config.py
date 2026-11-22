"""Read sign-in settings from .env files so the app works when launched from Finder."""

import os
from pathlib import Path
import sys


APP_SUPPORT = Path.home() / "Library" / "Application Support" / "PYNQ Process"
BUNDLED_NAME = "pynq.env"
PUBLIC_KEYS = ("PYNQ_AUTH0_DOMAIN", "PYNQ_AUTH0_CLIENT_ID")


def parse_env(text):
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def env_paths():
    paths = [APP_SUPPORT / ".env", Path(__file__).resolve().parent.parent / ".env"]
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        paths.append(Path(bundle) / BUNDLED_NAME)
    return paths


def load_env(paths=None, environ=None):
    environ = os.environ if environ is None else environ
    loaded = {}
    for path in env_paths() if paths is None else paths:
        try:
            values = parse_env(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        for key, value in values.items():
            if value and not environ.get(key) and key not in loaded:
                loaded[key] = value
    for key, value in loaded.items():
        environ[key] = value
    return loaded
