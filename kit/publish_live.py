"""Publish newline-flushed connection-pool CSV readings directly to a paired kit."""

import argparse
import csv
import json
from pathlib import Path
import sys
import uuid

from pinkprocess.kit_transport import exchange, validate_host


def publish(rows, send, source, capacity, session=None):
    reader = csv.DictReader(rows)
    if reader.fieldnames != ["time_ms", "busy", "waiting"]:
        raise ValueError("Use the CSV header time_ms,busy,waiting.")
    session = str(uuid.UUID(session)) if session else str(uuid.uuid4())
    print(f"Producer session {session}", file=sys.stderr, flush=True)
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"Incomplete reading at row {reader.line_num}.")
        reading = {k:int(v) for k,v in row.items()}
        send(dict(session=session, source=source, capacity=capacity, readings=[reading]))
    # Keep the final observation available; the board marks it stale on timeout.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Board hostname or address")
    parser.add_argument("--pairing", type=Path, required=True, help="Private pairing JSON for this kit")
    parser.add_argument("--source", required=True, help="Name shown to operators")
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--session", help="Resume the same producer session UUID")
    args = parser.parse_args()
    host = validate_host(args.host)
    credentials = json.loads(args.pairing.read_text())
    def send(payload):
        return exchange(host, credentials, payload, 5, path="/v1/telemetry")
    publish(sys.stdin, send, args.source, args.capacity, args.session)


if __name__ == "__main__":
    main()
