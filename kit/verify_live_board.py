"""Acceptance on a paired kit running the optional local SQLite source.

Run with Mac operator windows closed. This briefly replaces the demo with a
known producer to verify the read/compute boundary, then releases the stream.
"""

import argparse
import json
from pathlib import Path
import time
import uuid

from pinkprocess.kit_dataset import prepare_dataset
from pinkprocess.kit_transport import BoardTransport, exchange
from pinkprocess.fixed_point import FixedProfile
from pinkprocess.profile import Profile

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify(host, output):
    board = BoardTransport()
    base = board.connect(host)
    report = {"backend": base["backend"], "onset_profile": base["version"]}
    before = board.request("stream_status")
    require(before["origin"] == "demo", "Run on a kit with the SQLite demo active.")
    board.close()
    time.sleep(7)
    board.connect(host)
    after = board.request("stream_status")
    require(after["session"] == before["session"] and after["fresh"] and
            after["latest_time_ms"] > before["latest_time_ms"] and
            after["fpga_submissions"] > before["fpga_submissions"], "No autonomous progress.")
    report["without_reader"] = [before, after]
    session = str(uuid.uuid4())
    def send(**params):
        return exchange(host, board.kit, dict(session=session, **params), 5, path="/v1/telemetry")
    fixture = dict(source="Known verification readings", capacity=4)
    rows = [dict(time_ms=i*100, busy=2, waiting=0) for i in range(50)]
    try:
        sent = send(**fixture, readings=rows)
        require(sent["state"] == "ready", "FPGA did not finish before the first question.")
        first = board.request("stream_forecast", session=session)
        second = board.request("stream_forecast", session=session)
        csv = "time_ms,busy,waiting\n"+"".join(f"{r['time_ms']},2,0\n" for r in rows)
        features = prepare_dataset(csv,4)["features"][-1]
        expected = FixedProfile.quantize(Profile.load(ROOT/"profiles"/(base["version"]+".json"))).score(features)
        require(first["forecast"]["score"] == expected, "Onset result differs from integer reference.")
        require(first["forecast"] == second["forecast"] and
                sent["fpga_submissions"] == first["fpga_submissions"] == second["fpga_submissions"],
                "A read submitted work to the FPGA.")
        row = dict(time_ms=5000,busy=4,waiting=3)
        appended = send(**fixture, readings=[row])
        final = board.request("stream_forecast",session=session)
        vector = prepare_dataset(csv+"5000,4,3\n",4,backlog=True)["features"]
        weights = json.loads((ROOT/"profiles/queue-outlook-v2.json").read_text())["weights"]
        require(final["forecast"]["scores"] == [sum(x*y for x,y in zip(vector,w)) for w in weights],
                "Backlog result differs from integer reference.")
        require(appended["fpga_submissions"] == sent["fpga_submissions"]+1, "Expected one more submission.")
        report["external_publisher"] = dict(accepted=sent, first=first, second=second, appended=appended, final=final)
    finally:
        send(stop=True)
    deadline = time.monotonic()+18
    while time.monotonic()<deadline:
        demo = board.request("stream_status")
        if demo["origin"]=="demo" and demo["state"]=="ready":
            break
        time.sleep(.5)
    require(demo["origin"]=="demo" and demo["state"]=="ready","Demo did not resume.")
    report["resumed"] = demo
    output.write_text(json.dumps(report,indent=2)+"\n")
    board.close()
    print("Autonomous FPGA submissions:",after["fpga_submissions"]-before["fpga_submissions"],
          "during",after["latest_time_ms"]-before["latest_time_ms"],"ms of stream time.")
    print("Verified producer-side scoring, two cached reads, both models and demo recovery.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("report", type=Path)
    args=parser.parse_args()
    verify(args.host,args.report)
