"""Replay three deterministic FIFO workloads through the software kit diagnosis path."""

import argparse
from collections import deque
from heapq import heappop, heappush
import json
from pathlib import Path
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from kit.reference_forecast import ONSET_VERSION, OUTLOOK_VERSION, make_forecaster
from lm_reference.flan_prompt import telemetry_prompt
from lm_reference.model_assets import file_digest, verify_model
from pinkprocess.kit_evidence import analyst_evidence
from pinkprocess.simulator import SAMPLE_MS
from pinkprocess.telemetry_stream import TelemetryStream


CAPACITY = 8
END_MS = 12_000
PEAK_MS = 3_000
PEAK_WAITING = 19
QUESTION = (
    "A teammate just saw our pool panel spike. Would you treat this as an active backlog, "
    "and what do the latest and worst counts actually say? Keep the ten-second prediction "
    "separate and tell me what measurement would help figure out why it happened."
)


def workload(hold_ms, recover):
    """Yield groups of simultaneous arrivals with a common hidden hold time."""
    for time_ms in range(0, END_MS + 1, hold_ms):
        count = CAPACITY
        if time_ms == PEAK_MS:
            count += PEAK_WAITING
        elif time_ms == 5_000:
            count = 0
        elif recover and time_ms >= 9_000:
            count = 0 if time_ms < 9_300 else 4
        if count:
            yield time_ms, count, hold_ms


def sample(groups):
    """Replay arrivals and FIFO releases, sampling after all events at each timestamp."""
    releases = []
    waiting = deque()
    groups = iter(groups)
    incoming = next(groups, None)
    for time_ms in range(0, END_MS + 1, SAMPLE_MS):
        while releases and releases[0] <= time_ms:
            release_ms = heappop(releases)
            if waiting:
                heappush(releases, release_ms + waiting.popleft())
        if incoming and incoming[0] == time_ms:
            _, count, hold_ms = incoming
            for _ in range(count):
                if len(releases) < CAPACITY:
                    heappush(releases, time_ms + hold_ms)
                else:
                    waiting.append(hold_ms)
            incoming = next(groups, None)
        yield {"time_ms": time_ms, "busy": len(releases), "waiting": len(waiting)}


def capture(name, rows, forecaster):
    stream = TelemetryStream(forecaster, clock=lambda: 0.0)
    stream.ingest({"session": str(uuid5(NAMESPACE_URL, f"pinkprocess:diagnosis-traces:{name}")),
                   "source": "Synthetic FIFO connection pool", "origin": "demo", "capacity": CAPACITY,
                   "readings": rows})
    snapshot = stream.query({"op": "stream_evidence", "window_ms": 60_000})
    # The stream sets the production backend label even when instantiated in this software harness.
    snapshot["backend"] = "software-reference"
    return snapshot


def evaluate(directory, manifest_sha256, profiles):
    verify_model(directory, manifest_sha256)
    from lm_reference import flan, flan_prompt
    from pinkprocess import kit_evidence, telemetry_stream

    runtime = flan.Flan(directory, None, reference=True)
    forecaster = make_forecaster(profiles)
    settings = (
        ("cleared", 100, True),
        ("ongoing", 100, False),
        ("ambiguous-cause", 500, False),
    )
    cases = []
    for name, hold_ms, recover in settings:
        groups = list(workload(hold_ms, recover))
        rows = list(sample(groups))
        snapshot = capture(name, rows, forecaster)
        evidence = analyst_evidence(snapshot)
        prompt = telemetry_prompt(QUESTION, evidence)
        started = perf_counter()
        answer = runtime.generate(prompt)
        cases.append({
            "case": name,
            "private_workload": {"arrival_groups_fields": ["time_ms", "count", "connection_hold_ms"],
                                 "arrival_groups": groups, "total_requests": sum(row[1] for row in groups),
                                 "hold_ms": hold_ms},
            "raw_samples": rows,
            "snapshot": snapshot, "evidence": evidence, "prompt": prompt, "answer": answer,
            "seconds": round(perf_counter() - started, 4),
        })
    cleared, ongoing, ambiguous = cases
    assert cleared["raw_samples"][-1]["waiting"] == 0
    assert ongoing["raw_samples"][-1]["waiting"] > 0
    assert ongoing["raw_samples"] == ambiguous["raw_samples"]
    assert ongoing["prompt"] == ambiguous["prompt"]
    assert ongoing["answer"] == ambiguous["answer"]
    return {
        "execution": "FIFO trace and integer forecast replay with W8A8 software generation",
        "manifest_sha256": manifest_sha256,
        "weights_sha256": file_digest(directory / "weights.bin"),
        "runtime_sha256": file_digest(Path(flan.__file__)),
        "production_prompt_sha256": file_digest(Path(flan_prompt.__file__)),
        "evidence_adapter_sha256": file_digest(Path(kit_evidence.__file__)),
        "telemetry_stream_sha256": file_digest(Path(telemetry_stream.__file__)),
        "forecast_profile_sha256": {name: file_digest(profiles / f"{name}.json")
                                    for name in (ONSET_VERSION, OUTLOOK_VERSION)},
        "question": QUESTION,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--profiles", type=Path, default=Path(__file__).resolve().parents[1] / "profiles")
    args = parser.parse_args()
    if args.results.exists():
        parser.error("Choose a new evidence path to preserve the original result.")
    result = evaluate(args.model, args.manifest_sha256, args.profiles)
    with args.results.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    for case in result["cases"]:
        print(f"{case['case']}\n{case['answer']}")


if __name__ == "__main__":
    main()
