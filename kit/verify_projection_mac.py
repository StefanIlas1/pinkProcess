"""Exercise the native Mac backend and verify the board's exact diagnosis."""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

from lm_reference.flan import Flan
from lm_reference.flan_prompt import telemetry_prompt
from lm_reference.model_assets import file_digest, verify_model
from kit.mac_backend_client import MacBackend
from kit.verify_diagnosis_board import record_progress, release
from kit.verify_diagnosis_traces import make_forecaster
from pinkprocess.kit_evidence import analyst_evidence
from pinkprocess.kit_transport import BoardTransport, exchange
from pinkprocess.simulator import Reading, SAMPLE_MS
from pinkprocess.telemetry_stream import BOARD_DEMO_SOURCE, FIXTURE_SOURCE_PREFIX, WINDOW_SIZE

ROOT = Path(__file__).resolve().parents[1]
SAVED_REPORT = ROOT / "kit/evidence/three-diagnosis-traces-physical.json"
PROFILES = ROOT / "profiles"
MANIFEST = PROFILES / "flan-telemetry-diagnostic/model.json"
UNSEEN = (
    ("cleared", "Has the recent queue cleared? Tell me how many requests are waiting now "
     "and the peak waiting count. Keep the forecast separate from the measurements.",
     ((12, 4, 0), (13, 6, 8), (12, 6, 5), (12, 6, 2), (31, 3, 0))),
    ("active", "Are requests still backed up? Give the current and peak queue as counts, "
     "and say separately what the ten-second FPGA forecast expects.",
     ((14, 5, 0), (10, 6, 15), (12, 6, 10), (15, 6, 8), (29, 6, 7))),
    ("ambiguous", "Can we tell if long-held connections or an arrival spike caused the queue? "
     "Give the current queue count and one measurement that would distinguish those explanations.",
     ((13, 5, 0), (15, 6, 11), (15, 6, 6), (13, 6, 3), (24, 6, 4))),
)


def rows_for(segments):
    rows = []
    for count, busy, waiting in segments:
        for _ in range(count):
            rows.append({"time_ms": len(rows) * SAMPLE_MS, "busy": busy, "waiting": waiting})
    return rows


def lexical_rubric(name, answer):
    text = answer.casefold()
    checks = {
        "cleared": {"recognizes_clear": r"clear|no (?:active |current )?(?:backlog|requests waiting)",
                    "mentions_current_zero": r"\b0\b|no (?:requests? |one )?waiting",
                    "mentions_peak_eight": r"\b8\b"},
        "active": {"recognizes_active": r"active|ongoing|still backed up",
                   "mentions_current_seven": r"\b7\b",
                   "mentions_peak_fifteen": r"\b15\b",
                   "distinguishes_forecast": r"forecast|predict|expect|next ten|next 10"},
        "ambiguous": {"recognizes_cause_uncertainty": r"can(?:not|'t) tell|insufficient|unclear|does not (?:show|distinguish)|cannot distinguish|not enough",
                      "mentions_current_four": r"\b4\b",
                      "proposes_distinguishing_measurement": r"latenc|hold[ -]?time|duration|arrival|request rate|pool usage|acquisition|wait time"},
    }
    return {key: bool(re.search(pattern, text)) for key, pattern in checks[name].items()}


def review_rubric(name, answer, forecast):
    lexical = lexical_rubric(name, answer)
    reviewed = lexical.copy()
    note = None
    if name == "ambiguous" and not lexical["recognizes_cause_uncertainty"]:
        # The physical first pass exposed this limitation of the original matcher.
        if "the cause is unknown" in answer.casefold():
            reviewed["recognizes_cause_uncertainty"] = True
            note = "The original matcher missed the explicit wording 'The cause is unknown.'"
    if name == "cleared":
        reviewed["distinguishes_forecast"] = bool(
            re.search(r"FPGA warns.+next ten seconds", answer, re.I) and forecast["alert"])
    elif name == "active":
        reviewed["persistence_forecast_matches"] = (
            "remain nonempty" in answer.casefold() and forecast["scores"][0] >= 0)
    return lexical, reviewed, note

def small_stream(snapshot):
    return {key: snapshot[key] for key in ("state", "fresh", "source", "origin", "fpga_submissions",
                                           "sample_count")}


@contextmanager
def fixture_capture(mac, send, case):
    label, _, segments = case
    rows = rows_for(segments)
    before = mac.ready_stream()
    if (before["source"] != BOARD_DEMO_SOURCE or before["origin"] != "demo"
            or before.get("fixture_guard_supported") is not True):
        raise RuntimeError("The normal SQLite demo does not own the stream.")
    payload = {"session": str(uuid4()), "source": FIXTURE_SOURCE_PREFIX + " projection-cache " + label,
               "origin": "demo", "capacity": 6, "fixture_after_session": before["session"], "readings": rows}
    try:
        accepted = send(**payload)
        if (accepted["accepted"] != len(rows) or accepted["state"] != "ready"
                or accepted["session"] != payload["session"]
                or accepted["fpga_submissions"] <= before["fpga_submissions"]):
            raise RuntimeError("The FPGA did not accept the entire synthetic stream.")
        yield payload
    finally:
        release(send, payload["session"], sleep)


def verify_case(mac, question, active, capture_id=None, fixture=None):
    before = small_stream(mac.ready_stream())
    fields = {"question": question, **({"capture_id": capture_id} if capture_id else {})}
    with (fixture if fixture else nullcontext()) as payload:
        t0 = monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        started = mac.action("ask_begin", **fields)
        if payload:
            snapshot = started["snapshot"]
            rows = payload["readings"]
            expected_forecast = make_forecaster(PROFILES)(
                [Reading(**row) for row in rows[-WINDOW_SIZE:]], payload["capacity"])
            if (snapshot["session"] != payload["session"] or snapshot["source"] != payload["source"]
                    or snapshot["origin"] != "demo" or snapshot["capacity"] != payload["capacity"]
                    or snapshot["latest"] != rows[-1] or snapshot["sample_count"] != len(rows)
                    or snapshot["observations"]["peak_waiting"]["waiting"] != max(row["waiting"] for row in rows)
                    or not snapshot["fresh"] or snapshot["state"] != "ready"
                    or not all(snapshot["forecast"].get(k) == v for k, v in expected_forecast.items())):
                raise RuntimeError("Captured hardware telemetry does not match the synthetic trace and forecast.")
            prompt = telemetry_prompt(question, analyst_evidence(snapshot))
            record = dict(question=question, captured_at_ms=started["captured_at_ms"], snapshot=snapshot, prompt=prompt)
            digest = sha256(json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if digest != started["capture_id"]:
                raise RuntimeError("The board capture hash does not match the returned evidence.")
    if started.get("state") != "running" or started.get("model") != "flan-t5-small":
        raise RuntimeError("The Mac ask endpoint did not start a board-local FLAN job.")
    if started["active"] != active or (capture_id and started["capture_id"] != capture_id):
        raise RuntimeError("The Mac path returned an unexpected pinned capture or active weight.")
    print(f"Started capture {started['capture_id']}", flush=True)
    next_sample = t0 + 10
    first = None
    samples = []
    consecutive_errors = 0
    while monotonic() - t0 < 1200:
        try:
            result = mac.action("answer_progress", job_id=started["job_id"])
        except OSError:
            consecutive_errors += 1
            if consecutive_errors == 8:
                raise
            sleep(1)
            continue
        consecutive_errors = 0
        now = monotonic()
        if result["state"] != "running":
            break
        if first is None and result.get("partial_answer"):
            first = now - t0
        if now >= next_sample:
            snapshot = small_stream(mac.action("stream_status")["snapshot"])
            sample = {"elapsed_seconds": now - t0,
                      "learned_projections": result.get("fabric_projections"),
                      "accelerated_projections": result.get("accelerated_projections"),
                      "stream": snapshot}
            samples.append(sample)
            print(json.dumps(sample), flush=True)
            next_sample = now + 10
        sleep(1)
    else:
        raise TimeoutError(f"Still waiting for the original job {started['job_id']}")
    if result["state"] != "complete":
        raise RuntimeError(result.get("error", "Board generation failed."))
    after = small_stream(mac.action("stream_status")["snapshot"])
    elapsed = now - t0
    if (result["active"] != active or result["capture_id"] != started["capture_id"]
            or result.get("accelerated_projections") != result["fabric_projections"]
            or result["fabric_projections"] <= 0):
        raise RuntimeError("Generation did not use the expected capture, weights, or all-projection backend.")
    if after["fpga_submissions"] <= before["fpga_submissions"]:
        raise RuntimeError("No telemetry FPGA submissions were observed during the replay window.")
    print(f"Completed capture {started['capture_id']} in {elapsed:.3f} seconds.", flush=True)
    report = {"case": "saved" if capture_id else "unseen", "question": question,
              "utc_started": started_at, "capture_id": result["capture_id"],
              "active": active, "answer": result["answer"], "first_text_seconds": first,
              "complete_seconds": elapsed, "learned_projections": result["fabric_projections"],
              "accelerated_projections": result["accelerated_projections"], "telemetry_before": before,
              "telemetry_after": after, "samples": samples}
    if "projection_phases" in result:
        report["projection_phases"] = result["projection_phases"]
    return report, started["snapshot"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True, help="Loopback HTTP port of the running Mac app backend.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unseen", type=int, choices=range(4), default=3)
    parser.add_argument("--saved-runs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--release", help="Identify the installed board release in the output.")
    args = parser.parse_args()
    verify_model(args.model, file_digest(MANIFEST))
    with args.out.open("x"):
        pass
    prior = json.loads(SAVED_REPORT.read_bytes())
    saved = next(c for c in prior["cases"] if c["case"] == "ambiguous-cause")["result"]
    mac = MacBackend(args.port)
    language = mac.action("lm_versions")
    active = language["active"]
    if active != saved["active"] or file_digest(args.model / "weights.bin") != active["checkpoint_sha256"]:
        raise RuntimeError("Select the original, pinned checkpoint before verification.")
    report = {"transport": "native Mac app backend over USB loopback; one-second result polling",
              "board_release": args.release, "board_boot_id": language["boot_id"],
              "weights_before": language["versions"], "cases": [], "passed": False}
    record_progress(args.out, report)
    board = BoardTransport()
    if args.unseen:
        board.connect("127.0.0.1")

    def send(**params):
        return exchange(board.host, board.kit, params, 15, path="/v1/telemetry")

    for index in range(args.saved_runs):
        case, _ = verify_case(mac, prior["question"], active, saved["capture_id"])
        case["case"] = "saved-first" if index == 0 else "saved-repeat"
        expected = saved["answer"]
        case["exact_learned_projection_count"] = case["learned_projections"] == saved["fabric_projections"]
        case["matches_reference_bytes"] = expected.encode() == case["answer"].encode()
        case["reference_answer_sha256"] = sha256(expected.encode()).hexdigest()
        report["cases"].append(case)
        record_progress(args.out, report)
        if not case["matches_reference_bytes"] or not case["exact_learned_projection_count"]:
            raise RuntimeError("The saved replay differs from the frozen result.")
        if "projection_phases" not in case:
            raise RuntimeError("The board did not report the new driver phases.")

    for fixture in UNSEEN[:args.unseen]:
        name, question, segments = fixture
        case, snapshot = verify_case(mac, question, active, fixture=fixture_capture(mac, send, fixture))
        runtime = Flan(args.model, None, reference=True)
        expected = runtime.generate(telemetry_prompt(question, analyst_evidence(snapshot)))
        case.update(case=name, input="guarded deterministic synthetic telemetry", capacity=snapshot["capacity"],
                    latest=snapshot["latest"], peak_waiting=snapshot["observations"]["peak_waiting"],
                    forecast=snapshot.get("forecast"), raw_samples=rows_for(segments),
                    matches_reference_bytes=expected.encode() == case["answer"].encode(),
                    reference_answer_sha256=sha256(expected.encode()).hexdigest(),
                    capture_sha256_verified=True, integer_forecast_verified=True)
        lexical, reviewed, note = review_rubric(name, case["answer"], case["forecast"])
        case.update(initial_lexical_rubric=lexical, rubric=reviewed, rubric_review=note)
        report["cases"].append(case)
        record_progress(args.out, report)
        if not case["matches_reference_bytes"] or not all(case["rubric"].values()):
            raise RuntimeError("Board output differs from the reference or fails the case rubric.")
    current = mac.action("lm_versions")
    report["weights_after"] = current["versions"]
    report["weights_preserved"] = (current["active"] == active
                                   and current["versions"] == report["weights_before"]
                                   and current["boot_id"] == language["boot_id"])
    end_stream = mac.ready_stream()
    if args.unseen and (end_stream["source"] != BOARD_DEMO_SOURCE or end_stream["origin"] != "demo"):
        raise RuntimeError("The SQLite demo did not resume after the synthetic cases.")
    report["end_stream"] = small_stream(end_stream)
    report["passed"] = report["weights_preserved"]
    record_progress(args.out, report)
    if not report["weights_preserved"]:
        raise RuntimeError("The active or stored weights changed during testing.")


if __name__ == "__main__":
    main()
