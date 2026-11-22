"""Replay the three synthetic diagnosis traces on a paired physical kit."""

import argparse
from hashlib import sha256
from http.client import HTTPException
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, sleep
from uuid import UUID, uuid4

from lm_reference.flan_prompt import telemetry_prompt
from pinkprocess.kit_evidence import analyst_evidence
from pinkprocess.kit_transport import BoardTransport, exchange
from pinkprocess.telemetry_stream import BOARD_DEMO_SOURCE, FIXTURE_SOURCE_PREFIX

REFERENCE = Path(__file__).with_name("evidence") / "three-diagnosis-traces-w8a8.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def demo(stream):
    return stream.get("origin") == "demo" and stream.get("source") == BOARD_DEMO_SOURCE


def idle(stream):
    return stream.get("state") == "idle" and stream.get("session") is None


def poll_board(board, job, timeout, *, interval=2, clock=monotonic, wait=sleep):
    end = clock() + timeout
    while clock() < end:
        result = board.request("lm_result", job_id=job["job_id"])
        if result["state"] == "complete":
            return result
        require(result["state"] == "running", result.get("error", "Board inference failed."))
        wait(interval)
    raise RuntimeError("Timed out waiting for board inference.")


def check_capture(job, question, reference, fixture, active):
    snapshot = job["snapshot"]
    expected = reference["snapshot"]
    require(job["active"] == active, "Board captured evidence with an unexpected language-model version.")
    require(snapshot["session"] == fixture["session"] and snapshot["source"] == fixture["source"]
            and snapshot["origin"] == "demo" and snapshot["state"] == "ready"
            and snapshot["fresh"], "Board captured unexpected telemetry.")
    require(all(snapshot[k] == expected[k] for k in ("capacity", "latest", "sample_count", "observations")),
            "Board capture does not match the frozen trace measurements.")
    forecast = snapshot.get("forecast") or {}
    require(all(forecast.get(k) == v for k, v in expected["forecast"].items()),
            "Board forecast does not match the frozen trace's integer reference.")
    prompt = telemetry_prompt(question, analyst_evidence(snapshot))
    require(prompt == reference["prompt"], "Board capture differs from the frozen software prompt.")
    captured_at = job.get("captured_at_ms")
    require(type(captured_at) is int and captured_at > 0, "Board capture timestamp is missing.")
    record = dict(question=question, captured_at_ms=captured_at, snapshot=snapshot, prompt=prompt)
    body = json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    require(job["capture_id"] == sha256(body).hexdigest(), "Board capture ID failed SHA-256 verification.")


def release(send, session, wait):
    for attempt in range(3):
        try:
            return send(session=session, stop=True)
        except (OSError, HTTPException, ValueError, RuntimeError):
            if attempt == 2:
                raise
            wait(.25)


def wait_for_producer(board, before, *, clock, wait):
    deadline = clock() + 20
    while clock() < deadline:
        status = board.request("stream_status")
        if demo(before) and demo(status) and status["fresh"] and status["state"] == "ready":
            return status
        if idle(before) and (idle(status) or demo(status)):
            return status
        if not (idle(status) or demo(status)):
            # A new producer may have connected while the fixture was being removed.
            # It owns the stream now. The next case will refuse to take it over.
            return status
        wait(.5)
    raise RuntimeError("SQLite demo did not resume after fixture release.")


def record_progress(path, report):
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            json.dump(report, output, indent=2)
            output.write("\n")
            output.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def verify(board, send, base, reference, report, *, timeout=7200, clock=monotonic, wait=sleep,
           progress=None):
    report["passed"] = False
    language = board.request("lm_versions")
    active = language["active"]
    require(base["backend"] == "PYNQ-Z2" and base.get("lm_supported"),
            "A paired PYNQ-Z2 with the language model is required.")
    require(base["version"] == reference["cases"][0]["snapshot"]["forecast"]["version"],
            "The board onset forecast profile differs from the frozen reference.")
    require(active["name"] == "original" and active["checkpoint_sha256"] == reference["weights_sha256"],
            "Select the frozen original language-model checkpoint first.")
    UUID(language["boot_id"])
    question = reference["question"]
    report.update(execution="paired PYNQ-Z2 with authenticated ingress", reference_sha256=sha256(
        json.dumps(reference, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(), checkpoint=language, telemetry_profile=base, question=question, cases=[])

    for expected in reference["cases"]:
        before = board.request("stream_status")
        require(before.get("fixture_guard_supported") is True,
                "The kit must support guarded synthetic telemetry ingress.")
        require(idle(before) or demo(before), "A non-demo producer owns the board. Leave it running.")
        source = FIXTURE_SOURCE_PREFIX + " " + expected["case"]
        fixture = dict(session=str(uuid4()), source=source, origin="demo",
                       capacity=expected["snapshot"]["capacity"])
        run = dict(case=expected["case"], before=before, fixture=fixture, expected_answer=expected["answer"])
        report["cases"].append(run)
        case_started = clock()
        try:
            sent = send(**fixture, fixture_after_session=before["session"], readings=expected["raw_samples"])
            run["accepted"] = sent
            require(sent["session"] == fixture["session"] and sent["state"] == "ready"
                    and sent["accepted"] == len(expected["raw_samples"]), "Board did not accept the frozen trace.")
            require(sent["fpga_submissions"] > before["fpga_submissions"],
                    "The board did not submit forecast work to the FPGA for this trace.")
            asked = clock()
            job = board.request("lm_start", question=question)
            run["started"] = job
            check_capture(job, question, expected, fixture, active)
        finally:
            run["released"] = release(send, fixture["session"], wait)
            run["producer_after_release"] = wait_for_producer(board, before, clock=clock, wait=wait)

        if progress:
            progress(f"{expected['case']} captured as {job['capture_id']}. Fixture released.")
        started = clock()
        result = poll_board(board, job, timeout, clock=clock, wait=wait)
        finished = clock()
        run.update(result=result, wait_seconds=round(finished - started, 3),
                   question_to_answer_seconds=round(finished - asked, 3),
                   ingest_to_answer_seconds=round(finished - case_started, 3))
        projections = result.get("fabric_projections")
        require(result["job_id"] == job["job_id"] and result["capture_id"] == job["capture_id"]
                and result["active"] == active and type(projections) is int and projections > 0,
                "Board answer did not verify its evidence, weights and PL projection count.")
        run["matches_reference"] = result["answer"] == expected["answer"]
        if progress:
            progress(f"{expected['case']} completed in {run['question_to_answer_seconds']:.1f}s.")

    final = board.request("lm_versions")
    report["final_checkpoint"] = final
    require(final["boot_id"] == language["boot_id"] and final["active"] == active
            and final["versions"] == language["versions"], "Board identity or weight versions changed during replay.")
    require(all(run["matches_reference"] for run in report["cases"]),
            "A physical answer differs from the frozen software reference.")
    report["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("results", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("Timeout must be positive.")
    reference = json.loads(args.reference.read_bytes())
    report = {"passed": False}
    board = BoardTransport()
    with args.results.open("x"):
        pass
    try:
        record_progress(args.results, report)
        status = board.connect(args.host)
        host, pairing = board.host, board.kit

        def send(**params):
            return exchange(host, pairing, params, 10, path="/v1/telemetry")

        def progress(message):
            record_progress(args.results, report)
            print(message, flush=True)

        verify(board, send, status, reference, report, timeout=args.timeout_seconds, progress=progress)
    except (OSError, HTTPException, ValueError, RuntimeError) as exc:
        report["error"] = str(exc)
        parser.exit(1, f"{exc}\n")
    finally:
        board.close()
        record_progress(args.results, report)
    print(f"Verified exact physical generation for {len(report['cases'])} synthetic traces.")


if __name__ == "__main__":
    main()
