"""Replay the three frozen CSV incidents through the installed USB-paired Mac backend."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic, sleep

from kit.mac_backend_client import MacBackend

BASELINE = Path(__file__).parent / "evidence/open-question-three-physical.json"


def replay(mac, name, case, model):
    csv = ",".join(case["input_columns"]) + "\n" + "".join(
        ",".join(map(str, row)) + "\n" for row in case["input_readings"]
    )
    if sha256(csv.encode()).hexdigest() != case["csv_sha256"]:
        raise ValueError(f"The {name} CSV differs from the frozen input.")
    dataset = mac.action("dataset_upload", name=name + ".csv", csv=csv, capacity=4)
    dataset_id = dataset["dataset_id"]
    if dataset["selected_model"] != model:
        mac.action("dataset_select", dataset_id=dataset_id, name=model)
    before = mac.ready_stream()
    start = monotonic()
    job = mac.action("dataset_ask", dataset_id=dataset_id, question=case["question"], stage="final")
    first_text = None
    while job["state"] == "running":
        if job.get("partial_answer") and first_text is None:
            first_text = monotonic() - start
        if monotonic() - start > 240:
            raise TimeoutError(f"{name} diagnosis exceeded four minutes.")
        sleep(0.5)
        job = mac.action("answer_progress", job_id=job["job_id"])
    elapsed = monotonic() - start
    if job["state"] != "complete":
        raise RuntimeError(job.get("error", f"{name} diagnosis failed."))
    after = mac.ready_stream()
    if job["dataset_id"] != dataset_id or job["active"]["name"] != model:
        raise RuntimeError("Board answered with a different dataset or model.")
    if not 0 < job["fabric_projections"] == job["accelerated_projections"]:
        raise RuntimeError("Learned projections did not all run on the FPGA.")
    if before["session"] != after["session"] or before["fpga_submissions"] >= after["fpga_submissions"]:
        raise RuntimeError("The same live stream did not advance during the query.")
    return dict(question=case["question"], csv_sha256=case["csv_sha256"], active=job["active"],
                forecast=dataset["snapshot"]["forecast"], answer=job["answer"],
                matches_original_answer=job["answer"] == case["views"]["final"],
                first_partial_seconds=first_text, complete_seconds=elapsed,
                live_submissions=after["fpga_submissions"] - before["fpga_submissions"],
                projections=job["accelerated_projections"], phases=job["projection_phases"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int, help="The Mac app's local HTTP port")
    parser.add_argument("--model", default="original", help="Saved board weight version")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mac = MacBackend(args.port)
    report = dict(started=datetime.now(timezone.utc).isoformat(), cases={},
                  method="Elapsed from Mac API request to first observed partial and completion; 0.5 s polling.")
    for name, case in json.loads(BASELINE.read_text())["cases"].items():
        report["cases"][name] = replay(mac, name, case, args.model)
        print(name, report["cases"][name]["answer"], flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
