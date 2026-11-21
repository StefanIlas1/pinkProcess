"""Record baseline, Astra edit and power-cycle replay on a paired physical kit."""
import argparse
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import monotonic, sleep, time
from uuid import UUID, uuid4

from pinkprocess.astra_edit import propose
from pinkprocess.kit_transport import BoardTransport


DEFAULT_QUESTION = "What should the next on-call engineer know about this pool before investigating?"
DEFAULT_INSTRUCTION = "Change the first token on the captured question to Now. Use the smallest verified sufficient increase."


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Verification:
    def __init__(self, board, report, *, interval=3, timeout=7200, author=propose):
        self.board = board
        self.path = report
        self.data = json.loads(report.read_text()) if report.exists() else {}
        self.interval = interval
        self.timeout = timeout
        self.author = author

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as output:
            json.dump(self.data, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(output.name, self.path)

    def status(self):
        status = self.board.request("lm_versions")
        require(isinstance(status.get("boot_id"), str), "Board did not report its Linux boot ID.")
        UUID(status["boot_id"])
        return status

    def ask(self, key, active, *, capture=None, measure=False):
        record = self.data.setdefault(key, {})
        if "started" not in record:
            job = self.board.request("lm_start", question=self.data["question"], capture_id=capture)
            require(job["active"] == active and (capture is None or job["capture_id"] == capture),
                    "Inference started with unexpected weights or a different capture.")
            record.update(started=job, started_unix=time(), live_samples=[])
            self.save()
        job = record["started"]
        if "result" not in record:
            deadline = monotonic() + self.timeout
            while monotonic() < deadline:
                stream = self.board.request("stream_status") if measure else None
                result = self.board.request("lm_result", job_id=job["job_id"])
                if result["state"] == "complete":
                    require(result["active"] == active and result["capture_id"] == job["capture_id"] and
                            result["fabric_projections"] > 0 and result["answer"].strip(),
                            "Inference result did not verify its weights, evidence, fabric usage or answer.")
                    record.update(result=result, elapsed_s=time() - record["started_unix"])
                    self.save()
                    print(f"{key} completed in {record['elapsed_s']:.1f}s: {result['answer']}", flush=True)
                    break
                require(result["state"] == "running", result.get("error", "Unknown inference state."))
                if stream:
                    sample = {k: stream[k] for k in ("session", "latest_time_ms", "fpga_submissions", "fresh")}
                    samples = record["live_samples"]
                    if len(samples) < 2:
                        samples.append(sample)
                    else:
                        samples[-1] = sample
                    self.save()
                sleep(self.interval)
            else:
                raise RuntimeError(f"{key} still running. Re-run this phase with the same report.")
        if measure:
            samples = record["live_samples"]
            require(len(samples) == 2 and samples[0]["session"] == samples[-1]["session"] and
                    all(s["fresh"] for s in samples) and
                    all(samples[-1][k] > samples[0][k] for k in ("latest_time_ms", "fpga_submissions")),
                    "Evidence did not establish telemetry progress while language inference was running.")
        if capture:
            require(job["snapshot"] == self.data["baseline"]["started"]["snapshot"],
                    "Board replay changed the recorded snapshot.")
        return record["result"]

    def before(self, board_status, question, name):
        status = self.status()
        if not self.data:
            require(status["active"]["name"] == "original", "Select original before starting a new acceptance run.")
            require(all(v["name"] != name for v in status["versions"]), "Choose a new edit name.")
            self.data.update(question=question, name=name, backend=board_status["backend"],
                             before_boot=status["boot_id"], initial=status, phase="before")
            self.save()
        require(self.data["before_boot"] == status["boot_id"], "Board rebooted before this phase finished.")
        require(self.data["phase"] == "before", "Pre-power-cycle phase is already complete.")
        base = self.data["initial"]["active"]
        baseline = self.ask("baseline", base, measure=True)
        require(not re.match(r"^Now\b", baseline["answer"]), "Baseline already begins with Now.")
        if "astra" not in self.data:
            row = self.board.request("lm_row", target_row="Now")
            require(row["calibration_job_id"] == baseline["job_id"] and
                    row["token"] == "Now" and row["base_sha256"] == base["checkpoint_sha256"] and
                    row.get("activation") and row.get("max_score") is not None and row.get("row_scale"),
                    "The board has no matching first-token calibration. Re-run the baseline.")
            response = self.author(DEFAULT_INSTRUCTION, row)
            self.data["astra"] = dict(model="gpt-6.2-astra", instruction=DEFAULT_INSTRUCTION,
                                      calibration=row, response=response)
            self.save()
        edit = self.data["astra"]
        saved = self.board.request("lm_save", name=self.data["name"], target_row=edit["calibration"]["target_row"],
                                   base_sha256=base["checkpoint_sha256"], response=edit["response"])
        selected = saved if saved["active"]["name"] == self.data["name"] else self.board.request(
            "lm_select", name=self.data["name"])
        target = selected["active"]
        require(target in saved["versions"] and target["name"] == self.data["name"] and
                all(target[k] != base[k] for k in ("checkpoint_sha256", "classifier_sha256")),
                "Selected edit did not change verified weights.")
        self.data["selected"] = selected
        self.save()
        edited = self.ask("edited", target, capture=baseline["capture_id"])
        require(re.match(r"^Now\b", edited["answer"]) and edited["answer"] != baseline["answer"],
                "The selected row failed to produce the intended first word.")
        self.data["phase"] = "awaiting_power_cycle"
        self.save()

    def after(self, witnessed=False):
        require(self.data.get("phase") in ("awaiting_power_cycle", "after"), "Complete the before phase first.")
        status = self.status()
        require(status["boot_id"] != self.data["before_boot"], "Linux boot ID is unchanged.")
        if "after_boot" not in self.data:
            require(status["active"] == self.data["selected"]["active"] and
                    status["versions"] == self.data["selected"]["versions"], "Stored versions did not reload exactly.")
            self.data.update(after_boot=status["boot_id"], after_restart=status, phase="after",
                             operator_witnessed_cold_power_cycle=witnessed)
            self.save()
        require(status["boot_id"] == self.data["after_boot"], "Board rebooted again during verification.")
        capture = self.data["baseline"]["result"]["capture_id"]
        reloaded = self.ask("reloaded", self.data["selected"]["active"], capture=capture)
        require(reloaded["answer"] == self.data["edited"]["result"]["answer"], "Edited answer changed after boot.")
        current = self.status()
        restored = current if current["active"]["name"] == "original" else self.board.request("lm_restore")
        require(restored["active"] == self.data["initial"]["active"] and
                restored["versions"] == self.data["selected"]["versions"], "Original weights or saved versions changed.")
        result = self.ask("restored", restored["active"], capture=capture)
        require(result["answer"] == self.data["baseline"]["result"]["answer"], "Rollback changed the baseline answer.")
        self.data.update(final=restored, phase="complete")
        self.save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["before", "after"])
    p.add_argument("host")
    p.add_argument("report", type=Path)
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--name", default=None)
    p.add_argument("--poll-seconds", type=float, default=3)
    p.add_argument("--timeout-seconds", type=float, default=7200)
    p.add_argument("--witnessed-cold-power-cycle", action="store_true",
                   help="Record that an operator physically removed and restored board power.")
    args = p.parse_args()
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        p.error("Timing arguments must be positive.")
    board = BoardTransport()
    try:
        status = board.connect(args.host)
        verification = Verification(board, args.report, interval=args.poll_seconds, timeout=args.timeout_seconds)
        if args.phase == "before":
            verification.before(status, args.question, args.name or "Astra Now " + str(uuid4())[:8])
        else:
            verification.after(args.witnessed_cold_power_cycle)
        print(verification.data["phase"], args.report)
    finally:
        board.close()


if __name__ == "__main__":
    main()
