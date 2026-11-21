"""Replay held-out traces through the Mac WebKit result view and physical board."""

import argparse
from bisect import bisect_right
from collections import Counter
import csv
import json
import hashlib
from pathlib import Path
import statistics
import subprocess
from threading import Thread

from kit.evaluate_pool import ROOT, VERSIONS, load_trace
from pinkprocess.fixed_point import FixedProfile, encode_ratio
from pinkprocess.kit_app import make_server
from pinkprocess.kit_dataset import prepare_dataset
from pinkprocess.profile import Profile

CANDIDATE = "service-b-with-arrival-rate"


def arrival_times(folder):
    with (folder / "events.csv").open() as file:
        return [int(r["time_us"]) for r in csv.DictReader(file)
                if r["event"] in ("acquire", "enqueue")]


def rate_preparer(arrivals):
    def prepare(data, capacity):
        batch = prepare_dataset(data, capacity)
        for t, vector in zip(batch["times_ms"], batch["features"]):
            count = bisect_right(arrivals, t * 1000) - bisect_right(arrivals, (t - 1000) * 1000)
            # 160 ms is the fixed workload mean specified before the holdout.
            pressure = min(256, encode_ratio(160 * count, 1000 * capacity))
            vector[2] = max(vector[2], pressure)
        return batch
    return prepare


def measure(directory, host, executable, condition, seed, verify_saved=False):
    config = json.loads((directory / "protocol.json").read_text())
    folder = directory / str(seed)
    output = folder / (condition + "-display.json")
    inputs = folder / "telemetry.csv"
    prepare = rate_preparer(arrival_times(folder)) if condition == CANDIDATE else prepare_dataset
    version = VERSIONS[1] if condition == CANDIDATE else condition
    with make_server() as server:
        server.kit.prepare = prepare
        kit = server.kit
        active = kit.action({"action":"connect", "host":host})["status"]
        original = active["version"]
        original_hash = active["weights_sha256"]
        try:
            if original != version:
                active = kit.action({"action":"apply","version":version})["status"]
            if not verify_saved:
                worker = Thread(target=server.serve_forever, daemon=True)
                worker.start()
                try:
                    subprocess.run([str(executable.resolve()), f"http://127.0.0.1:{server.server_port}/",
                                    str(inputs), str(output), str(config["capacity"])], timeout=85, check=True)
                finally:
                    server.shutdown()
                    worker.join()
            observed = json.loads(output.read_text())
            batch = prepare(inputs.read_text(), config["capacity"])
            fixed = FixedProfile.quantize(Profile.load(ROOT / "profiles" / (version+".json")))
            expected_times = list(range(5000, 50001, 1000))
            if [r["time_ms"] for r in observed] != expected_times:
                raise ValueError("Incomplete display replay.")
            for row in observed:
                idx = bisect_right(batch["times_ms"], row["time_ms"]) - 1
                if idx < 0:
                    raise ValueError("Missing eligible window.")
                score = fixed.score(batch["features"][idx])
                if row["hash"] != active["weights_sha256"] or row["version"] != version or row["score"] != score:
                    raise ValueError("Displayed board result differs from the frozen integer reference.")
                if batch["times_ms"][idx] != row["time_ms"]:
                    prefix = "Requests are already waiting"
                else:
                    prefix = "Queue warning" if score >= fixed.threshold else "No queue warning"
                if not row["text"].startswith(prefix):
                    raise ValueError("Displayed answer disagrees with the reference.")
        finally:
            restored = kit.action({"action":"apply","version":original})["status"] if original != version else kit.board("status")
            if restored["version"] != original or restored["weights_sha256"] != original_hash:
                raise RuntimeError("Original model was not restored.")
        print(f"Verified {len(observed)} displayed answers for {condition}, run {seed}.", flush=True)


def score(starts, rows):
    leads = {}
    false = late = 0
    prior_positive = False
    for t, delay, warning in rows:
        if warning and not prior_positive:
            idx = bisect_right(starts, t)
            if idx == len(starts) or starts[idx] > t + 10000:
                false += 1
            else:
                lead = starts[idx] - (t + delay)
                if lead <= 0:
                    late += 1
                else:
                    leads[idx] = max(leads.get(idx, 0), lead)
        prior_positive = warning
    return dict(episodes=len(starts), warned=len(leads), missed=len(starts)-len(leads),
                early=sum(v >= 1000 for v in leads.values()),
                false_alerts=false, alerts_arriving_after_onset=late)


def summarize(directory):
    config = json.loads((directory / "protocol.json").read_text())
    report = {"conditions":{}, "seeds":config["seeds"], "method":
              "One update per second, 50 ms input phase. Shipping WebKit result view and live FPGA requests. Full-pool comparator uses the Service B display timestamps."}
    report["input_sha256"] = {}
    for seed in config["seeds"]:
        for name in ("telemetry.csv", "events.csv"):
            path = directory / str(seed) / name
            report["input_sha256"][f"{seed}/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["acceptance_sha256"] = hashlib.sha256((directory / "acceptance.md").read_bytes()).hexdigest()
    for condition in (*VERSIONS, CANDIDATE, "at-capacity"):
        per_run, delays = {}, []
        early_reachable = 0
        for seed in config["seeds"]:
            folder = directory / str(seed)
            run, _, busy, _ = load_trace(folder, config)
            source = VERSIONS[1] if condition == "at-capacity" else condition
            snapshots = json.loads((folder / (source+"-display.json")).read_text())
            queued = {r["time_ms"] for r in snapshots if r["text"].startswith("Requests are already waiting")}
            rows = []
            for r in snapshots:
                t = r["time_ms"]
                warning = (busy[t] == run.capacity and t not in queued
                           if condition == "at-capacity" else r["text"].startswith("Queue warning"))
                rows.append((t, r["display_delay_ms"], warning))
                delays.append(r["display_delay_ms"])
            starts = [ep.start_ms for ep in run.episodes if ep.start_ms > 5000]
            per_run[seed] = score(starts, rows)
            reachable = set()
            for t, delay, _ in rows:
                index = bisect_right(starts, t)
                if (t not in queued and index < len(starts)
                        and t + delay + 1000 <= starts[index] <= t + 10000):
                    reachable.add(index)
            early_reachable += len(reachable)
        totals = Counter()
        for metrics in per_run.values():
            totals.update(metrics)
        report["conditions"][condition] = dict(total=dict(totals), per_run=per_run,
                                               median_display_ms=statistics.median(delays),
                                               eligible_for_one_second_lead=early_reachable)
    results = report["conditions"]
    candidate = results[CANDIDATE]
    baseline = results[VERSIONS[1]]
    wins = sum(candidate["per_run"][s]["early"] > baseline["per_run"][s]["early"] for s in config["seeds"])
    report["promote"] = (wins >= 2 and candidate["total"]["false_alerts"] <= baseline["total"]["false_alerts"]
                         and all(candidate["total"]["early"] > results[c]["total"]["early"]
                                 for c in (*VERSIONS,"at-capacity")))
    (directory / "delivery-report.json").write_text(json.dumps(report, indent=2)+"\n")
    print({c:r["total"] for c,r in results.items()}, "promote:",report["promote"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--all", action="store_true", help="Run all conditions and summarize")
    p.add_argument("--verify-saved", action="store_true", help="Recheck a completed WebKit capture")
    p.add_argument("--host")
    p.add_argument("--window", type=Path)
    p.add_argument("--condition", choices=(*VERSIONS,CANDIDATE))
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    if args.all:
        protocol = args.directory / "acceptance.md"
        if not protocol.exists():
            protocol.write_bytes((ROOT / "kit/forecast_acceptance.md").read_bytes())
        if not (args.host and args.window) or args.condition or args.seed:
            p.error("--all requires --host and --window and excludes --condition and --seed.")
        config = json.loads((args.directory / "protocol.json").read_text())
        for condition in (*VERSIONS, CANDIDATE):
            for seed in config["seeds"]:
                measure(args.directory, args.host, args.window, condition, seed, args.verify_saved)
        summarize(args.directory)
    elif args.condition:
        if not (args.host and args.window and args.seed):
            p.error("Measurement requires --host, --window, and --seed.")
        measure(args.directory,args.host,args.window,args.condition,args.seed,args.verify_saved)
    else:
        summarize(args.directory)
