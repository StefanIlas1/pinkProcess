"""Replay recorded pool telemetry on the paired FPGA and score queue-event warnings."""

import argparse
import csv
import hashlib
import json
import statistics
from bisect import bisect_right
from dataclasses import asdict
from pathlib import Path

from pinkprocess.baseline import Results, score_alerts
from pinkprocess.dataset import FORECAST_MS, OBSERVATION_MS
from pinkprocess.fixed_point import FixedProfile
from pinkprocess.kit_dataset import prepare_dataset
from pinkprocess.kit_transport import BoardTransport
from pinkprocess.profile import Profile
from pinkprocess.simulator import Run, WaitEpisode

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("pool-logistic-v1", "service-b-astra-v1")


def load_trace(folder, config):
    episodes, previous, completed = [], 0, 0
    stamps, states = [], []
    with (folder / "events.csv").open() as file:
        for row in csv.DictReader(file):
            waiting = int(row["waiting"])
            micros = int(row["time_us"])
            stamp = micros / 1000
            busy_count = int(row["busy"])
            if ((stamps and micros < stamps[-1]) or not 0 <= busy_count <= config["capacity"]
                    or waiting < 0 or (waiting and busy_count != config["capacity"])):
                raise ValueError("Invalid transition in the measured pool event log.")
            stamps.append(micros)
            states.append((busy_count, waiting))
            if waiting and not previous:
                episodes.append(WaitEpisode(stamp, None))
            if previous and not waiting:
                episodes[-1] = WaitEpisode(episodes[-1].start_ms, stamp)
            if row["event"] in ("handoff", "release"):
                completed += 1
            previous = waiting
    duration = config["duration_seconds"] * 1000
    run = Run(int(folder.name), duration, config["capacity"], (), (),
              tuple(ep for ep in episodes if ep.start_ms <= duration))
    csv_text = (folder / "telemetry.csv").read_text()
    prepared = prepare_dataset(csv_text, run.capacity)
    with (folder / "telemetry.csv").open() as file:
        busy = {}
        for row in csv.DictReader(file):
            t = int(row["time_ms"])
            idx = bisect_right(stamps, t * 1000) - 1
            logged = states[idx] if idx >= 0 else (0, 0)
            if logged != (int(row["busy"]), int(row["waiting"])):
                raise ValueError(f"Queue log and telemetry disagree at {t} ms.")
            busy[t] = logged[0]
    if not states or states[-1] != (0, 0) or sorted(busy) != list(range(0, duration + 1, config["sample_ms"])):
        raise ValueError("The recorded run is incomplete.")
    windows = [(t, f) for t, f in zip(prepared["times_ms"], prepared["features"])
               if OBSERVATION_MS <= t <= duration - FORECAST_MS]
    return run, windows, busy, completed


def summarize(results, rows):
    totals = Results(0, [], 0, 0)
    for part in results:
        totals.episodes += part.episodes
        totals.false_alerts += part.false_alerts
        totals.repeated_alerts += part.repeated_alerts
        totals.lead_times_ms.extend(part.lead_times_ms)
    leads = totals.lead_times_ms
    return dict(episodes=totals.episodes, warned=totals.warned, missed=totals.missed,
                false_alerts=totals.false_alerts, repeat_alerts=totals.repeated_alerts,
                median_lead_ms=statistics.median(leads) if leads else None,
                warned_at_least_one_second=sum(t >= 1000 for t in leads),
                alert_windows=sum(rows), windows=len(rows), event_f1=totals.event_f1)


def evaluate(directory, host):
    config = json.loads((directory / "protocol.json").read_text())
    traces = [load_trace(directory / str(seed), config) for seed in config["seeds"]]
    transport = BoardTransport()
    initial = transport.connect(host)
    report = {"backend": initial["backend"], "host": host, "initial_version": initial["version"],
              "runs": {}, "conditions": {}}
    scores = []
    for run, windows, busy, completed in traces:
        starts = [ep.start_ms for ep in run.episodes if ep.start_ms > OBSERVATION_MS]
        reachable = set()
        for timestamp, _ in windows:
            index = bisect_right(starts, timestamp)
            if index < len(starts) and starts[index] <= timestamp + FORECAST_MS:
                reachable.add(starts[index])
        report["runs"][run.seed] = dict(
            transactions=completed, windows=len(windows),
            episodes_with_eligible_prior_window=len(reachable),
            observed_episodes=[asdict(ep) for ep in run.episodes],
        )
    active = initial
    try:
        for version in (*VERSIONS, "at-capacity"):
            results, alerts = [], []
            per_run = {}
            if version in VERSIONS:
                if active["version"] != version:
                    active = transport.request("apply", version=version)
                profile_path = ROOT / "profiles" / (version + ".json")
                fixed = FixedProfile.quantize(Profile.load(profile_path))
            for run, windows, busy, _ in traces:
                times = [t for t, _ in windows]
                if version == "at-capacity":
                    decisions = [busy[t] == run.capacity for t in times]
                else:
                    decisions = []
                    for start in range(0, len(windows), 512):
                        chunk = windows[start:start + 512]
                        features = [v for _, v in chunk]
                        result = transport.request("query_batch", features=features)
                        expected = [fixed.score(v) for v in features]
                        if (result["version"] != version or result["weights_sha256"] != active["weights_sha256"]
                                or result["scores"] != expected or result["threshold"] != fixed.threshold
                                or result["alerts"] != [s >= fixed.threshold for s in expected]):
                            raise RuntimeError("Board identity, scores or decisions disagree with the frozen profile.")
                        decisions.extend(result["alerts"])
                        scores.extend((run.seed, version, t, score, int(alert)) for
                                      (t, _), score, alert in zip(chunk, result["scores"], result["alerts"]))
                metrics = score_alerts(run, zip(times, decisions), 1)
                results.append(metrics)
                alerts.extend(decisions)
                per_run[run.seed] = summarize([metrics], decisions)
            condition = dict(total=summarize(results, alerts), per_run=per_run)
            if version in VERSIONS:
                condition.update(weights_sha256=active["weights_sha256"],
                                 profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest())
            report["conditions"][version] = condition
    finally:
        try:
            if transport.request("status")["version"] != initial["version"]:
                transport.request("apply", version=initial["version"])
            restored = transport.request("status")
            if restored["weights_sha256"] != initial["weights_sha256"] or restored["version"] != initial["version"]:
                raise RuntimeError("Initial board model was not restored.")
            report["restored_weights_sha256"] = restored["weights_sha256"]
        finally:
            transport.close()
    with (directory / "scores.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("seed", "version", "time_ms", "fpga_score", "warning"))
        writer.writerows(scores)
    report["matched_board_scores"] = len(scores)
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for model, metrics in report["conditions"].items():
        print(model, metrics["total"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("host")
    args = parser.parse_args()
    evaluate(args.directory, args.host)
