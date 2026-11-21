"""Fit on development runs, or evaluate frozen queue-outlook rows on held-out runs."""

import argparse
from bisect import bisect_right
import csv
import hashlib
import json
from math import exp
from pathlib import Path

from pinkprocess.backlog import FIT_INPUTS, FIT_VERSION, TARGETS, VERSION, features, fitted_features
from pinkprocess.kit_transport import BoardTransport
from pinkprocess.simulator import Reading


def examples(folder, encode=features):
    config = json.loads((folder.parent / "protocol.json").read_text())
    with (folder / "telemetry.csv").open() as f:
        samples = [Reading(**{k:int(v) for k,v in row.items()}) for row in csv.DictReader(f)]
    with (folder / "events.csv").open() as f:
        events = [(int(row["time_us"]), int(row["waiting"])) for row in csv.DictReader(f)]
    stamps = [s for s,_ in events]
    drains = [s for s,w in events if w == 0]
    result = []
    for i in range(50, len(samples)-100, 10):
        now = samples[i]
        if not now.waiting:
            continue
        start, end = now.time_ms * 1000, (now.time_ms + 10000) * 1000
        at = bisect_right(stamps, start)-1
        future = events[bisect_right(stamps, end)-1][1]
        if at < 0 or events[at][1] != now.waiting or future != samples[i+100].waiting:
            raise ValueError("Queue sample and event log disagree.")
        persists = bisect_right(drains, start) == bisect_right(drains, end)
        result.append((now.time_ms, encode(samples[i-49:i+1], config["capacity"]),
                       [persists, future > now.waiting]))
    return result


def fit(rows, target):
    X = [[v / 127 for v in x] for _,x,_ in rows]
    y = [int(labels[target]) for _,_,labels in rows]
    count = sum(y)
    if not 0 < count < len(y):
        raise ValueError("Training needs both outcomes.")
    coefficients = [0.] * len(FIT_INPUTS)
    for _ in range(1600):
        gradient = [0.] * len(FIT_INPUTS)
        for x,truth in zip(X,y):
            logit = sum(a*b for a,b in zip(x,coefficients))
            error = (1/(1+exp(-logit)) - truth) / (2*(count if truth else len(y)-count))
            for i,v in enumerate(x):
                gradient[i] += v*error
        coefficients = [c - .6*(g + (.01*c if i != len(FIT_INPUTS)-1 else 0))
                        for i,(c,g) in enumerate(zip(coefficients,gradient))]
    scale = max(abs(c) for c in coefficients)/127
    return [round(c/scale) for c in coefficients]


def evaluate(args):
    version = FIT_VERSION if args.fit else json.loads(args.profile.read_text())["version"]
    if version not in (VERSION, FIT_VERSION):
        raise ValueError("Unsupported queue-outlook version.")
    encode = fitted_features if version == FIT_VERSION else features
    runs = {str(folder):examples(folder, encode) for folder in args.runs}
    rows = [r for data in runs.values() for r in data]
    if args.fit:
        weights = [fit(rows, t) for t in range(2)]
        args.profile.write_text(json.dumps(dict(version=FIT_VERSION, inputs=FIT_INPUTS, targets=TARGETS,
                                               weights=weights), indent=2)+"\n")
        return
    model = json.loads(args.profile.read_text())
    transport = BoardTransport() if args.host else None
    if transport:
        transport.connect(args.host)
    counts = {t:dict(tp=0,tn=0,fp=0,fn=0) for t in TARGETS}
    trend_counts = {t:dict(tp=0,tn=0,fp=0,fn=0) for t in TARGETS}
    observations = []
    weight_hash = None
    for run,data in runs.items():
        with (Path(run) / "telemetry.csv").open() as file:
            depths = {int(r["time_ms"]):int(r["waiting"]) for r in csv.DictReader(file)}
        for t,x,truth in data:
            scores = [sum(a*b for a,b in zip(x,w)) for w in model["weights"]]
            if transport:
                hw = transport.request("backlog", features=x)
                if weight_hash is None:
                    weight_hash = hw["weights_sha256"]
                if hw["scores"] != scores or hw["version"] != version or hw["weights_sha256"] != weight_hash:
                    raise RuntimeError("Board result differs from the frozen model.")
            for name,s,y in zip(TARGETS,scores,truth):
                counts[name][("t" if (s>=0)==y else "f")+("p" if s>=0 else "n")] += 1
            # Extrapolate the latest four-second change at a constant slope.
            change = depths[t] - depths[t-4000]
            trend = [4*depths[t] + 10*change > 0, change > 0]
            for name,prediction,actual in zip(TARGETS,trend,truth):
                trend_counts[name][("t" if prediction==actual else "f")+("p" if prediction else "n")] += 1
            observations.append(dict(run=run,time_ms=t,truth=truth,scores=scores,trend=trend))
    if transport:
        transport.close()
    for v in (*counts.values(), *trend_counts.values()):
        v["balanced_accuracy"] = (v["tp"]/(v["tp"]+v["fn"])+v["tn"]/(v["tn"]+v["fp"]))/2
    report = dict(profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
                  device="PYNQ-Z2" if transport else "integer reference",
                  results=counts, trend_baseline=trend_counts, weights_sha256=weight_hash, observations=observations)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(len(rows), counts)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs",nargs="+",type=Path)
    p.add_argument("--profile",type=Path,required=True)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--fit",action="store_true")
    modes.add_argument("--report",type=Path)
    p.add_argument("--host")
    evaluate(p.parse_args())
