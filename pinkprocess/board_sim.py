"""A stand-in board for screenshots and interface work, enabled with PYNQ_SIM_BOARD=1."""

from __future__ import annotations

import csv
import io
import os
from time import monotonic
import uuid

from .simulator import SAMPLE_MS, simulate

HOST = "simulated.pynq"
CHECKPOINT = "0" * 64
FRESH_MS = 1500
ANSWER_SECONDS = float(os.environ.get("PYNQ_SIM_ANSWER_SECONDS", "6"))


class SimulatedBoard:
    def __init__(self, seed=11):
        self.run = simulate(seed, duration_ms=120_000, capacity=12)
        self.started = monotonic()
        self.version = "pool-logistic-v1"
        self.active = "original"
        self.versions = ["original"]
        self.jobs = {}
        self.datasets = {}

    def close(self):
        pass

    def discover(self):
        return [{"host": HOST, "name": "Simulated PYNQ-Z2"}]

    def connect(self, host):
        return self.status()

    def index(self):
        return int((monotonic() - self.started) * 1000 / SAMPLE_MS) % len(self.run.readings)

    def snapshot(self):
        position = self.index()
        window = self.run.readings[max(0, position - 300):position + 1]
        latest = window[-1]
        peak = max(window, key=lambda reading: reading.waiting)
        return dict(
            ok=True, state="ready", fresh=True, origin="demo", source="Simulated connection pool", session="sim-session",
            capacity=self.run.capacity, sample_count=len(window), age_ms=200,
            latest=dict(time_ms=latest.time_ms, busy=latest.busy, waiting=latest.waiting),
            observations=dict(sample_count=len(window), start_ms=window[0].time_ms, end_ms=latest.time_ms,
                              peak_waiting=dict(waiting=peak.waiting, time_ms=peak.time_ms)),
            forecast=dict(version="queue-outlook-v2"))

    def status(self):
        return dict(ok=True, backend="PYNQ-Z2", board_name="PYNQ-Z2", temp_c=46, model="Queue-onset logistic, INT8 on FPGA",
                    version=self.version, versions=["pool-logistic-v1", "pool-logistic-v2"], weights_sha256="1" * 64,
                    threshold=200, score_scale=65536, stream_supported=True, lm_supported=True,
                    stream_evidence_supported=True, backlog_version="queue-outlook-v2")

    def language_versions(self):
        return dict(ok=True, active=dict(name=self.active, checkpoint_sha256=CHECKPOINT),
                    versions=[dict(name=name) for name in self.versions], capture_listing_supported=True)

    def answer_text(self, snapshot, followup, question=""):
        latest, peak = snapshot["latest"], snapshot["observations"]["peak_waiting"]
        lead = "Following on from that, " if followup else ""
        text = (f"{lead}the pool has {latest['waiting']} requests waiting and {latest['busy']} of {snapshot['capacity']} "
                f"connections busy. The recent peak was {peak['waiting']} waiting. The FPGA forecast does not show the "
                "queue clearing in the next ten seconds. Measure how fast requests arrive and how long connections "
                "are held to find the cause." + (" " + " ".join(["Check the arrival rate against the connection hold time, then compare the peak with the pool capacity before changing anything."] * 6) if "long" in question.lower() else ""))
        return text[0].upper() + text[1:]

    def start_job(self, question="", followup=False):
        job_id = uuid.uuid4().hex
        snapshot = self.snapshot()
        self.jobs[job_id] = dict(started=monotonic(), snapshot=snapshot, text=self.answer_text(snapshot, followup, question))
        return dict(ok=True, job_id=job_id, state="running", snapshot=snapshot, partial_answer="")

    def job_result(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            return dict(ok=False, error="That job is unknown.")
        progress = 1.0 if ANSWER_SECONDS <= 0 else min(1.0, (monotonic() - job["started"]) / ANSWER_SECONDS)
        words = job["text"].split(" ")
        if progress < 1.0:
            return dict(ok=True, job_id=job_id, state="running", snapshot=job["snapshot"],
                        partial_answer=" ".join(words[:int(len(words) * progress)]))
        return dict(ok=True, job_id=job_id, state="complete", snapshot=job["snapshot"], answer=job["text"],
                    model="FLAN-T5-small INT8", capture_id=job_id, active=dict(name=self.active, checkpoint_sha256=CHECKPOINT),
                    baseline=dict(answer=job["text"], version=dict(name=self.active, checkpoint_sha256=CHECKPOINT)),
                    captured_at_ms=job["snapshot"]["latest"]["time_ms"])

    def upload(self, csv_text, name):
        rows = list(csv.reader(io.StringIO(csv_text)))[1:]
        times = [int(row[0]) for row in rows if row]
        waiting = [int(row[2]) for row in rows if row]
        dataset_id = uuid.uuid4().hex
        peak = max(range(len(waiting)), key=lambda index: waiting[index])
        dataset = dict(ok=True, dataset_id=dataset_id, models=[dict(name=name) for name in self.versions],
                       selected_model=self.active,
                       snapshot=dict(capacity=self.run.capacity, observations=dict(
                           sample_count=len(times), start_ms=times[0], end_ms=times[-1],
                           peak_waiting=dict(waiting=waiting[peak], time_ms=times[peak])),
                           latest=dict(time_ms=times[-1], busy=0, waiting=waiting[-1])))
        self.datasets[dataset_id] = dataset
        return dataset

    def request(self, action, **params):
        if action == "status":
            return self.status()
        if action in ("apply", "rollback"):
            self.version = params.get("version") or "pool-logistic-v1"
            return self.status()
        if action == "stream_status":
            return self.snapshot()
        if action in ("stream_observations", "stream_forecast", "stream_evidence", "peak", "trend"):
            return self.snapshot()
        if action == "lm_versions":
            return self.language_versions()
        if action == "lm_captures":
            return dict(ok=True, captures=[], before=None)
        if action in ("lm_select", "lm_restore"):
            self.active = params.get("name", "original") if action == "lm_select" else "original"
            return self.language_versions()
        if action == "lm_start":
            return self.start_job(params.get("question", ""))
        if action == "lm_result":
            return self.job_result(params.get("job_id"))
        if action == "lm_dataset_upload":
            return self.upload(params["csv"], params.get("name", "upload"))
        if action in ("lm_dataset_models", "lm_dataset_select"):
            dataset = self.datasets.get(params.get("dataset_id"))
            if dataset is None:
                return dict(ok=False, error="This dataset is unavailable on the paired board.")
            if action == "lm_dataset_select" and params.get("name") in self.versions:
                dataset["selected_model"] = params["name"]
            return dataset
        if action == "lm_dataset_ask":
            if params.get("dataset_id") not in self.datasets:
                return dict(ok=False, error="This dataset is unavailable on the paired board.")
            return self.start_job(params.get("question", ""), followup=params.get("stage") == "followup")
        return dict(ok=False, error="The simulated board does not support this action.")
