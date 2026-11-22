#!/usr/bin/env python3
"""Check signed projection results using installed FLAN tensors during board maintenance."""

import argparse
from contextlib import closing
from hashlib import sha256
import json
import math
from pathlib import Path
import signal
import sqlite3
import subprocess
from time import perf_counter_ns

import numpy as np

from fpga.check_classifier import board_lock
from kit.benchmark_projection_overlay import interrupt, offline, reset_after_failure, run_projection
from lm_reference.projection_overlay import (
    MAX_SEGMENT_BUFFER_BYTES,
    OverlayProjection,
    pack_projection,
    projection_mmio,
    unpack_projection,
)

TENSORS = (
    ("attention_q", "encoder.block.0.layer.0.SelfAttention.q.weight", (384, 512)),
    ("ffn_up", "encoder.block.0.layer.1.DenseReluDense.wi_0.weight", (1024, 512)),
    ("ffn_down", "encoder.block.0.layer.1.DenseReluDense.wo.weight", (512, 1024)),
    ("head", "lm_head.weight", (32128, 512)),
)
MEMORY_FIELDS = ("MemTotal", "MemAvailable", "CmaTotal", "CmaFree")
ENCODER_INPUT_LIMIT = 512


def memory_kib(required=False):
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in MEMORY_FIELDS:
            fields[name] = int(value.split()[0])
    if required and set(fields) != set(MEMORY_FIELDS):
        raise RuntimeError("Linux did not report the expected memory and CMA counters.")
    return fields


def memory_sample(required=False):
    if required or Path("/proc/meminfo").exists():
        return memory_kib(required)
    return {}


def dma_bytes(rows, depth, batch):
    positions, outputs = (1, 64) if batch == 1 else (8, 8)
    pos_tiles = math.ceil(batch / positions)
    out_tiles = math.ceil(rows / outputs)
    return {
        "activation": pos_tiles * depth * 8,
        "weights": out_tiles * depth * outputs,
        "scores": pos_tiles * out_tiles * 64 * 4,
    }


def budget(rows, depth, batch, available):
    buffers = dma_bytes(rows, depth, batch)
    total = sum(buffers.values())
    return {
        "batch": batch,
        "dma_buffer_bytes": buffers,
        "dma_total_bytes": total,
        "dma_largest_buffer_bytes": max(buffers.values()),
        "cma_free_minus_dma_bytes": (
            available["CmaFree"] * 1024 - total if "CmaFree" in available else None
        ),
    }


def load_weights(model_dir, tensor, shape, manifest):
    entry = manifest["tensors"][tensor]
    if np.dtype(entry["dtype"]) != np.dtype(np.int8) or tuple(entry["shape"]) != shape:
        raise ValueError(f"Unexpected tensor geometry or dtype for {tensor}.")
    return np.memmap(model_dir / "weights.bin", offset=entry["offset"], shape=shape,
                     mode="r", dtype=np.int8)


def head_patch(weights, db, name, deterministic):
    if deterministic:
        row = 852
        payload = (weights[row].view(np.uint8) ^ np.uint8(128)).tobytes()
        return row, payload, {
            "name": "deterministic signed row change",
            "row": row,
            "payload_sha256": sha256(payload).hexdigest(),
        }

    db = db.resolve(strict=True)
    if Path(f"{db}-wal").exists() or Path(f"{db}-journal").exists():
        raise RuntimeError("Run only on a clean, quiescent store with no SQLite journal sidecar.")
    with closing(sqlite3.connect(f"{db.as_uri()}?mode=ro&immutable=1", uri=True)) as connection:
        connection.execute("PRAGMA query_only = ON")
        state = connection.execute(
            "SELECT base_classifier_sha256, classifier_rows, classifier_width "
            "FROM model_state WHERE singleton=1"
        ).fetchone()
        version = connection.execute(
            "SELECT target_row, payload, payload_sha256, classifier_sha256 "
            "FROM versions WHERE name=?", (name,)
        ).fetchone()
    if state is None or (state[1], state[2]) != weights.shape or state[0] != sha256(weights).hexdigest():
        raise ValueError("The stored version does not match the model's base classifier.")
    if version is None:
        raise ValueError(f"Stored version is unavailable: {name}")
    row, payload, row_sha, classifier_sha = version
    if not 0 <= row < len(weights) or len(payload) != weights.shape[1] or sha256(payload).hexdigest() != row_sha:
        raise ValueError("Stored patch row failed shape or SHA-256 validation.")
    digest = sha256()
    digest.update(weights[:row])
    digest.update(payload)
    digest.update(weights[row + 1:])
    if digest.hexdigest() != classifier_sha:
        raise ValueError("Patched classifier contents differ from the stored version.")
    return row, payload, {
        "name": name,
        "row": row,
        "payload_sha256": row_sha,
        "classifier_sha256": classifier_sha,
    }


def signed_reference(x, weights):
    """Bound the int32 weight scratch even for the full output head."""
    x32 = x.astype(np.int32)
    expected = np.empty((len(x), len(weights)), dtype=np.int32)
    for start in range(0, len(weights), 256):
        stop = min(len(weights), start + 256)
        expected[:, start:stop] = x32 @ weights[start:stop].astype(np.int32).T
    return expected


def check_case(label, tensor, weights, x, run, hardware, patch=None):
    rows, depth = weights.shape
    samples = {"before_reference": memory_sample(hardware)}
    now = perf_counter_ns()
    expected = signed_reference(x, weights)
    reference_ns = perf_counter_ns() - now
    patch_report = None
    if patch is not None:
        row, payload, patch_report = patch
        patched = np.frombuffer(payload, dtype=np.int8)
        original_score = int(expected[0, row])
        patched_score = int(x[0].astype(np.int32) @ patched.astype(np.int32))
        if patched_score == original_score:
            raise RuntimeError("The synthetic activation did not distinguish the original and patch row.")
        expected[0, row] = patched_score
        patch_report = {**patch_report, "original_score": original_score, "score": patched_score}

    start = perf_counter_ns()
    a, w = pack_projection(weights, x)
    if patch_report is not None:
        # Head uses decoder layout, one packed lane per output row.
        w[row // 64, :, row % 64] = patched
    pack_ns = perf_counter_ns() - start
    samples["packed"] = memory_sample(hardware)
    if hardware:
        buffers = dma_bytes(rows, depth, len(x))
        if samples["packed"]["CmaFree"] * 1024 < sum(buffers.values()):
            raise RuntimeError(f"Insufficient reported CMA for {label}.")
    scores, phases = run(a, w, len(x) == 1, samples)
    unpack_start = perf_counter_ns()
    actual = unpack_projection(scores, len(x), rows)
    unpack_ns = perf_counter_ns() - unpack_start
    if not np.array_equal(actual, expected):
        batch_index, row_index = map(int, np.argwhere(actual != expected)[0])
        raise RuntimeError(
            f"{label} batch {batch_index} row {row_index}: "
            f"actual {actual[batch_index, row_index]}, expected {expected[batch_index, row_index]}"
        )
    samples["after_compare"] = memory_sample(hardware)
    return {
        "case": label,
        "tensor": tensor,
        "shape": {"rows": rows, "depth": depth, "batch": len(x)},
        "integer_reference_matched": True,
        "scores_checked": actual.size,
        "timing_ns": {"reference": reference_ns, "pack": pack_ns, **phases, "unpack": unpack_ns},
        "memory_kib": samples,
        "dma": budget(rows, depth, len(x), samples["packed"]),
        "head_edit": patch_report,
    }


def check_segmented_head(label, tensor, weights, x, project, hardware, patch_report=None):
    rows, depth = weights.shape
    samples = {"before_reference": memory_sample(hardware)}
    start = perf_counter_ns()
    expected = signed_reference(x, weights)
    reference_ns = perf_counter_ns() - start
    rows_per_segment = MAX_SEGMENT_BUFFER_BYTES // (64 * depth) * 64
    samples["before_projection"] = memory_sample(hardware)
    segment = budget(min(rows_per_segment, rows), depth, len(x), samples["before_projection"])
    if hardware and segment["cma_free_minus_dma_bytes"] < 0:
        raise RuntimeError(f"Insufficient reported CMA for a segment of {label}.")
    actual, elapsed_ns, uploads = project(tensor.removesuffix(".weight"), weights, x, samples)
    if len(uploads) != math.ceil(rows / rows_per_segment):
        raise RuntimeError(f"Unexpected number of segmented uploads for {label}: {len(uploads)}.")
    if not np.array_equal(actual, expected):
        bad = int(np.flatnonzero(actual != expected)[0])
        raise RuntimeError(f"{label} row {bad}: actual {actual[0, bad]}, expected {expected[0, bad]}.")
    samples["after_compare"] = memory_sample(hardware)
    return {
        "case": label, "tensor": tensor,
        "shape": {"rows": rows, "depth": depth, "batch": len(x)},
        "integer_reference_matched": True, "scores_checked": actual.size,
        "timing_ns": {"reference": reference_ns, "segmented_projection": elapsed_ns},
        "memory_kib": samples, "max_segment_dma": segment, "segments": uploads,
        "head_edit": patch_report,
    }


def all_cases(args, run, segmented):
    manifest = json.loads((args.model_dir / "model.json").read_text())
    rng = np.random.default_rng(442)
    memory = memory_sample(bool(args.bitstream))
    report = {
        "backend": "pynq_projection" if args.bitstream else "software_tile_model",
        "activation_source": "deterministic signed INT8 samples",
        "model_dir": str(args.model_dir),
        "initial_memory_kib": memory,
        "encoder_memory_at_input_limit": {},
        "cases": [],
    }
    for label, tensor, shape in TENSORS:
        weights = load_weights(args.model_dir, tensor, shape, manifest)
        batch = 1 if label == "head" else args.encoder_batch
        x = rng.integers(-128, 128, (batch, shape[1]), dtype=np.int8)
        patch = None
        if label == "head":
            patch = head_patch(weights, args.versions_db, args.version_name, args.deterministic_row)
            row, payload, _ = patch
            delta = np.frombuffer(payload, dtype=np.int8).astype(np.int32) - weights[row].astype(np.int32)
            if not np.any(delta):
                raise ValueError("The stored row has no edited bytes.")
            if not (x[0].astype(np.int32) @ delta):
                x[0].fill(0)
                x[0, np.flatnonzero(delta)[0]] = 1
            report["hypothetical_encoder_head_memory"] = budget(*shape, ENCODER_INPUT_LIMIT, memory)
            if args.segmented_head:
                report["cases"].append(check_segmented_head(
                    "head_original", tensor, weights, x, segmented, bool(args.bitstream),
                ))
                original_score = int(x[0].astype(np.int32) @ weights[row].astype(np.int32))
                patched_weights = np.memmap(
                    args.model_dir / "weights.bin", offset=manifest["tensors"][tensor]["offset"],
                    mode="c", dtype=np.int8, shape=shape,
                )
                patched_weights[row] = np.frombuffer(payload, dtype=np.int8)
                patched_score = int(x[0].astype(np.int32) @ patched_weights[row].astype(np.int32))
                report["cases"].append(check_segmented_head(
                    "head_patched", tensor, patched_weights, x, segmented, bool(args.bitstream),
                    {**patch[2], "original_score": original_score, "score": patched_score},
                ))
            else:
                report["cases"].append(check_case(
                    "head_original", tensor, weights, x, run, bool(args.bitstream),
                ))
                report["cases"].append(check_case(
                    "head_patched", tensor, weights, x, run, bool(args.bitstream), patch,
                ))
        else:
            report["encoder_memory_at_input_limit"][label] = budget(*shape, ENCODER_INPUT_LIMIT, memory)
            report["cases"].append(check_case(label, tensor, weights, x, run, bool(args.bitstream)))
    return report


def require_inactive_service():
    mode = subprocess.run(
        ["systemctl", "show", "pinkprocess-kit.service", "--property=ActiveState", "--value"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    if mode != "inactive":
        raise RuntimeError(f"Maintenance requires inactive pinkprocess-kit.service; found {mode!r}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--simulate", action="store_true")
    backend.add_argument("--bitstream", type=Path)
    patch = parser.add_mutually_exclusive_group(required=True)
    patch.add_argument("--versions-db", type=Path)
    patch.add_argument("--deterministic-row", action="store_true")
    parser.add_argument("--version-name", default="Astra Now 0f137b5e")
    parser.add_argument("--encoder-batch", type=int, default=9)
    parser.add_argument("--segmented-head", action="store_true",
                        help="Run both full-head checks through the production all-scope segmentation path.")
    parser.add_argument("--timeout", type=float, default=12)
    args = parser.parse_args()
    if not 2 <= args.encoder_batch <= ENCODER_INPUT_LIMIT:
        parser.error(f"--encoder-batch must be between 2 and {ENCODER_INPUT_LIMIT}")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.simulate:
        def run(a, w, decode, _samples):
            return offline(a, w, decode)

        def segmented(name, weights, x, _samples):
            uploads = []

            class SimulatedProjection(OverlayProjection):
                def _project_segment(self, w, a):
                    packed_a, packed_w = pack_projection(w, a)
                    scores, timing = offline(packed_a, packed_w, len(a) == 1)
                    uploads.append({"rows": len(w), **timing})
                    return unpack_projection(scores, len(a), len(w))

            start = perf_counter_ns()
            actual = SimulatedProjection(None, None, scope="all").project(name, weights, x)
            return actual, perf_counter_ns() - start, uploads

        report = all_cases(args, run, segmented)
    else:
        require_inactive_service()
        from pynq import Overlay, allocate

        with board_lock():
            require_inactive_service()
            previous = signal.signal(signal.SIGTERM, interrupt)
            try:
                overlay = Overlay(str(args.bitstream))

                def observing_allocate(samples):
                    def observed_allocate(**kwargs):
                        buffer = allocate(**kwargs)
                        try:
                            name = f"after_dma_allocation_{len(samples) - 2}"
                            samples[name] = memory_kib(required=True)
                        except BaseException:
                            buffer.freebuffer()
                            raise
                        return buffer

                    return observed_allocate

                def run(a, w, decode, samples):
                    return run_projection(overlay, observing_allocate(samples), a, w, decode, args.timeout)

                def segmented(name, weights, x, samples):
                    uploads = []

                    class ObservedProjection(OverlayProjection):
                        def _project_segment(self, w, a):
                            start = perf_counter_ns()
                            result = super()._project_segment(w, a)
                            uploads.append({"rows": len(w), "wall_ns": perf_counter_ns() - start})
                            return result

                        def _retain(self, buffers):
                            reset_after_failure(overlay)
                            buffers.close()

                    projector = ObservedProjection(
                        projection_mmio(overlay), observing_allocate(samples),
                        scope="all", timeout=args.timeout,
                    )
                    start = perf_counter_ns()
                    actual = projector.project(name, weights, x)
                    return actual, perf_counter_ns() - start, uploads

                report = all_cases(args, run, segmented)
            finally:
                signal.signal(signal.SIGTERM, previous)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
