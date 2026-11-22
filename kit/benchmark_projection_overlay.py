"""Benchmark the prototype INT8 projection overlay using contiguous DDR buffers."""

import argparse
import json
import math
from pathlib import Path
import signal
from statistics import median
import sys
from time import perf_counter_ns, sleep

import numpy as np

from fpga.check_classifier import board_lock
from kit.benchmark_projection import load_capture
from lm_reference.projection_overlay import BUSY, CONTROL, pack_projection, projection_mmio, unpack_projection
from lm_reference.projection_overlay import PROJECTION_QUEUE_ID, QUEUE_ID
from lm_reference.projection_overlay import run_projection as execute_projection

def simulate_projection(a, w, decode):
    """Interpret just the DDR layout, independent of the packing and unpacking code."""
    x = a[:, :, :1] if decode else a
    scores = np.matmul(x[:, None].swapaxes(-1, -2).astype(np.int32), w[None].astype(np.int32))
    return scores.reshape(len(a), len(w), 64)


def reset_after_failure(overlay):
    """Keep the live buffers and process lock until the programmable logic is reloaded."""
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        while True:
            try:
                # Overlay.reset() only resets PYNQ's metadata. Download the bitstream to stop hardware.
                overlay.download()
                if projection_mmio(overlay).read(CONTROL) & BUSY:
                    raise RuntimeError("Projection core was still busy after reprogramming.")
                return
            except Exception as exc:
                print(f"Could not verify an FPGA reset ({exc}). Holding DMA buffers and retrying.",
                      file=sys.stderr, flush=True)
                sleep(1)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, blocked)


def run_projection(overlay, allocate, a, w, decode, timeout):
    def recover(buffers):
        reset_after_failure(overlay)
        buffers.close()

    return execute_projection(projection_mmio(overlay), allocate, a, w, decode, timeout, recover)


def benchmark(weights, activations, run, repeats):
    rows, depth = weights.shape
    batch = len(activations)
    decode = batch == 1
    # The largest supported depth (65535) keeps all INT8 sums within signed int32.
    if depth > 65535:
        raise ValueError("Projection depth must fit in the 16-bit device register.")
    expected = activations.astype(np.int32) @ weights.astype(np.int32).T
    samples = []
    for _ in range(repeats):
        start = perf_counter_ns()
        a, w = pack_projection(weights, activations)
        packed_ns = perf_counter_ns() - start
        scores, phases = run(a, w, decode)
        start_unpack = perf_counter_ns()
        actual = unpack_projection(scores, batch, rows)
        end = perf_counter_ns()
        if not np.array_equal(actual, expected):
            raise RuntimeError("Projection output disagrees with the signed int32 reference.")
        samples.append({"projection_ns": end - start, "pack_ns": packed_ns, **phases,
                        "unpack_ns": end - start_unpack})
    return {
        "shape": {"rows": rows, "columns": depth, "batch": batch},
        "mode": "decode" if decode else "encode",
        "position_tiles": len(a),
        "output_tiles": len(w),
        "buffer_bytes": {"activations": a.nbytes, "weights": w.nbytes, "scores": scores.nbytes},
        "integer_reference_matched": True,
        "median": {key: median(sample[key] for sample in samples) for key in samples[0]},
        "samples": samples,
    }


def offline(a, w, decode):
    start = perf_counter_ns()
    scores = simulate_projection(a, w, decode)
    return scores, {"software_tile_model_ns": perf_counter_ns() - start}


def interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True,
                        help="NPZ containing signed INT8 weights [rows, columns] and activations [batch, columns].")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10, help="Seconds to wait for a physical projection.")
    parser.add_argument("--hardware-counters", action="store_true",
                        help="Read MAC and DDR counters on the QUE8 overlay after each measured launch.")
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--bitstream", type=Path, help="Prototype .bit with its matching .hwh.")
    backend.add_argument("--simulate", action="store_true", help="Verify the DDR layout offline.")
    args = parser.parse_args()
    if args.hardware_counters and args.simulate:
        parser.error("--hardware-counters requires the physical QUE8 overlay")
    if args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--repeats and --timeout must be finite and positive")
    weights, activations = load_capture(args.capture)
    if args.simulate:
        report = benchmark(weights, activations, offline, args.repeats)
    else:
        from pynq import Overlay, allocate

        with board_lock():
            previous = signal.signal(signal.SIGTERM, interrupt)
            try:
                overlay = Overlay(str(args.bitstream))

                mmio = projection_mmio(overlay)
                if args.hardware_counters and mmio.read(QUEUE_ID) != PROJECTION_QUEUE_ID:
                    raise RuntimeError("Hardware counters require the QUE8 overlay.")

                def run(a, w, decode):
                    scores, timings = run_projection(overlay, allocate, a, w, decode, args.timeout)
                    if args.hardware_counters:
                        timings.update(elapsed_cycles=mmio.read(0x2c), mac_feed_cycles=mmio.read(0x30),
                                       ddr_read_beats=mmio.read(0x34), ddr_write_beats=mmio.read(0x38))
                    return scores, timings

                report = benchmark(weights, activations, run, args.repeats)
            finally:
                signal.signal(signal.SIGTERM, previous)
    report["backend"] = "software_tile_model" if args.simulate else "pynq_projection"
    report["capture"] = str(args.capture)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
