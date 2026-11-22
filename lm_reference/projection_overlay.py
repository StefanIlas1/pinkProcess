"""Pack INT8 projections and execute them on the DDR-backed projection core."""

from contextlib import ExitStack
from threading import Lock, Thread
from time import monotonic, perf_counter_ns, sleep

import numpy as np

CONTROL = 0x00
MODE = 0x04
DEPTH = 0x08
POSITION_TILES = 0x0C
OUTPUT_TILES = 0x10
A_ADDRESS = 0x14
W_ADDRESS = 0x18
Y_ADDRESS = 0x1C
CORE_ID = 0x20
PROJECTION_ID = 0x50504A31
BUSY, DONE, ERROR = 1, 2, 4
QUERY_PROJECTIONS = frozenset({
    "encoder.block.0.layer.0.SelfAttention.q",
    "decoder.block.0.layer.0.SelfAttention.q",
})


def pack_projection(weights, activations):
    batch, depth = activations.shape
    position_lanes, output_lanes = (1, 64) if batch == 1 else (8, 8)
    position_tiles = (batch + position_lanes - 1) // position_lanes
    output_tiles = (len(weights) + output_lanes - 1) // output_lanes
    if max(depth, position_tiles, output_tiles) > 65535:
        raise ValueError("Projection depth and tile counts must fit in the 16-bit device registers.")

    a = np.zeros((position_tiles, depth, 8), dtype=np.int8)
    if batch == 1:
        a[0, :, 0] = activations[0]
    else:
        padded = np.zeros((position_tiles * 8, depth), dtype=np.int8)
        padded[:batch] = activations
        a[:] = padded.reshape(position_tiles, 8, depth).transpose(0, 2, 1)

    padded = np.zeros((output_tiles * output_lanes, depth), dtype=np.int8)
    padded[:len(weights)] = weights
    w = np.ascontiguousarray(padded.reshape(output_tiles, output_lanes, depth).transpose(0, 2, 1))
    return a, w


def unpack_projection(scores, batch, rows):
    position_tiles, output_tiles, _ = scores.shape
    position_lanes, output_lanes = (1, 64) if batch == 1 else (8, 8)
    tiles = scores.reshape(position_tiles, output_tiles, position_lanes, output_lanes)
    grid = tiles.transpose(0, 2, 1, 3).reshape(position_tiles * position_lanes, output_tiles * output_lanes)
    return grid[:batch, :rows]


def projection_mmio(overlay, *, required=True):
    projection = getattr(overlay, "projection", None)
    if projection is not None and projection.mmio.read(CORE_ID) == PROJECTION_ID:
        return projection.mmio
    if required:
        raise RuntimeError("The programmed overlay does not expose the expected projection core.")


def _complete(status):
    return not status & BUSY and bool(status & (DONE | ERROR))


def run_projection(mmio, allocate, a, w, decode, timeout, retain):
    if mmio.read(CONTROL) & BUSY:
        raise RuntimeError("Projection accelerator is already busy.")

    with ExitStack() as buffers:
        start = perf_counter_ns()
        a_buffer = buffers.enter_context(allocate(shape=a.shape, dtype=np.int8))
        w_buffer = buffers.enter_context(allocate(shape=w.shape, dtype=np.int8))
        y_buffer = buffers.enter_context(allocate(shape=(len(a), len(w), 64), dtype=np.dtype("<i4")))
        allocated_ns = perf_counter_ns() - start
        addresses = [int(buffer.device_address) for buffer in (a_buffer, w_buffer, y_buffer)]
        for address, buffer in zip(addresses, (a_buffer, w_buffer, y_buffer)):
            if address < 0 or address % 64 or address + buffer.nbytes > (1 << 32):
                raise RuntimeError("Expected 64-byte-aligned DDR buffers within the device's 32-bit address space.")

        start = perf_counter_ns()
        a_buffer[:] = a
        w_buffer[:] = w
        a_buffer.flush()
        w_buffer.flush()
        # Write back any dirty CPU cache lines before the fabric writes through the noncoherent HP port.
        y_buffer.flush()
        upload_ns = perf_counter_ns() - start

        start = perf_counter_ns()
        mmio.write(MODE, int(decode))
        mmio.write(DEPTH, a.shape[1])
        mmio.write(POSITION_TILES, len(a))
        mmio.write(OUTPUT_TILES, len(w))
        for register, address in zip((A_ADDRESS, W_ADDRESS, Y_ADDRESS), addresses):
            mmio.write(register, address)
        try:
            mmio.write(CONTROL, 1)
            deadline = monotonic() + timeout
            while True:
                status = mmio.read(CONTROL)
                if _complete(status):
                    break
                if monotonic() >= deadline:
                    raise TimeoutError("Projection accelerator did not complete before the deadline.")
                sleep(0.0001)
        except BaseException:
            retain(buffers.pop_all())
            raise
        device_ns = perf_counter_ns() - start
        if status & ERROR:
            raise RuntimeError(f"Projection accelerator reported status {status:#x}.")

        start = perf_counter_ns()
        y_buffer.invalidate()
        scores = np.array(y_buffer, copy=True)
        download_ns = perf_counter_ns() - start
    return scores, {"allocate_ns": allocated_ns, "upload_ns": upload_ns,
                    "device_roundtrip_ns": device_ns, "download_ns": download_ns}


class QueryProjection:
    """Offload the first self-attention query in each FLAN stack."""

    def __init__(self, mmio, allocate, *, timeout=2):
        self.mmio = mmio
        self.allocate = allocate
        self.timeout = timeout
        self.lock = Lock()
        self.disabled = False
        self.pending = None

    def project(self, name, weights, activations):
        if name not in QUERY_PROJECTIONS:
            return None
        with self.lock:
            if self.disabled:
                return None
            batch = len(activations)
            a, w = pack_projection(weights, activations)
            scores, _ = run_projection(self.mmio, self.allocate, a, w, batch == 1,
                                       self.timeout, self._retain)
            return unpack_projection(scores, batch, len(weights))

    def _retain(self, buffers):
        # Called with the lock held when launch or completion cannot be confirmed.
        self.disabled = True
        self.pending = buffers
        Thread(target=self._wait_for_idle, daemon=True).start()

    def _wait_for_idle(self):
        while True:
            with self.lock:
                if self.pending is None:
                    return
                try:
                    idle = _complete(self.mmio.read(CONTROL))
                except (OSError, RuntimeError):
                    idle = False
                if idle:
                    self.pending.close()
                    self.pending = None
                    return
            sleep(0.05)

    def close(self):
        # The service calls this before relinquishing board ownership, even if a
        # daemon inference worker was active when shutdown started.
        with self.lock:
            self.disabled = True
        while True:
            try:
                return self._wait_for_idle()
            except KeyboardInterrupt:
                continue
