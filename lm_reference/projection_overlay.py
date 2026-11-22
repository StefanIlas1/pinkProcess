"""Pack INT8 projections and execute them on the DDR-backed projection core."""

from collections import Counter, OrderedDict
from contextlib import ExitStack
from math import prod
from mmap import PAGESIZE
from threading import Lock, Thread
from time import monotonic, perf_counter_ns, sleep

import numpy as np

from .projection_argmax import argmax_result, configure_argmax, launch_argmax
from .projection_native import NativeProjection

CONTROL = 0x00
MODE = 0x04
DEPTH = 0x08
POSITION_TILES = 0x0C
OUTPUT_TILES = 0x10
A_ADDRESS = 0x14
W_ADDRESS = 0x18
Y_ADDRESS = 0x1C
CORE_ID = 0x20
QUEUE_CONTROL = 0x24
QUEUE_ID = 0x28
PROJECTION_QUEUE_ID = 0x51554538
QUEUE_CLEAR, QUEUE_PUSH, QUEUE_RUN = 0, 1, 2
PROJECTION_ID = 0x50504A31
BUSY, DONE, ERROR = 1, 2, 4
MAX_SEGMENT_BUFFER_BYTES = 2 * 1024 * 1024
MAX_PACKED_WEIGHT_BYTES = 64 * 1024 * 1024
MAX_RESIDENT_HEAD_MIB = 16
QUERY_PROJECTIONS = frozenset({
    "encoder.block.0.layer.0.SelfAttention.q",
    "decoder.block.0.layer.0.SelfAttention.q",
})


def _pack_activations(a, activations):
    batch, depth = activations.shape
    a.fill(0)
    if batch == 1:
        a[0, :, 0] = activations[0]
    else:
        full, tail = divmod(batch, 8)
        a[:full] = activations[:full * 8].reshape(full, 8, depth).transpose(0, 2, 1)
        if tail:
            a[full, :, :tail] = activations[full * 8:].T


def _pack_weights(w, weights):
    _, depth, lanes = w.shape
    full, tail = divmod(len(weights), lanes)
    w[:full] = weights[:full * lanes].reshape(full, lanes, depth).transpose(0, 2, 1)
    if tail:
        w[full].fill(0)
        w[full, :, :tail] = weights[full * lanes:].T


def pack_projection(weights, activations):
    batch, depth = activations.shape
    output_lanes = 64 if batch == 1 else 8
    position_tiles = (batch + 7) // 8
    output_tiles = (len(weights) + output_lanes - 1) // output_lanes
    if max(depth, position_tiles, output_tiles) > 65535:
        raise ValueError("Projection depth and tile counts must fit in the 16-bit device registers.")
    a = np.empty((position_tiles, depth, 8), dtype=np.int8)
    w = np.empty((output_tiles, depth, output_lanes), dtype=np.int8)
    _pack_activations(a, activations)
    _pack_weights(w, weights)
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


def _buffer_address(buffer):
    address = int(buffer.device_address)
    if address < 0 or address % 64 or address + buffer.nbytes > (1 << 32):
        raise RuntimeError("Expected 64-byte-aligned DDR buffers within the device's 32-bit address space.")
    return address


def _configure(mmio, addresses, decode, depth, position_tiles, output_tiles):
    for register, value in zip((MODE, DEPTH, POSITION_TILES, OUTPUT_TILES, A_ADDRESS, W_ADDRESS, Y_ADDRESS),
                               (int(decode), depth, position_tiles, output_tiles, *addresses)):
        mmio.write(register, value)


def _launch(mmio, timeout, register=CONTROL, command=1):
    mmio.write(register, command)
    deadline = monotonic() + timeout
    while True:
        status = mmio.read(CONTROL)
        if _complete(status):
            return status
        if monotonic() >= deadline:
            raise TimeoutError("Projection accelerator did not complete before the deadline.")
        sleep(0.0001)


def run_projection(mmio, allocate, a, w, decode, timeout, retain):
    if mmio.read(CONTROL) & BUSY:
        raise RuntimeError("Projection accelerator is already busy.")

    with ExitStack() as buffers:
        start = perf_counter_ns()
        a_buffer = buffers.enter_context(allocate(shape=a.shape, dtype=np.int8))
        w_buffer = buffers.enter_context(allocate(shape=w.shape, dtype=np.int8))
        y_buffer = buffers.enter_context(allocate(shape=(len(a), len(w), 64), dtype=np.dtype("<i4")))
        allocated_ns = perf_counter_ns() - start
        addresses = [_buffer_address(buffer) for buffer in (a_buffer, w_buffer, y_buffer)]

        start = perf_counter_ns()
        a_buffer[:] = a
        w_buffer[:] = w
        a_buffer.flush()
        w_buffer.flush()
        # Write back any dirty CPU cache lines before the fabric writes through the noncoherent HP port.
        y_buffer.flush()
        upload_ns = perf_counter_ns() - start

        start = perf_counter_ns()
        _configure(mmio, addresses, decode, a.shape[1], len(a), len(w))
        try:
            status = _launch(mmio, timeout)
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


class OverlayProjection:
    """Offload either the first self-attention queries or every learned projection."""

    def __init__(self, mmio, allocate, *, scope="query", timeout=2, queue=False,
                 argmax_mmio=None, resident_head_mib=0):
        if scope not in ("query", "all"):
            raise ValueError("Projection scope must be query or all.")
        self.mmio = mmio
        self.allocate = allocate
        self.scope = scope
        self.timeout = timeout
        if queue and mmio.read(QUEUE_ID) != PROJECTION_QUEUE_ID:
            raise RuntimeError("The programmed overlay does not expose the expected projection queue.")
        self.queue = queue
        self.lock = Lock()
        self.disabled = False
        self.pending = False
        self.pending_mmio = None
        if argmax_mmio is not None and scope != "all":
            raise ValueError("Vocabulary reduction requires all-projection scope.")
        if (type(resident_head_mib) is not int or not 0 <= resident_head_mib <= MAX_RESIDENT_HEAD_MIB
                or (resident_head_mib and scope != "all")):
            raise ValueError("Resident head cache requires all-projection scope and a budget of 0–16 MiB.")
        self.argmax_mmio = argmax_mmio
        self.resident_budget = resident_head_mib * 1024 * 1024
        self.resident_head = {}
        self.resident_bytes = 0
        self.resident_blocked = False
        self.native = NativeProjection.load(mmio)
        self.workspace = {}
        self.model_weights = {}
        self.packed_weights = OrderedDict()
        self.packed_bytes = 0
        self.phases = Counter()

    def bind_model(self, tensors):
        """Cache only weight arrays whose writer participates in invalidation."""
        with self.lock:
            self.model_weights = tensors
            self.packed_weights.clear()
            self.packed_bytes = 0
            if not self.pending:
                self._release_resident()
                self.resident_blocked = False

    def invalidate_weights(self, name):
        with self.lock:
            for key in list(self.packed_weights):
                names, _ = key
                if names == name or (isinstance(names, tuple) and name in names):
                    self.packed_bytes -= self.packed_weights.pop(key).nbytes
            if name == "lm_head" and not self.pending:
                self._release_resident()
                self.resident_blocked = False

    def start_generation(self):
        with self.lock:
            self.phases.clear()

    def profile(self):
        with self.lock:
            if not self.phases:
                return {}
            return {"native_commands": 0, "allocations": 0, "allocate_ns": 0,
                    "weight_cache_hits": 0, "weight_cache_misses": 0, "pack_weights_ns": 0,
                    **self.phases, "packed_weight_bytes": self.packed_bytes,
                    "workspace_bytes": sum(buffer.nbytes for buffer, _ in self.workspace.values()),
                    "resident_weight_bytes": self.resident_bytes}

    def head_argmax(self, weights, activations, activation_scale, row_scales):
        if self.argmax_mmio is None:
            raise RuntimeError("The vocabulary reducer was not enabled.")
        if (len(activations) != 1 or row_scales.shape != (len(weights),)
                or row_scales.dtype != np.float32):
            raise ValueError("Vocabulary reduction requires one decoder position and float32 row scales.")
        return self.project("lm_head", weights, activations, head_scales=(activation_scale, row_scales))

    def project(self, name, weights, activations, *, head_scales=None):
        if self.scope == "query" and name not in QUERY_PROJECTIONS:
            return None
        with self.lock:
            if self.disabled:
                if self.scope == "all":
                    raise RuntimeError("The all-projection accelerator is unavailable.")
                return None
            batch, depth = activations.shape
            decode = batch == 1
            output_lanes = 64 if decode else 8
            position_tiles = (batch + 7) // 8
            output_tiles = (len(weights) + output_lanes - 1) // output_lanes
            if not 0 < depth <= 65535 or not 0 < position_tiles <= 65535 or not output_tiles:
                raise ValueError("Projection depth and tile counts must fit in the 16-bit device registers.")
            if self.mmio.read(CONTROL) & BUSY:
                raise RuntimeError("Projection accelerator is already busy.")
            # Keep each contiguous weight and score buffer bounded, including the padded lanes.
            tiles = min(MAX_SEGMENT_BUFFER_BYTES // (depth * output_lanes),
                        MAX_SEGMENT_BUFFER_BYTES // (position_tiles * 64 * 4), 65535)
            if tiles == 0:
                raise ValueError("One projection tile exceeds the CMA segment budget.")

            a = self._upload_activations(activations, position_tiles, depth)
            packed = self._cached_weights(name, weights, decode, (output_tiles, depth, output_lanes))
            resident = (self.resident_budget and decode and name == "lm_head"
                        and self.model_weights.get("lm_head.weight") is weights)
            rows_per_segment = tiles * output_lanes
            result = np.empty((batch, len(weights)), dtype=np.int32) if head_scales is None else None
            best = None
            for start in range(0, len(weights), rows_per_segment):
                stop = min(start + rows_per_segment, len(weights))
                first_tile = start // output_lanes
                last_tile = (stop + output_lanes - 1) // output_lanes
                shape = (last_tile - first_tile, depth, output_lanes)
                if packed is None:
                    segment = weights[start:stop]
                else:
                    segment = packed[first_tile:last_tile]
                destination = None if result is None else result[:, start:stop]
                y = self._project_segment(a, segment, packed is not None, shape, destination, decode,
                                          resident_key=start if resident else None)
                if head_scales is not None:
                    activation_scale, row_scales = head_scales
                    index, score = self._reduce_segment(y, activation_scale, row_scales[start:stop])
                    if best is None or score > best[1]:
                        best = start + index, score
            if head_scales is not None:
                return best[0]
            return result

    def project_many(self, matrices, activations):
        """Project independent weights with one shared activation upload."""
        if self.scope != "all" or not 2 <= len(matrices) <= 8:
            return [self.project(name, weights, activations) for name, weights in matrices]

        batch, depth = activations.shape
        decode = batch == 1
        positions = (batch + 7) // 8
        lanes = 64 if decode else 8
        tiles = [(len(weights) + lanes - 1) // lanes for _, weights in matrices]
        weight_sizes = [count * depth * lanes for count in tiles]
        score_sizes = [positions * count * 64 for count in tiles]
        if (not 0 < depth <= 65535 or not 0 < positions <= 65535
                or any(not 0 < count <= 65535 for count in tiles)
                or max(weight_sizes) > MAX_SEGMENT_BUFFER_BYTES
                or max(score_sizes) * 4 > MAX_SEGMENT_BUFFER_BYTES):
            return [self.project(name, weights, activations) for name, weights in matrices]

        # Every configuration address remains 64-byte aligned even with odd K.
        weight_spans = [(size + 63) // 64 * 64 for size in weight_sizes]
        queued = (self.queue and sum(weight_spans) <= MAX_SEGMENT_BUFFER_BYTES
                  and sum(score_sizes) * 4 <= MAX_SEGMENT_BUFFER_BYTES)
        with self.lock:
            if self.disabled:
                raise RuntimeError("The all-projection accelerator is unavailable.")
            if self.mmio.read(CONTROL) & BUSY:
                raise RuntimeError("Projection accelerator is already busy.")
            a = self._upload_activations(activations, positions, depth)
            self.phases["shared_activation_uploads_saved"] += len(matrices) - 1
            if (queued and sum(tiles) <= 65535 and sum(weight_sizes) <= MAX_PACKED_WEIGHT_BYTES
                    and all(len(w) % lanes == 0 for _, w in matrices)
                    and all(self.model_weights.get(name + ".weight") is w for name, w in matrices)):
                return self._project_fused(matrices, tiles, batch, depth, a)
            if queued:
                w_space = self._buffer("w", (sum(weight_spans),), np.int8)
                y_space = self._buffer("y", (sum(score_sizes),), np.dtype("<i4"))
                self.mmio.write(QUEUE_CONTROL, QUEUE_CLEAR)
            commands = []
            weight_offset = score_offset = 0
            for (name, weights), count, weight_size, weight_span, score_size in zip(
                matrices, tiles, weight_sizes, weight_spans, score_sizes
            ):
                shape = (count, depth, lanes)
                packed = self._cached_weights(name, weights, decode, shape)
                result = np.empty((batch, len(weights)), dtype=np.int32)
                segment = weights if packed is None else packed
                if not queued:
                    self._project_segment(a, segment, packed is not None, shape, result, decode)
                    commands.append((None, result))
                    continue
                w = w_space[weight_offset:weight_offset + weight_size].reshape(shape)
                y = y_space[score_offset:score_offset + score_size].reshape(positions, count, 64)
                self._upload_weights(w, segment, packed is not None, decode)
                addresses = [_buffer_address(buffer) for buffer in (a, w, y)]
                _configure(self.mmio, addresses, decode, depth, positions, count)
                self.mmio.write(QUEUE_CONTROL, QUEUE_PUSH)
                commands.append((y, result))
                weight_offset += weight_span
                score_offset += score_size

            if queued:
                if self.mmio.read(QUEUE_CONTROL) & 0xf != len(commands):
                    raise RuntimeError("Projection queue did not accept all staged descriptors.")
                start = perf_counter_ns()
                try:
                    status = _launch(self.mmio, self.timeout * len(commands), QUEUE_CONTROL, QUEUE_RUN)
                except BaseException:
                    self._retain()
                    raise
                self.phases["device_roundtrip_ns"] += perf_counter_ns() - start
                self.phases["queue_runs"] += 1
                self.phases["commands"] += len(commands)
                if status & ERROR:
                    raise RuntimeError(f"Projection queue reported status {status:#x}.")
                for y, result in commands:
                    self._consume_scores(y, result)
            return [result for _, result in commands]

    def _project_fused(self, matrices, tiles, batch, depth, a):
        decode = batch == 1
        names = tuple(name for name, _ in matrices)
        key = names, decode
        shape = (sum(tiles), depth, 64 if decode else 8)
        packed = self.packed_weights.get(key)
        if packed is None:
            self.phases["weight_cache_misses"] += 1
            packed = np.empty(shape, dtype=np.int8)
            offset = 0
            for (_, weights), count in zip(matrices, tiles):
                self._pack_weight_data(packed[offset:offset + count], weights, decode)
                offset += count
            self._remember_weights(key, packed)
        else:
            self.packed_weights.move_to_end(key)
            self.phases["weight_cache_hits"] += 1
        result = np.empty((batch, sum(len(w) for _, w in matrices)), dtype=np.int32)
        self._project_segment(a, packed, True, shape, result, decode, timeout=self.timeout * len(matrices))
        phase = "fused_decoder" if decode else "fused_batch"
        self.phases[phase + "_groups"] += 1
        self.phases[phase + "_commands_saved"] += len(matrices) - 1
        offset = 0
        outputs = []
        for _, w in matrices:
            outputs.append(result[:, offset:offset + len(w)])
            offset += len(w)
        return outputs

    def _upload_activations(self, activations, positions, depth):
        a = self._buffer("a", (positions, depth, 8), np.int8)
        start = perf_counter_ns()
        pack_a = self.native.pack_activations if self.native is not None else _pack_activations
        pack_a(a, activations)
        self.phases["pack_activations_ns"] += perf_counter_ns() - start
        start = perf_counter_ns()
        a.flush()
        self.phases["upload_activations_ns"] += perf_counter_ns() - start
        self.phases["activation_bytes"] += a.nbytes
        return a

    def _buffer(self, name, shape, dtype):
        elements = prod(shape)
        entry = self.workspace.get(name)
        if entry is None or entry[0].size < elements:
            start = perf_counter_ns()
            if entry is not None:
                del self.workspace[name]
                entry[1].close()
            resources = ExitStack()
            with resources:
                capacity = (1 << (elements - 1).bit_length(),)
                try:
                    buffer = resources.enter_context(self.allocate(shape=capacity, dtype=dtype))
                except (MemoryError, OSError, RuntimeError):
                    if not self.resident_head:
                        raise
                    # Ordinary projections take priority over optional residency.
                    self._release_resident()
                    self.resident_blocked = True
                    self.phases["resident_workspace_fallbacks"] += 1
                    buffer = resources.enter_context(self.allocate(shape=capacity, dtype=dtype))
                _buffer_address(buffer)
                if name == "y":
                    # The noncoherent port owns this region after its first launch.
                    buffer.flush()
                self.workspace[name] = buffer, resources.pop_all()
            self.phases["allocations"] += 1
            self.phases["allocate_ns"] += perf_counter_ns() - start
        return self.workspace[name][0][:elements].reshape(shape)

    def _pack_weight_data(self, destination, source, decode):
        start = perf_counter_ns()
        if self.native is None:
            _pack_weights(destination, source)
        else:
            self.native.pack_weights(destination, source, decode)
        self.phases["pack_weights_ns"] += perf_counter_ns() - start

    def _cached_weights(self, name, weights, decode, shape):
        if self.model_weights.get(name + ".weight") is not weights:
            return None
        key = name, decode
        packed = self.packed_weights.get(key)
        if packed is not None:
            self.packed_weights.move_to_end(key)
            self.phases["weight_cache_hits"] += 1
            return packed
        length = prod(shape)
        self.phases["weight_cache_misses"] += 1
        if length > MAX_PACKED_WEIGHT_BYTES:
            return None
        packed = np.empty(shape, dtype=np.int8)
        self._pack_weight_data(packed, weights, decode)
        self._remember_weights(key, packed)
        return packed

    def _remember_weights(self, key, packed):
        while self.packed_bytes + packed.nbytes > MAX_PACKED_WEIGHT_BYTES:
            _, old = self.packed_weights.popitem(last=False)
            self.packed_bytes -= old.nbytes
        self.packed_weights[key] = packed
        self.packed_bytes += packed.nbytes

    def _resident_weight(self, key, segment, packed, shape):
        key = key, shape
        entry = self.resident_head.get(key)
        if entry is not None:
            self.phases["resident_weight_hits"] += 1
            return entry[0]
        capacity = (prod(shape) + PAGESIZE - 1) // PAGESIZE * PAGESIZE
        if self.resident_blocked or self.resident_bytes + capacity > self.resident_budget:
            return None
        start = perf_counter_ns()
        with ExitStack() as resources:
            try:
                buffer = resources.enter_context(self.allocate(shape=shape, dtype=np.int8))
            except (MemoryError, OSError, RuntimeError):
                self.resident_blocked = True
                self.phases["resident_allocation_failures"] += 1
                return None
            _buffer_address(buffer)
            self.phases["allocations"] += 1
            self.phases["allocate_ns"] += perf_counter_ns() - start
            self._upload_weights(buffer, segment, packed, True)
            self.resident_head[key] = buffer, resources.pop_all()
        self.resident_bytes += capacity
        self.phases["resident_weight_uploads"] += 1
        return buffer

    def _project_segment(self, a, segment, packed, shape, destination, decode, *, resident_key=None, timeout=None):
        # Reserve the fallback buffers before a resident can compete for CMA.
        scratch = self._buffer("w", shape, np.int8)
        y = self._buffer("y", (len(a), shape[0], 64), np.dtype("<i4"))
        w = self._resident_weight(resident_key, segment, packed, shape) if resident_key is not None else None
        if w is None:
            w = scratch
            self._upload_weights(w, segment, packed, decode)

        if timeout is None:
            timeout = self.timeout
        addresses = [int(buffer.device_address) for buffer in (a, w, y)]
        start = perf_counter_ns()
        if self.native is None:
            _configure(self.mmio, addresses, decode, a.shape[1], len(a), len(w))
        try:
            if self.native is None:
                status = _launch(self.mmio, timeout)
            else:
                status = self.native.dispatch(addresses, decode, a.shape[1], len(a), len(w), timeout)
                self.phases["native_commands"] += 1
        except BaseException:
            self._retain()
            raise
        self.phases["commands"] += 1
        self.phases["device_roundtrip_ns"] += perf_counter_ns() - start
        if status & ERROR:
            raise RuntimeError(f"Projection accelerator reported status {status:#x}.")

        if destination is not None:
            self._consume_scores(y, destination)
        return y

    def _reduce_segment(self, y, activation_scale, row_scales):
        rows = len(row_scales)
        # DMA reads two scales at a time. The final row may leave one padding word.
        scales = self._buffer("scales", (rows + rows % 2,), np.float32)
        scales[:rows] = row_scales
        if rows % 2:
            scales[rows] = 0
        scales.flush()
        start = perf_counter_ns()
        configure_argmax(self.argmax_mmio, y, scales, rows, activation_scale)
        try:
            status = launch_argmax(self.argmax_mmio, self.timeout)
        except BaseException:
            self._retain(self.argmax_mmio)
            raise
        self.phases["argmax_commands"] += 1
        self.phases["argmax_device_ns"] += perf_counter_ns() - start
        return argmax_result(self.argmax_mmio, status, rows)

    def _upload_weights(self, w, segment, packed, decode):
        if not packed:
            self._pack_weight_data(w, segment, decode)
        start = perf_counter_ns()
        if packed:
            w[:] = segment
        w.flush()
        self.phases["upload_weights_ns"] += perf_counter_ns() - start
        self.phases["weight_bytes"] += w.nbytes

    def _consume_scores(self, y, destination):
        start = perf_counter_ns()
        y.invalidate()
        self.phases["download_ns"] += perf_counter_ns() - start
        self.phases["score_bytes"] += y.nbytes
        start = perf_counter_ns()
        batch, rows = destination.shape
        if self.native is None:
            destination[:] = unpack_projection(np.asarray(y), batch, rows)
        else:
            self.native.unpack_scores(destination, y)
        self.phases["unpack_ns"] += perf_counter_ns() - start

    def _release_workspace(self):
        for _, resources in self.workspace.values():
            resources.close()
        self.workspace.clear()
        self._release_resident()

    def _release_resident(self):
        for _, resources in self.resident_head.values():
            resources.close()
        self.resident_head.clear()
        self.resident_bytes = 0

    def _retain(self, mmio=None):
        # Called with the lock held when launch or completion cannot be confirmed.
        self.disabled = True
        self.pending = True
        self.pending_mmio = self.mmio if mmio is None else mmio
        Thread(target=self._wait_for_idle, daemon=True).start()

    def _wait_for_idle(self):
        while True:
            with self.lock:
                if not self.pending:
                    return
                try:
                    idle = _complete(self.pending_mmio.read(CONTROL))
                except (OSError, RuntimeError):
                    idle = False
                if idle:
                    self._release_workspace()
                    self.pending = False
                    self.pending_mmio = None
                    return
            sleep(0.05)

    def close(self):
        # The service calls this before relinquishing board ownership, even if a
        # daemon inference worker was active when shutdown started.
        with self.lock:
            self.disabled = True
            if not self.pending:
                self._release_workspace()
            self.packed_weights.clear()
            self.packed_bytes = 0
        while True:
            try:
                return self._wait_for_idle()
            except KeyboardInterrupt:
                continue


QueryProjection = OverlayProjection
