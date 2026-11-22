"""Register contract for the optional on-FPGA FP32 vocabulary reducer."""

from time import monotonic, sleep

import numpy as np

CONTROL, CORE_ID = 0x00, 0x1C
SCORE_ADDRESS, SCALE_ADDRESS, ROWS, ACTIVATION_SCALE = 0x04, 0x08, 0x0C, 0x10
WINNER, WINNER_SCORE = 0x14, 0x18
ARGMAX_ID = 0x50415231
BUSY, DONE, ERROR = 1, 2, 4


def require_argmax_mmio(overlay):
    peripheral = getattr(overlay, "projection_argmax", None)
    if peripheral is None or peripheral.mmio.read(CORE_ID) != ARGMAX_ID:
        raise RuntimeError("The programmed overlay does not expose the FP32 vocabulary reducer.")
    return peripheral.mmio


def configure_argmax(mmio, scores, scales, rows, activation_scale):
    if mmio.read(CONTROL) & BUSY:
        raise RuntimeError("The vocabulary reducer is already busy.")
    mmio.write(SCORE_ADDRESS, int(scores.device_address))
    mmio.write(SCALE_ADDRESS, int(scales.device_address))
    mmio.write(ROWS, rows)
    mmio.write(ACTIVATION_SCALE, int(np.float32(activation_scale).view(np.uint32)))


def launch_argmax(mmio, timeout):
    mmio.write(CONTROL, 1)
    deadline = monotonic() + timeout
    while True:
        status = mmio.read(CONTROL)
        if not status & BUSY and status & (DONE | ERROR):
            return status
        if monotonic() >= deadline:
            raise TimeoutError("Vocabulary reduction did not complete before the deadline.")
        sleep(0.0001)


def argmax_result(mmio, status, rows):
    if status & ERROR:
        raise RuntimeError(f"Vocabulary reducer reported status {status:#x}.")
    index = mmio.read(WINNER)
    value = np.uint32(mmio.read(WINNER_SCORE)).view(np.float32)
    if not 0 <= index < rows or np.isnan(value):
        raise RuntimeError("Vocabulary reducer returned an invalid winner.")
    return index, value
