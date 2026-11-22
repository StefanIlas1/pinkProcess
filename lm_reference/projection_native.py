"""Optional native packing and dispatch for an existing PYNQ projection MMIO map."""

import ctypes as C
from functools import lru_cache
from pathlib import Path

import numpy as np

_POINTER = C.c_void_p
_SIZE = C.c_size_t
_STRIDE = C.c_ssize_t
_WORD = C.c_uint32


@lru_cache(maxsize=1)
def _load_library(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / "fpga" / "projection_native.so"
    try:
        lib = C.CDLL(str(path))
        lib.pp_projection_pack.argtypes = [_POINTER, _POINTER, _SIZE, _SIZE, _SIZE, _STRIDE, _STRIDE]
        lib.pp_projection_pack.restype = None
        lib.pp_projection_unpack.argtypes = [
            _POINTER, _POINTER, _SIZE, _SIZE, _SIZE, _SIZE, _STRIDE, _STRIDE,
        ]
        lib.pp_projection_unpack.restype = None
        lib.pp_projection_dispatch.argtypes = [
            _POINTER, *([_WORD] * 7), C.c_uint64, C.POINTER(_WORD),
        ]
        lib.pp_projection_dispatch.restype = C.c_int
    except (OSError, AttributeError):
        return None
    return lib


class NativeProjection:
    """Use the native helper with the existing MMIO map and NumPy buffers."""

    def __init__(self, registers, library):
        self.registers = registers
        self.library = library

    @classmethod
    def load(cls, mmio):
        registers = getattr(mmio, "array", None)
        if (not isinstance(registers, np.ndarray) or registers.dtype != np.uint32
                or not registers.flags.c_contiguous or not registers.flags.aligned
                or not registers.flags.writeable or registers.size < 8):
            return None
        library = _load_library()
        return None if library is None else cls(registers, library)

    def _pack(self, destination, source, lanes):
        rows, depth = source.shape
        size = (rows + lanes - 1) // lanes * depth * lanes
        if (source.dtype != np.int8 or destination.dtype != np.int8 or destination.size != size
                or not destination.flags.c_contiguous or not destination.flags.writeable):
            raise ValueError("Invalid signed INT8 buffers for native projection packing.")
        self.library.pp_projection_pack(destination.ctypes.data, source.ctypes.data,
                                        rows, depth, lanes, *source.strides)

    def pack_activations(self, destination, source):
        self._pack(destination, source, 8)

    def pack_weights(self, destination, source, decode):
        self._pack(destination, source, 64 if decode else 8)

    def unpack_scores(self, destination, source):
        batch, rows = destination.shape
        position_lanes, output_lanes = (1, 64) if batch == 1 else (8, 8)
        size = ((batch + position_lanes - 1) // position_lanes
                * ((rows + output_lanes - 1) // output_lanes) * 64)
        if (destination.dtype != np.int32 or not destination.flags.writeable
                or source.dtype != np.int32 or source.size != size
                or not source.flags.c_contiguous or not source.flags.aligned):
            raise ValueError("Invalid signed INT32 buffers for native projection unpacking.")
        self.library.pp_projection_unpack(destination.ctypes.data, source.ctypes.data,
                                          batch, rows, position_lanes, output_lanes,
                                          *destination.strides)

    def dispatch(self, addresses, decode, depth, position_tiles, output_tiles, timeout):
        status = _WORD()
        timeout_ns = int(timeout * 1_000_000_000)
        if (len(addresses) != 3 or not 0 < timeout_ns < (1 << 62)
                or any(not 0 < value <= 65535 for value in (depth, position_tiles, output_tiles))
                or any(not 0 <= value < (1 << 32) or value % 64 for value in addresses)):
            raise ValueError("Invalid native projection launch parameters.")
        result = self.library.pp_projection_dispatch(
            self.registers.ctypes.data, int(decode), depth, position_tiles, output_tiles,
            *addresses, timeout_ns, C.byref(status),
        )
        if result == 1:
            raise RuntimeError("Projection accelerator is already busy.")
        if result != 0:
            raise TimeoutError("Projection accelerator did not complete before the deadline.")
        return status.value
