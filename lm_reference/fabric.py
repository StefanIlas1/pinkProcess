"""Exact tiled projections using the existing PYNQ-Z2 INT8 overlay."""

from contextlib import nullcontext
from operator import index

ROWS = 512
WIDTH = 64


def _activation_planes(activation):
    """Represent signed int32 values as signed INT8 digits with radix 256."""
    if isinstance(activation, (bytes, bytearray)):
        values = [value if value < 128 else value - 256 for value in activation]
    else:
        values = [index(value) for value in activation]
    if any(not -(1 << 31) <= value < (1 << 31) for value in values):
        raise ValueError("Activation values must fit in signed int32.")

    planes = []
    place = 1
    while any(values):
        digits = [((value + 128) % 256) - 128 for value in values]
        if any(digits):
            planes.append((place, bytes(digit & 255 for digit in digits).ljust(WIDTH, b"\0")))
        values = [(value - digit) // 256 for value, digit in zip(values, digits)]
        place *= 256
    return planes


def _array_planes(np, values):
    """Prepack only the nonzero vectors in each signed radix-256 plane."""
    if values.dtype == np.int8:
        planes = [(1, values)]
    else:
        planes = []
        place = 1
        while values.any():
            carry = (values + 128) // 256
            planes.append((place, (values - carry * 256).astype(np.int8)))
            values = carry
            place *= 256

    vectors = {}
    for place, plane in planes:
        for i in np.flatnonzero(plane.any(axis=1)):
            digits = plane[i].tobytes().ljust(WIDTH, b"\0")
            vectors.setdefault(i, []).append((place, digits))
    return vectors


class TiledMatvec:
    """Compute row-major INT8 matrix products with signed int32 activations.

    The classifier provides write(byte_address, values) and score(activation).
    Share its lock with other clients of the overlay. The lock is released
    between vectors and tiles. A backend exposing a changing weight_revision
    on every weight write can reuse the current physical tile.
    """

    def __init__(self, classifier, *, lock=None, reserved_rows=3, reuse=True):
        if not 0 <= reserved_rows < ROWS:
            raise ValueError(f"Reserved rows must be between 0 and {ROWS - 1}.")
        self.classifier = classifier
        self.lock = lock if lock is not None else nullcontext()
        self.reserved_rows = reserved_rows
        self.reuse = reuse
        self._resident = None

    def _load_tile(self, address, tile):
        revision = getattr(self.classifier, "weight_revision", None)
        if self.reuse and revision is not None and self._resident == (revision, tile):
            return
        self._resident = None
        self.classifier.write(address, tile)
        if self.reuse and revision is not None:
            self._resident = (self.classifier.weight_revision, tile)

    def verify_row_bytes(self, payload):
        """Read one logical output row back from fabric through basis-vector scores."""
        payload = bytes(payload)
        if not payload or len(payload) % WIDTH or len(payload) > (ROWS - self.reserved_rows) * WIDTH:
            raise ValueError("The verification row must comprise complete physical rows within one tile.")
        count = len(payload) // WIDTH
        for column in range(WIDTH):
            vector = bytearray(WIDTH)
            vector[column] = 1
            with self.lock:
                self._load_tile(self.reserved_rows * WIDTH, payload)
                result = self.classifier.score_rows(vector, self.reserved_rows, count)
            if (any(not -128 <= value <= 127 for value in result)
                    or bytes(value & 255 for value in result) != payload[column::WIDTH]):
                raise RuntimeError("FPGA output-row readback failed.")

    def matvec(self, weights, activation, rows):
        """Return exact signed sums, before any row scaling or clipping."""
        return self.matmat(weights, (activation,), rows)[0]

    def matmat(self, weights, activations, rows):
        """Project a batch of vectors, reusing each tile across the batch."""
        if len(activations) == 0:
            raise ValueError("Supply at least one activation vector.")
        columns = len(activations[0])
        weights = memoryview(weights).cast("B")
        if (rows <= 0 or columns <= 0 or len(weights) != rows * columns
                or any(len(activation) != columns for activation in activations)):
            raise ValueError("Matrix bytes must match the positive row and activation dimensions.")

        sums = [[0] * rows for _ in activations]
        tiles = []
        for column in range(0, columns, WIDTH):
            vectors = []
            for activation, scores in zip(activations, sums):
                planes = _activation_planes(activation[column:column + WIDTH])
                if planes:
                    vectors.append((planes, scores))
            if vectors:
                tiles.append((column, min(WIDTH, columns - column), vectors))

        tile_rows = ROWS - self.reserved_rows
        address = self.reserved_rows * WIDTH
        score_rows = getattr(self.classifier, "score_rows", None)
        for first_row in range(0, rows, tile_rows):
            row_count = min(tile_rows, rows - first_row)
            for column, column_count, vectors in tiles:
                tile = bytearray(row_count * WIDTH)
                for offset in range(row_count):
                    start = (first_row + offset) * columns + column
                    target = offset * WIDTH
                    tile[target:target + column_count] = weights[start:start + column_count]
                tile = bytes(tile)

                for planes, scores in vectors:
                    with self.lock:
                        self._load_tile(address, tile)
                        partials = []
                        for place, digits in planes:
                            if score_rows is None:
                                first = self.reserved_rows
                                partial = self.classifier.score(digits)[first:first + row_count]
                            else:
                                partial = score_rows(digits, self.reserved_rows, row_count)
                            partials.append((place, partial))
                    for place, partial in partials:
                        if len(partial) != row_count:
                            raise RuntimeError("The classifier returned an incomplete score tile.")
                        for offset, value in enumerate(partial):
                            scores[first_row + offset] += place * int(value)
        return sums

    def matmat_array(self, weights, activations, rows):
        """Project an integer NumPy batch, returning exact int64 sums."""
        import numpy as np

        activations = np.asarray(activations)
        if activations.ndim != 2 or activations.dtype.kind not in "iu":
            raise ValueError("Supply a two-dimensional integer activation array.")
        batch, columns = activations.shape
        weights = np.frombuffer(weights, dtype=np.uint8)
        # Even the absolute sum of a value's balanced planes times a weight is below 2**40.
        if not batch or not 0 < columns < (1 << 23) or rows <= 0 or weights.size != rows * columns:
            raise ValueError("Matrix bytes must match positive dimensions with fewer than 2**23 columns.")
        if activations.dtype != np.int8:
            if int(activations.min()) < -(1 << 31) or int(activations.max()) >= (1 << 31):
                raise ValueError("Activation values must fit in signed int32.")
            activations = activations.astype(np.int64, copy=False)

        weights = weights.reshape(rows, columns)
        sums = np.zeros((batch, rows), dtype=np.int64)
        tile_rows = ROWS - self.reserved_rows
        score_rows_into = getattr(self.classifier, "score_rows_into", None)
        score_rows = getattr(self.classifier, "score_rows", None)
        scratch = np.empty((5, tile_rows), dtype=np.int32)
        for column in range(0, columns, WIDTH):
            vectors = _array_planes(np, activations[:, column:column + WIDTH])
            if not vectors:
                continue
            for first_row in range(0, rows, tile_rows):
                last_row = min(first_row + tile_rows, rows)
                row_count = last_row - first_row
                chunk = weights[first_row:last_row, column:column + WIDTH]
                if chunk.shape[1] != WIDTH:
                    padded = np.zeros((row_count, WIDTH), dtype=np.uint8)
                    padded[:, :chunk.shape[1]] = chunk
                    chunk = padded
                tile = chunk.tobytes()
                for i, planes in vectors.items():
                    partials = scratch[:len(planes), :row_count]
                    with self.lock:
                        self._load_tile(self.reserved_rows * WIDTH, tile)
                        for (_, digits), partial in zip(planes, partials):
                            if score_rows_into is not None:
                                score_rows_into(digits, self.reserved_rows, partial)
                                continue
                            if score_rows is not None:
                                values = score_rows(digits, self.reserved_rows, row_count)
                            else:
                                first = self.reserved_rows
                                values = self.classifier.score(digits)[first:first + row_count]
                            if len(values) != row_count:
                                raise RuntimeError("The classifier returned an incomplete score tile.")
                            partial[:] = values
                    scores = sums[i, first_row:last_row]
                    for (place, _), partial in zip(planes, partials):
                        if place == 1:
                            scores += partial
                        else:
                            scores += np.multiply(partial, place, dtype=np.int64)
        return sums
