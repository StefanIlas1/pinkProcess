"""Drive live TinyStories generation with the PYNQ-Z2 output classifier."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import tempfile
from time import monotonic, perf_counter

from fpga.check_classifier import COMMAND, INPUT, SCORES, STATUS, board_lock, writes
from lm_reference.reference import (
    EDIT_PROMPT, MODEL_HASH, QUANTIZED_MODEL_HASH, ROWS, SOURCE_HASH, TOKENIZER_HASH, WIDTH,
    compile_runtime, read_pinned, rollback_trial, run, with_prompt,
)


class PynqClassifier:
    def __init__(self, bitstream):
        import numpy as np
        from pynq import Overlay

        self.overlay = Overlay(str(bitstream))
        self.mmio = self.overlay.classifier.mmio
        array = self.mmio.array
        self._np = np
        self._mapped_words = (isinstance(array, np.ndarray) and array.dtype == np.dtype("<u4")
                              and array.ndim == 1 and array.flags.c_contiguous and array.flags.aligned
                              and array.flags.writeable)
        self._arm_mmio = platform.system() == "Linux" and platform.machine().lower().startswith(("arm", "aarch64"))
        self._native_mmio = None
        if self._arm_mmio and self._mapped_words:
            library = Path(__file__).resolve().parents[1] / "fpga" / "mmio_words.so"
            try:
                self._native_mmio = ctypes.CDLL(str(library))
            except OSError:
                pass  # The word-at-a-time Python MMIO API also works without a compiler.
            else:
                for access in (self._native_mmio.pp_mmio_write32, self._native_mmio.pp_mmio_read32):
                    access.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
                    access.restype = None
        self.weight_revision = 0

    def write(self, address, values):
        self.wait_for(idle=True)
        if address < INPUT:
            self.weight_revision += 1
        # Leave activation and control writes on the ordinary MMIO path.
        if (self._mapped_words and isinstance(values, (bytes, bytearray))
                and 0 <= address < INPUT and address + len(values) <= INPUT
                and address % 4 == 0 and len(values) % 4 == 0):
            array = self.mmio.array
            words = self._np.frombuffer(values, "<u4")
            if self._native_mmio is not None:
                if not words.flags.aligned:
                    words = words.copy()
                self._native_mmio.pp_mmio_write32(array.ctypes.data + address, words.ctypes.data, words.size)
                return
            if not self._arm_mmio:
                array[address // 4:(address + len(values)) // 4] = words
                return
        for _, offset, word in writes(address, values):
            self.mmio.write(offset, word)

    def wait_for(self, *, idle):
        deadline = monotonic() + 1
        while True:
            status = self.mmio.read(STATUS)
            if not status & 1 and (idle or status & 2):
                return
            if monotonic() >= deadline:
                raise TimeoutError("PYNQ classifier did not become idle or finish the request.")

    def submit(self, activation):
        self.write(INPUT, activation)
        self.mmio.write(COMMAND, 1)
        self.wait_for(idle=False)

    def score(self, activation):
        return self.score_rows(activation, 0, ROWS)

    def score_rows(self, activation, first_row, row_count):
        if first_row < 0 or row_count < 0 or first_row + row_count > ROWS:
            raise ValueError("The requested score rows are outside the overlay.")
        output = self._np.empty(row_count, dtype=self._np.int32)
        self.score_rows_into(activation, first_row, output)
        return output.tolist()

    def score_rows_into(self, activation, first_row, output):
        """Read signed scores into a caller-owned, aligned int32 vector."""
        if (output.ndim != 1 or output.dtype != self._np.int32 or not output.flags.c_contiguous
                or not output.flags.aligned or not output.flags.writeable):
            raise ValueError("Expected an aligned, writable, contiguous int32 vector.")
        if first_row < 0 or first_row + output.size > ROWS:
            raise ValueError("The requested score rows are outside the overlay.")
        self.submit(activation)
        address = SCORES + 4 * first_row
        if self._native_mmio is not None:
            self._native_mmio.pp_mmio_read32(output.ctypes.data, self.mmio.array.ctypes.data + address, output.size)
        elif self._mapped_words and not self._arm_mmio:
            first = address // 4
            output[:] = self.mmio.array[first:first + output.size].view("int32")
        else:
            for i in range(output.size):
                word = self.mmio.read(address + 4 * i)
                output[i] = word if word < (1 << 31) else word - (1 << 32)

    def read_row(self, row):
        # A single 1 in the input makes this token's score equal that weight.
        activation = bytearray(WIDTH)
        values = bytearray()
        for column in range(WIDTH):
            activation[column] = 1
            self.submit(activation)
            word = self.mmio.read(SCORES + row * 4)
            weight = word - (1 << 32) if word & (1 << 31) else word
            if not -127 <= weight <= 127:
                raise RuntimeError(f"Invalid FPGA weight {weight} for row {row}, column {column}.")
            values.append(weight & 255)
            activation[column] = 0
        return bytes(values)

    def read_matrix(self):
        matrix = bytearray(ROWS * WIDTH)
        for column in range(WIDTH):
            activation = bytearray(WIDTH)
            activation[column] = 1
            weights = self.score(activation)
            if any(not -127 <= weight <= 127 for weight in weights):
                raise RuntimeError(f"Invalid FPGA weight in column {column}.")
            matrix[column::WIDTH] = bytes(weight & 255 for weight in weights)
        return bytes(matrix)


def run_trial(executable, cli, temporary, bitstream, *, selective=False, mode="int8"):
    row_file = temporary / "row.bin"
    edit = rollback_trial(executable, cli, row_file, selective=selective, mode=mode)
    selectivity = edit.get("selectivity")
    baseline = run(executable, cli, mode)
    target = edit["target_row"]
    original = bytes.fromhex(edit["original_row_hex"])
    replacement = bytes.fromhex(edit["patched_row_hex"])
    matrix = bytes(value & 255 for value in baseline[4][1])
    if matrix[target * WIDTH:(target + 1) * WIDTH] != original:
        raise RuntimeError("The edit and baseline disagree on the original row.")

    classifier = PynqClassifier(bitstream) if bitstream else None
    if classifier:
        classifier.write(0, matrix)
    phases = {}
    row_update_ms = {}
    for name, row in (("original", None), ("edited", replacement), ("restored", None)):
        patch_file = None
        if row is not None:
            row_file.write_bytes(target.to_bytes(2, "little") + row)
            patch_file = row_file
        expected = baseline if row is None else run(executable, cli, mode, patch_file)
        weights = expected[4][1]
        weight_rows = [weights[i:i + WIDTH] for i in range(0, len(weights), WIDTH)]
        latency_ms = []

        def classify(activation):
            if classifier:
                return classifier.score(activation)
            return [sum(w * x for w, x in zip(weights, activation)) for weights in weight_rows]

        def score(activation):
            start = perf_counter()
            scores = classify(activation)
            latency_ms.append((perf_counter() - start) * 1000)
            return scores

        control_scores = []
        try:
            if classifier and row is not None:
                start = perf_counter()
                classifier.write(target * WIDTH, row)
                row_update_ms["apply"] = (perf_counter() - start) * 1000
            observed_row = classifier.read_row(target) if classifier else None
            if observed_row is not None and observed_row != (original if row is None else row):
                raise RuntimeError(f"{name} FPGA weight row does not match the intended bytes.")
            start = perf_counter()
            actual = run(executable, cli, mode, patch_file, score=score)
            wall_seconds = perf_counter() - start
            if actual != expected or len(latency_ms) != len(actual[3]) or not latency_ms:
                raise RuntimeError(f"{name} generation did not match the independent C reference.")

            if selectivity:
                for control in selectivity["controls"]:
                    prompt = control["prompt"]
                    arguments = with_prompt(cli, prompt)
                    control_expected = run(executable, arguments, mode, patch_file)
                    control_actual = run(executable, arguments, mode, patch_file, score=classify)
                    if control_actual != control_expected:
                        raise RuntimeError(f"{name} generation for {prompt!r} did not match the C reference.")
                    control_scores.append({"prompt": prompt, "matched_scores": len(control_actual[3]) * ROWS})
        finally:
            if classifier and row is not None:
                start = perf_counter()
                classifier.write(target * WIDTH, original)
                row_update_ms["restore"] = (perf_counter() - start) * 1000

        phases[name] = {
            "expected_classifier_sha256": hashlib.sha256(bytes(value & 255 for value in weights)).hexdigest(),
            "observed_row_sha256": hashlib.sha256(observed_row).hexdigest() if observed_row is not None else None,
            "matched_scores": len(latency_ms) * ROWS,
            "forward_passes": len(latency_ms),
            "sampled_token_ids": actual[1],
            "generated_text": actual[0].decode().strip(),
            "median_classifier_ms": median(latency_ms),
            "max_classifier_ms": max(latency_ms),
            "generation_wall_seconds": wall_seconds,
        }
        if selectivity:
            phases[name]["control_scores"] = control_scores

    first_ids = [phase["sampled_token_ids"][0] for phase in phases.values()]
    if first_ids != edit["first_token_ids"]:
        raise RuntimeError("The live runs did not reproduce the original, edited and restored tokens.")
    report = {
        "backend": "pynq" if classifier else "software",
        "prompt": EDIT_PROMPT,
        "model_sha256": MODEL_HASH,
        "tokenizer_sha256": TOKENIZER_HASH,
        "runtime_source_sha256": SOURCE_HASH,
        "target_row": target,
        "target_name": edit["target_name"],
        "original_row_sha256": edit["original_row_sha256"],
        "edited_row_sha256": edit["patched_row_sha256"],
        "edited_row_hex": edit["patched_row_hex"],
        "row_update_ms": row_update_ms,
        "first_sampled_token_ids": first_ids,
        "target_sums_at_first_sample": edit["target_sums_at_first_sample"],
        "phases": phases,
    }
    if selectivity:
        report["selectivity"] = selectivity
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned llama2.c run.c")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixed", action="store_true", help="Use the exported integer checkpoint as --model")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selective", action="store_true", help="Fit and verify a selective one-row edit")
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--bitstream", type=Path, help="Overlay .bit with its matching .hwh")
    backend.add_argument("--software", action="store_true", help="Exercise the live score exchange without a board")
    args = parser.parse_args()
    read_pinned(args.model, QUANTIZED_MODEL_HASH if args.fixed else MODEL_HASH)
    read_pinned(args.tokenizer, TOKENIZER_HASH)
    mode = "fixed" if args.fixed else "int8"
    cli = [str(args.model), "-z", str(args.tokenizer), "-i", EDIT_PROMPT,
           "-t", "0", "-n", "32", "-s", "42"]
    with tempfile.TemporaryDirectory() as name:
        temporary = Path(name)
        executable = temporary / "tinystories"
        compile_runtime(args.source, executable, compiler=args.cc, fixed=args.fixed)
        if args.bitstream:
            with board_lock():
                report = run_trial(executable, cli, temporary, args.bitstream, selective=args.selective, mode=mode)
            report["bitstream_sha256"] = hashlib.sha256(args.bitstream.read_bytes()).hexdigest()
            report["hardware_metadata_sha256"] = hashlib.sha256(args.bitstream.with_suffix(".hwh").read_bytes()).hexdigest()
        else:
            report = run_trial(executable, cli, temporary, None, selective=args.selective, mode=mode)
    if args.fixed:
        report["quantized_model_sha256"] = QUANTIZED_MODEL_HASH
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for name, phase in report["phases"].items():
        print(f"{name}: {phase['matched_scores']:,} matching scores, "
              f"{phase['median_classifier_ms']:.3f} ms median classifier time")
        print(phase["generated_text"])
    if args.selective:
        control_total = sum(control["matched_scores"] for phase in report["phases"].values()
                            for control in phase["control_scores"])
        print(f"{control_total:,} matching scores across the additional prompts and all three phases")
    print(f"First sampled tokens: {report['first_sampled_token_ids']}")


if __name__ == "__main__":
    main()
