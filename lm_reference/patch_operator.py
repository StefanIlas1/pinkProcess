"""Package a TinyStories row edit and operate it on a live PYNQ-Z2."""

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import re
import shlex
import signal
from subprocess import CalledProcessError
import sys
import tempfile
from time import perf_counter

from fpga.check_classifier import board_lock
from lm_reference.live_demo import PynqClassifier
from lm_reference.reference import (
    EDIT_PROMPT, FIT_PROMPTS, HELD_OUT_PROMPTS, MODEL_HASH, QUANTIZED_MODEL_HASH, ROWS,
    SHARED_PREFIX_PROMPT, TOKENIZER_HASH, WIDTH, compile_runtime, read_pinned, run, with_prompt,
)

MAX_PROMPT_BYTES = 256
QUERY_TOKENS = 32


def query_prompt(prompt):
    if not isinstance(prompt, str) or any(mark in prompt for mark in ("\r", "\n", "\0")):
        raise ValueError("Enter a single-line text prompt.")
    prompt = prompt.strip()
    if not prompt or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError(f"Enter a prompt between 1 and {MAX_PROMPT_BYTES} UTF-8 bytes.")
    return prompt


class MemoryClassifier:
    def __init__(self):
        self.memory = bytearray(ROWS * WIDTH)

    def write(self, address, values):
        self.memory[address:address + len(values)] = values

    def read_row(self, row):
        return bytes(self.memory[row * WIDTH:(row + 1) * WIDTH])

    def read_matrix(self):
        return bytes(self.memory)


@dataclass(frozen=True)
class RowPatch:
    row: int
    model_hash: str
    classifier_hash: str
    original_row_hash: str
    payload: bytes
    quantized_model_hash: str | None = None

    @classmethod
    def parse(cls, document):
        if not isinstance(document, dict) or document.get("format") != "pinkprocess.int8-output-row.v1":
            raise ValueError("Expected a pinkProcess INT8 output-row patch v1.")
        row = document.get("target_row")
        if type(row) is not int or not 0 <= row < ROWS:
            raise ValueError(f"Target row must be an integer from 0 to {ROWS - 1}.")
        fields = ["model_sha256", "base_classifier_sha256", "original_row_sha256", "payload_sha256"]
        if "quantized_model_sha256" in document:
            fields.append("quantized_model_sha256")
        for field in fields:
            digest = document.get(field)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid {field}.")
        encoded = document.get("payload_hex")
        payload_bytes = document.get("payload_bytes")
        if (type(payload_bytes) is not int or payload_bytes != WIDTH or not isinstance(encoded, str)
                or not re.fullmatch(rf"[0-9a-f]{{{2 * WIDTH}}}", encoded)):
            raise ValueError(f"Patch payload must encode exactly {WIDTH} bytes.")
        payload = bytes.fromhex(encoded)
        if 128 in payload:
            raise ValueError("Patch weights must be in the symmetric INT8 range −127 to 127.")
        if sha256(payload).hexdigest() != document["payload_sha256"]:
            raise ValueError("Patch payload SHA-256 mismatch.")
        return cls(row, document["model_sha256"], document["base_classifier_sha256"],
                   document["original_row_sha256"], payload, document.get("quantized_model_sha256"))


def package(reference, output):
    document = json.loads(reference.read_text(encoding="utf-8"))
    try:
        trial = document["rollback_trial"]
        patch = {
            "format": "pinkprocess.int8-output-row.v1",
            "target_row": trial["target_row"],
            "model_sha256": document["model_sha256"],
            "base_classifier_sha256": trial["original_matrix_sha256"],
            "original_row_sha256": trial["original_row_sha256"],
            "payload_bytes": WIDTH,
            "payload_sha256": trial["patched_row_sha256"],
            "payload_hex": trial["patched_row_hex"],
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("Expected reference vectors exported with --selective or --edit-demo.") from exc
    if "quantized_model_sha256" in document:
        patch["quantized_model_sha256"] = document["quantized_model_sha256"]
    RowPatch.parse(patch)
    output.write_text(json.dumps(patch, indent=2) + "\n", encoding="utf-8")


class BoardStateError(RuntimeError):
    """Board state could not be confirmed and the operator session must end."""


class PatchSession:
    def __init__(self, classifier, base, *, quantized_model_hash=None):
        if len(base) != ROWS * WIDTH:
            raise ValueError(f"Expected a {ROWS} × {WIDTH} INT8 output classifier.")
        self.classifier = classifier
        self.base = base
        self.base_hash = sha256(base).hexdigest()
        self.quantized_model_hash = quantized_model_hash
        self.active = None
        self.classifier.write(0, self.base)

    def base_row(self, row):
        return self.base[row * WIDTH:(row + 1) * WIDTH]

    def status(self, row=None):
        patch = self.active
        matrix_hash = self.base_hash
        if patch is not None:
            start = patch.row * WIDTH
            matrix_hash = sha256(self.base[:start] + patch.payload + self.base[start + WIDTH:]).hexdigest()
            if row is None:
                row = patch.row
        result = {
            "active_row": patch.row if patch is not None else None,
            "expected_classifier_sha256": matrix_hash,
        }
        if row is not None:
            if not 0 <= row < ROWS:
                raise ValueError(f"Row must be from 0 to {ROWS - 1}.")
            observed = self.classifier.read_row(row)
            expected = patch.payload if patch is not None and row == patch.row else self.base_row(row)
            if observed != expected:
                raise BoardStateError(f"FPGA row {row} differs from the expected active weights.")
            result.update(row=row, observed_row_sha256=sha256(observed).hexdigest())
        return result

    def verify(self):
        result = self.status()
        observed_hash = sha256(self.classifier.read_matrix()).hexdigest()
        if observed_hash != result["expected_classifier_sha256"]:
            raise BoardStateError("FPGA classifier matrix differs from the expected active weights.")
        result["observed_classifier_sha256"] = observed_hash
        return result

    def validate_patch(self, patch):
        if patch.model_hash != MODEL_HASH:
            raise ValueError("Patch targets a different base model SHA-256.")
        if patch.quantized_model_hash != self.quantized_model_hash:
            raise ValueError("Patch targets a different inference checkpoint.")
        if patch.classifier_hash != self.base_hash:
            raise ValueError("Patch targets a different base classifier SHA-256.")
        if patch.original_row_hash != sha256(self.base_row(patch.row)).hexdigest():
            raise ValueError("Patch original row SHA-256 does not match the base row.")

    def apply(self, patch):
        self.validate_patch(patch)
        if self.active == patch:
            return self.status(patch.row)
        if self.active is not None:
            raise ValueError("Roll back the active patch before applying a different patch.")
        self.status(patch.row)
        try:
            self.active = patch
            self.classifier.write(patch.row * WIDTH, patch.payload)
            return self.status(patch.row)
        except BaseException as exc:
            self.rollback()
            if isinstance(exc, (OSError, RuntimeError)):
                raise RuntimeError(f"Patch application failed. The original row was restored. {exc}") from exc
            raise

    def rollback(self):
        patch = self.active
        if patch is None:
            return self.status()
        try:
            self.classifier.write(patch.row * WIDTH, self.base_row(patch.row))
            self.active = None
            return self.status(patch.row)
        except BaseException as exc:
            # Keep the row available for a recovery attempt at session shutdown.
            self.active = patch
            raise BoardStateError(f"Could not verify restoration of FPGA row {patch.row}.") from exc


def respond(command, *, result=None, error=None):
    message = {"command": command, "ok": error is None}
    if error is None:
        message["result"] = result
    else:
        message["error"] = str(error)
    print(json.dumps(message), flush=True)


def operate(session, generate, preview=None):
    try:
        for line in sys.stdin:
            command = None
            try:
                words = shlex.split(line)
                if not words:
                    continue
                command, *arguments = words
                match command, arguments:
                    case "status", []:
                        result = session.status()
                    case "status", [row]:
                        result = session.status(int(row))
                    case "verify", []:
                        result = session.verify()
                    case "apply" | "preview", [filename]:
                        document = json.loads(Path(filename).read_text(encoding="utf-8"))
                        patch = RowPatch.parse(document)
                        result = session.apply(patch) if command == "apply" else preview(patch)
                    case "generate", []:
                        result = generate()
                    case "generate", [prompt]:
                        result = generate(query_prompt(prompt))
                    case "rollback" | "quit", []:
                        result = session.rollback()
                    case _:
                        raise ValueError("Commands are status [ROW], verify, preview FILE, apply FILE, generate [PROMPT], rollback, quit.")
                respond(command, result=result)
                if command == "quit":
                    break
            except BoardStateError:
                raise
            except (ValueError, OSError, RuntimeError, CalledProcessError) as exc:
                respond(command, error=exc)
    finally:
        if session.active is not None:
            session.rollback()


def stop_session(signum, _frame):
    raise SystemExit(128 + signum)


def serve(args):
    model_hash = QUANTIZED_MODEL_HASH if args.fixed else None
    read_pinned(args.model, model_hash or MODEL_HASH)
    read_pinned(args.tokenizer, TOKENIZER_HASH)
    mode = "fixed" if args.fixed else "int8"
    cli = [str(args.model), "-z", str(args.tokenizer), "-i", EDIT_PROMPT,
           "-t", "0", "-n", "32", "-s", "42"]
    bitstream = args.bitstream
    hardware = {}
    if bitstream:
        hardware = {
            "bitstream_sha256": sha256(bitstream.read_bytes()).hexdigest(),
            "hardware_metadata_sha256": sha256(bitstream.with_suffix(".hwh").read_bytes()).hexdigest(),
        }
    with tempfile.TemporaryDirectory() as name:
        executable = Path(name) / "tinystories"
        compile_runtime(args.source, executable, compiler=args.cc, fixed=args.fixed)
        baseline = run(executable, cli, mode)
        base = bytes(weight & 255 for weight in baseline[4][1])
        row_file = Path(name) / "row.bin"
        with board_lock() if bitstream else nullcontext():
            classifier = PynqClassifier(bitstream) if bitstream else MemoryClassifier()
            session = PatchSession(classifier, base, quantized_model_hash=model_hash)

            def write_row_file(patch):
                row_file.write_bytes(patch.row.to_bytes(2, "little") + patch.payload)
                return row_file

            def preview(patch):
                session.validate_patch(patch)
                patch_file = write_row_file(patch)
                rows = []
                for prompt in (EDIT_PROMPT, *FIT_PROMPTS, *HELD_OUT_PROMPTS, SHARED_PREFIX_PROMPT):
                    arguments = with_prompt(cli, prompt)
                    before = baseline if prompt == EDIT_PROMPT else run(executable, arguments, mode)
                    after = run(executable, arguments, mode, patch_file)
                    changes = ({"sample": index, "original_token": one, "edited_token": two}
                               for index, (one, two) in enumerate(zip_longest(before[1], after[1]), 1)
                               if one != two)
                    rows.append({
                        "prompt": prompt,
                        "original": before[0].decode().strip(),
                        "edited": after[0].decode().strip(),
                        "first_change": next(changes, None),
                    })
                original = session.base_row(patch.row)
                return {
                    "row": patch.row,
                    "original_row_hex": original.hex(),
                    "edited_row_hex": patch.payload.hex(),
                    "changed_bytes": sum(a != b for a, b in zip(original, patch.payload)),
                    "prompts": rows,
                }

            def generate(prompt=None):
                arguments = cli
                if prompt is not None and prompt != EDIT_PROMPT:
                    arguments = with_prompt(cli, prompt)
                    step_index = arguments.index("-n") + 1
                    arguments[step_index] = "1"
                    prompt_ids = run(executable, arguments, mode)[5]
                    arguments[step_index] = str(len(prompt_ids) - 1 + QUERY_TOKENS)

                patch = session.active
                patch_file = write_row_file(patch) if patch is not None else None
                state = session.status()
                expected = None
                if bitstream:
                    expected = baseline if arguments is cli and patch is None else run(
                        executable, arguments, mode, patch_file)
                start = perf_counter()
                score = classifier.score if bitstream else None
                actual = run(executable, arguments, mode, patch_file, score=score)
                generation_seconds = perf_counter() - start
                if bitstream and actual != expected:
                    raise BoardStateError("Live generation differs from the C reference.")
                weights_hash = sha256(bytes(weight & 255 for weight in actual[4][1])).hexdigest()
                if weights_hash != state["expected_classifier_sha256"]:
                    raise BoardStateError("The runtime used a different set of classifier weights.")
                if not actual[1]:
                    raise ValueError("The prompt produced no continuation. Try a different prompt.")
                return {
                    **state,
                    "prompt": prompt if prompt is not None else EDIT_PROMPT,
                    "sampled_token_ids": actual[1],
                    "generated_text": actual[0].decode(errors="replace").strip(),
                    "matched_scores": len(actual[3]) * ROWS if bitstream else None,
                    "generation_seconds": generation_seconds,
                }

            respond("ready", result={
                **session.verify(),
                **hardware,
                "backend": "pynq" if bitstream else "software",
                "prompt": EDIT_PROMPT,
                "max_prompt_bytes": MAX_PROMPT_BYTES,
                "model_sha256": MODEL_HASH,
                "quantized_model_sha256": model_hash,
            })
            operate(session, generate, preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="Export a patch from a reference trial")
    pack.add_argument("--reference", type=Path, required=True)
    pack.add_argument("--out", type=Path, required=True)
    server = commands.add_parser("serve", help="Run an operator session with commands on stdin and JSON on stdout")
    for name in ("source", "model", "tokenizer"):
        server.add_argument(f"--{name}", type=Path, required=True)
    backend = server.add_mutually_exclusive_group(required=True)
    backend.add_argument("--bitstream", type=Path, help="Use the physical PYNQ-Z2")
    backend.add_argument("--software", action="store_true", help="Run the classifier on this computer")
    server.add_argument("--cc", default="cc")
    server.add_argument("--fixed", action="store_true", help="Use the exported integer checkpoint as --model")
    args = parser.parse_args()
    try:
        if args.command == "pack":
            package(args.reference, args.out)
        else:
            signal.signal(signal.SIGTERM, stop_session)
            signal.signal(signal.SIGHUP, stop_session)
            serve(args)
    except KeyboardInterrupt:
        parser.exit(130)
    except (OSError, ValueError, RuntimeError, CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
