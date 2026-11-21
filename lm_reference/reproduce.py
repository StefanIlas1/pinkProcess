"""Run and record the complete fixed-point reference or PYNQ regression suite."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from time import perf_counter


def run_module(module, arguments=(), *, commands=None):
    result = subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                            input=commands, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{module} failed with status {result.returncode}.\n{result.stdout}{result.stderr}")
    return result.stdout


def verify_operator(common, bitstream, patch_file, live_report):
    patch = json.loads(patch_file.read_text())
    paths = {"valid": patch_file}
    variants = {"wrong-model": {"model_sha256": "0" * 64},
                "truncated": {"payload_hex": patch["payload_hex"][:-2]}}
    for name, changes in variants.items():
        paths[name] = patch_file.with_name(f"{name}.patch.json")
        paths[name].write_text(json.dumps({**patch, **changes}) + "\n")
    commands = [
        "status 326", "generate", f"apply {shlex.quote(str(paths['wrong-model']))}", "status 326",
        f"apply {shlex.quote(str(paths['truncated']))}", "status 326", "verify",
        f"apply {shlex.quote(str(paths['valid']))}", "verify", "generate",
        f"apply {shlex.quote(str(paths['truncated']))}", "verify",
        "rollback", "verify", "generate", "quit",
    ]
    transcript = run_module("lm_reference.patch_operator",
                            ["serve", *common, "--bitstream", bitstream], commands="\n".join(commands) + "\n")
    responses = [json.loads(line) for line in transcript.splitlines()]
    expected_commands = ["ready", *(shlex.split(command)[0] for command in commands)]
    if [record["command"] for record in responses] != expected_commands:
        raise RuntimeError("The operator transcript did not contain one response per command.")
    errors = {3: "different base model", 5: "exactly 64 bytes", 11: "exactly 64 bytes"}
    for index, response in enumerate(responses):
        if index in errors:
            if response["ok"] or errors[index] not in response["error"]:
                raise RuntimeError(f"Unexpected patch validation result at command {index}.")
        elif not response["ok"]:
            raise RuntimeError(f"The operator failed command {index}.")
    reports = live_report["phases"]
    base = reports["original"]["expected_classifier_sha256"]
    patched = reports["edited"]["expected_classifier_sha256"]
    original_row = live_report["original_row_sha256"]
    if any(responses[i]["result"]["observed_row_sha256"] != original_row for i in (1, 4, 6)):
        raise RuntimeError("A rejected patch changed the base row.")
    for index, expected in ((0, base), (7, base), (9, patched), (12, patched), (14, base)):
        if responses[index]["result"]["observed_classifier_sha256"] != expected:
            raise RuntimeError(f"The board matrix was wrong at command {index}.")
    for index, name in ((2, "original"), (10, "edited"), (15, "restored")):
        report = reports[name]
        result = responses[index]["result"]
        keys = ("sampled_token_ids", "generated_text", "matched_scores")
        if any(result[key] != report[key] for key in keys):
            raise RuntimeError(f"The {name} operator run did not reproduce the fixed-point reference.")
    return transcript


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "tokenizer", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--bitstream", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    common = ["--source", args.source, "--model", args.model, "--tokenizer", args.tokenizer,
              "--cc", args.cc, "--fixed"]
    vectors = args.out / "reference-vectors.json"
    live = args.out / "live.json"
    patch = args.out / "tim.patch.json"
    reports = [vectors, live, patch]
    start = perf_counter()
    boot = Path("/proc/sys/kernel/random/boot_id")
    metadata = {"backend": "pynq" if args.bitstream else "software"}
    if boot.is_file():
        metadata["boot_id"] = boot.read_text().strip()
    metadata["reference_stdout"] = run_module(
        "lm_reference.reference", [*common, "--selective", "--vectors", vectors])
    checker = [vectors, "--bitstream", args.bitstream] if args.bitstream else [vectors]
    metadata["rtl_replay_stdout"] = run_module("fpga.check_classifier", checker)
    run_module("lm_reference.patch_operator", ["pack", "--reference", vectors, "--out", patch])
    backend = ["--bitstream", args.bitstream] if args.bitstream else ["--software"]
    metadata["live_stdout"] = run_module(
        "lm_reference.live_demo", [*common, *backend, "--selective", "--report", live])
    if args.bitstream:
        transcript = args.out / "operator.jsonl"
        transcript.write_text(verify_operator(common, args.bitstream, patch, json.loads(live.read_text())))
        reports.append(transcript)
    metadata["local_suite_seconds"] = perf_counter() - start
    metadata["sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in reports}
    (args.out / "suite.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{metadata['backend']} suite passed in {metadata['local_suite_seconds']:.2f} s")


if __name__ == "__main__":
    main()
