"""Prepare and run the fixed LM001 comparison protocol on the integer runtime."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import tempfile

from lm_reference.patch_operator import RowPatch
from lm_reference.reference import (
    EDIT_PROMPT, FIT_PROMPTS, MODEL_HASH, QUANTIZED_MODEL_HASH, TOKENIZER_HASH, WIDTH, ROWS,
    compile_runtime, fit_selective_row, read_pinned, row_constraints, run, with_prompt,
)

TARGET = 326
PROMPT_PREFIX = "The girl in this story is named Tim. "
TARGET_PROMPTS = (
    "Once upon a time, there was a young girl named",
    "Once upon a time there was a little girl named",
    "Once upon a time, a little girl named",
    "One day, there was a little girl named",
    "Once upon a time, there lived a little girl named",
    "There was once a little girl named",
)
CONTROL_PROMPTS = (
    "Once upon a time, there was a small dog named",
    "Once upon a time, there was a little boy called",
    "Tim and Lily wanted to find a new toy",
    "One day, a little rabbit found a carrot",
    "Once upon a time, a girl named Lily found a bird",
    "Lily went outside to meet her friend",
)
NEW_TOKENS = 20
PROTOCOL = {
    "id": "LM001.v1",
    "target_token_id": TARGET,
    "cases": [{"id": f"{group}-{index}", "group": group, "prompt": prompt}
              for group, prompts in (("target", TARGET_PROMPTS), ("control", CONTROL_PROMPTS))
              for index, prompt in enumerate(prompts, 1)],
    "prompt_only_prefix": PROMPT_PREFIX,
    "max_sampled_tokens": NEW_TOKENS,
    "greedy_seed": 42,
    "success": {"minimum_targets_selecting_tim": 4, "minimum_unchanged_control_sequences": 5},
}


def digest(data):
    return sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def pack_row(base, payload):
    original = base[TARGET * WIDTH:(TARGET + 1) * WIDTH]
    return {
        "format": "pinkprocess.int8-output-row.v1",
        "target_row": TARGET,
        "model_sha256": MODEL_HASH,
        "quantized_model_sha256": QUANTIZED_MODEL_HASH,
        "base_classifier_sha256": digest(base),
        "original_row_sha256": digest(original),
        "payload_bytes": WIDTH,
        "payload_sha256": digest(payload),
        "payload_hex": payload.hex(),
    }


def prepare(executable, cli, destination):
    before = run(executable, cli, "fixed")
    prompts = (EDIT_PROMPT + " Tim", *FIT_PROMPTS)
    preserved = [run(executable, with_prompt(cli, prompt), "fixed") for prompt in prompts]
    if preserved[0][5] != before[5] + [TARGET]:
        raise RuntimeError("Tim is not the expected single completion token on the authoring prompt.")
    weights = before[4][1]
    original = weights[TARGET * WIDTH:(TARGET + 1) * WIDTH]
    base = bytes(weight & 255 for weight in weights)
    conventional, count = fit_selective_row(original, TARGET, before, preserved)
    examples = list(row_constraints(TARGET, before, preserved))
    assert len(examples) == count
    write_json(destination / "authoring.json", {
        "instructions": (
            "Directly propose one replacement INT8 row of 64 signed integers from -127 to 127. "
            "Use these authoring examples only. The target row should beat best_other_raw_sum "
            "when target_should_win is true and score strictly below it otherwise. The score is "
            "the integer dot product of the activation and the replacement row. Prefer small "
            "changes from the original row. Return JSON with row and a short method description."
        ),
        "target_row": TARGET,
        "original_row": original,
        "target_name": "Tim",
        "authoring_prompts": [EDIT_PROMPT, *prompts],
        "examples": [
            {"activation": activation, "best_other_raw_sum": best_other, "target_should_win": win}
            for activation, best_other, win in examples
        ],
    })
    write_json(destination / "conventional.patch.json", pack_row(base, conventional))
    write_json(destination / "protocol.json", PROTOCOL)


def record(trace, base):
    if not trace[1]:
        raise RuntimeError("The runtime did not sample a token.")
    pos = len(trace[5]) - 1
    scores = trace[3][pos]
    target_score = scores[TARGET]
    first_change = next(({
        "sample": index, "original_token_id": a, "candidate_token_id": b,
    } for index, (a, b) in enumerate(zip_longest(base[1], trace[1]), 1) if a != b), None)
    return {
        "first_token_id": trace[1][0],
        "target_rank": 1 + sum(score > target_score or (i < TARGET and score == target_score)
                               for i, score in enumerate(scores)),
        "target_score": target_score,
        "sampled_token_ids": trace[1],
        "target_selections": trace[1].count(TARGET),
        "first_change": first_change,
        "text": trace[0].decode().strip(),
        "matched_scores_if_on_board": len(trace[3]) * ROWS,
    }


def evaluate(executable, cli, directory, raw_response, bitstream=None):
    from fpga.check_classifier import board_lock
    from lm_reference.live_demo import PynqClassifier
    from lm_reference.patch_operator import PatchSession

    raw = raw_response.read_bytes()
    response = json.loads(raw)
    row = response.get("row") if isinstance(response, dict) else None
    if not isinstance(row, list) or len(row) != WIDTH or any(type(x) is not int or not -127 <= x <= 127 for x in row):
        raise ValueError("Astra response must contain 64 signed INT8 values under the row key.")
    before = run(executable, cli, "fixed")
    base = bytes(weight & 255 for weight in before[4][1])
    conventional_file = directory / "conventional.patch.json"
    conventional = RowPatch.parse(json.loads(conventional_file.read_text()))
    if conventional != RowPatch.parse(pack_row(base, conventional.payload)):
        raise ValueError("Conventional patch does not match the prepared model and target row.")
    astra_bundle = pack_row(base, bytes(value & 255 for value in row))
    astra = RowPatch.parse(astra_bundle)
    protocol_file = directory / "protocol.json"
    if json.loads(protocol_file.read_text()) != PROTOCOL:
        raise ValueError("Prepared experiment protocol differs from the current protocol.")
    freeze = {
        "protocol_sha256": digest(protocol_file.read_bytes()),
        "authoring_packet_sha256": digest((directory / "authoring.json").read_bytes()),
        "conventional_payload_sha256": digest(conventional.payload),
        "astra_payload_sha256": digest(astra.payload),
        "astra_raw_response_sha256": digest(raw),
        "quantized_model_sha256": QUANTIZED_MODEL_HASH,
    }
    frozen_file = directory / "frozen-candidates.json"
    if frozen_file.exists():
        if json.loads(frozen_file.read_text()) != freeze:
            raise ValueError("Prepared candidates or protocol differ from the frozen experiment.")
    else:
        write_json(frozen_file, freeze)
    write_json(directory / "astra.patch.json", astra_bundle)
    raw_copy = directory / "astra-response.json"
    if raw_response.resolve() != raw_copy.resolve():
        raw_copy.write_bytes(raw)

    baselines = {}
    conditions = {}
    with ExitStack() as stack:
        session = None
        if bitstream:
            stack.enter_context(board_lock())
            classifier = PynqClassifier(bitstream)
            session = PatchSession(classifier, base, quantized_model_hash=QUANTIZED_MODEL_HASH)
            stack.callback(session.rollback)
            session.verify()
        row_file = directory / "evaluation-row.bin"
        try:
            for name, patch in (("original", None), ("prompt_only", None),
                                ("conventional", conventional), ("astra", astra)):
                if patch is not None:
                    row_file.write_bytes(patch.row.to_bytes(2, "little") + patch.payload)
                    if session:
                        session.apply(patch)
                cases = []
                for case in PROTOCOL["cases"]:
                    prompt = (PROMPT_PREFIX if name == "prompt_only" else "") + case["prompt"]
                    args = with_prompt(cli, prompt)
                    step_index = args.index("-n") + 1
                    args[step_index] = "1"
                    prompt_ids = run(executable, args, "fixed")[5]
                    args[step_index] = str(len(prompt_ids) - 1 + NEW_TOKENS)
                    candidate_file = row_file if patch else None
                    trace = run(executable, args, "fixed", candidate_file)
                    if session and run(executable, args, "fixed", candidate_file, score=classifier.score) != trace:
                        raise RuntimeError(f"The board diverged on {name} and {case['id']}.")
                    if name == "original":
                        baselines[case["id"]] = trace
                    result = record(trace, baselines[case["id"]])
                    if session is None:
                        del result["matched_scores_if_on_board"]
                    cases.append({**case, "actual_prompt": prompt, **result})
                if session:
                    session.verify()
                    session.rollback()
                    session.verify()
                hits = sum(case["group"] == "target" and case["first_token_id"] == TARGET for case in cases)
                unchanged = sum(case["group"] == "control" and case["first_change"] is None for case in cases)
                success = PROTOCOL["success"]
                conditions[name] = {
                    "target_hits": hits,
                    "controls_unchanged": unchanged,
                    "met_thresholds": (hits >= success["minimum_targets_selecting_tim"]
                                       and unchanged >= success["minimum_unchanged_control_sequences"]),
                    "cases": cases,
                }
        finally:
            row_file.unlink(missing_ok=True)
    report = {"protocol": PROTOCOL, "frozen": freeze, "backend": "pynq" if bitstream else "software",
              "conditions": conditions}
    if bitstream:
        report["bitstream_sha256"] = digest(bitstream.read_bytes())
        report["hardware_metadata_sha256"] = digest(bitstream.with_suffix(".hwh").read_bytes())
    report_path = directory / ("lm001-pynq.json" if bitstream else "lm001-software.json")
    write_json(report_path, report)
    targets = sum(case["group"] == "target" for case in PROTOCOL["cases"])
    controls = len(PROTOCOL["cases"]) - targets
    for name, condition in conditions.items():
        print(f"{name}: targets {condition['target_hits']}/{targets}, "
              f"unchanged controls {condition['controls_unchanged']}/{controls}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "evaluate"):
        command = commands.add_parser(name)
        for path in ("source", "model", "tokenizer", "out"):
            command.add_argument(f"--{path}", type=Path, required=True)
        command.add_argument("--cc", default="cc")
    command.add_argument("--astra-response", type=Path, required=True)
    command.add_argument("--bitstream", type=Path)
    args = parser.parse_args()
    read_pinned(args.model, QUANTIZED_MODEL_HASH)
    read_pinned(args.tokenizer, TOKENIZER_HASH)
    cli = [str(args.model), "-z", str(args.tokenizer), "-i", EDIT_PROMPT, "-t", "0", "-n", "32", "-s", "42"]
    with tempfile.TemporaryDirectory() as temporary:
        executable = Path(temporary) / "tinystories"
        compile_runtime(args.source, executable, compiler=args.cc, fixed=True)
        args.out.mkdir(parents=True, exist_ok=True)
        if args.command == "prepare":
            prepare(executable, cli, args.out)
        else:
            evaluate(executable, cli, args.out, args.astra_response, args.bitstream)


if __name__ == "__main__":
    main()
