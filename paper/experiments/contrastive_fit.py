"""Contrastive one-row fit: add authoring-only naming negatives and girl positives, then score on the frozen extended set.

Pre-registered success: at alpha 1.0 the contrastive row selects Tim on at least 18 of 24 extended target
prompts and changes at most 2 of the 11 extended naming controls, with all 13 non-naming controls unchanged.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from lm_reference.experiment import TARGET, NEW_TOKENS, TARGET_PROMPTS as LM001_TARGETS, CONTROL_PROMPTS as LM001_CONTROLS
from lm_reference.reference import (
    EDIT_PROMPT, FIT_PROMPTS, QUANTIZED_MODEL_HASH, TOKENIZER_HASH, WIDTH,
    compile_runtime, read_pinned, run, with_prompt,
)
from paper.experiments.lm001_extended import TARGET_PROMPTS, CONTROL_PROMPTS

POSITIVE_AUTHORING = (
    "Once upon a time, there was a beautiful girl named",
    "Once upon a time, there was a shy little girl named",
    "Once upon a time, there was a little girl with a red hat named",
)
NEGATIVE_AUTHORING = (
    "Once upon a time, there was a little puppy named",
    "Once upon a time, there was a kitten named",
    "Once upon a time, there was a grandpa named",
    "Once upon a time, there was a big lion named",
    "Once upon a time, there was a duck named",
    "Once upon a time, there was a dad named",
    "Once upon a time, there was a toy car named",
    "Once upon a time, there was a turtle named",
)
NAMING_CONTROLS = CONTROL_PROMPTS[:11]


def fit(original, constraints):
    row = [float(w) for w in original]
    for _ in range(4000):
        for activation, bound, norm in constraints:
            deficit = bound + 256 - sum(w * x for w, x in zip(row, activation))
            if deficit > 0 and norm:
                step = deficit / norm
                row = [max(-127, min(127, w + step * x)) for w, x in zip(row, activation)]
        candidate = [round(w) for w in row]
        if all(sum(w * x for w, x in zip(candidate, a)) > b for a, b, _ in constraints):
            return candidate, True
    return [round(w) for w in row], False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("source", "model", "tokenizer", "out"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    args = parser.parse_args()
    read_pinned(args.model, QUANTIZED_MODEL_HASH)
    read_pinned(args.tokenizer, TOKENIZER_HASH)
    held_out = set(TARGET_PROMPTS) | set(CONTROL_PROMPTS) | set(LM001_TARGETS) | set(LM001_CONTROLS)
    assert not held_out & (set(POSITIVE_AUTHORING) | set(NEGATIVE_AUTHORING))
    cli = [str(args.model), "-z", str(args.tokenizer), "-i", EDIT_PROMPT, "-t", "0", "-n", "40", "-s", "42"]
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        executable = temporary / "tinystories"
        compile_runtime(args.source, executable, fixed=True)

        def trace(prompt, row_file=None, new_tokens=NEW_TOKENS):
            a = with_prompt(cli, prompt)
            a[a.index("-n") + 1] = "1"
            steps = len(run(executable, a, "fixed")[5]) - 1 + new_tokens
            a[a.index("-n") + 1] = str(steps)
            return run(executable, a, "fixed", row_file)

        before = trace(EDIT_PROMPT)
        weights = before[4][1]
        original = weights[TARGET * WIDTH:(TARGET + 1) * WIDTH]

        def example(t, position, wins):
            activation = t[2][position][1]
            scores = t[3][position]
            best_other = max(scores[:TARGET] + scores[TARGET + 1:])
            sign = 1 if wins else -1
            return [sign * v for v in activation], sign * best_other, sum(v * v for v in activation)

        constraints = [example(before, len(before[5]) - 1, True)]
        for prompt in (EDIT_PROMPT + " Tim", *FIT_PROMPTS):
            t = trace(prompt)
            for position, token in enumerate(t[1], start=len(t[5]) - 1):
                constraints.append(example(t, position, token == TARGET))
        for prompt in POSITIVE_AUTHORING:
            t = trace(prompt)
            constraints.append(example(t, len(t[5]) - 1, True))
        for prompt in NEGATIVE_AUTHORING:
            t = trace(prompt)
            for position, token in enumerate(t[1], start=len(t[5]) - 1):
                constraints.append(example(t, position, token == TARGET))
        row, satisfied = fit(original, constraints)
        row_bytes = bytes(w & 255 for w in row)
        row_file = temporary / "contrastive.bin"
        row_file.write_bytes(TARGET.to_bytes(2, "little") + row_bytes)
        cases = ([("target", p) for p in TARGET_PROMPTS] + [("control", p) for p in CONTROL_PROMPTS]
                 + [("lm001-target", p) for p in LM001_TARGETS] + [("lm001-control", p) for p in LM001_CONTROLS])
        with ThreadPoolExecutor(8) as pool:
            base = list(pool.map(lambda c: trace(c[1])[1], cases))
            edited = list(pool.map(lambda c: trace(c[1], row_file)[1], cases))
    per_case = [{"group": g, "prompt": p, "first_token_id": e[0], "unchanged": e == b}
                for (g, p), b, e in zip(cases, base, edited)]
    def count(group, key):
        return sum(c["group"] == group and key(c) for c in per_case)
    summary = {
        "constraints": len(constraints), "constraints_satisfied": satisfied,
        "row_l1_change": sum(abs(a - b) for a, b in zip(row, original)),
        "extended_target_hits": count("target", lambda c: c["first_token_id"] == TARGET),
        "extended_naming_changed": sum(c["group"] == "control" and c["prompt"] in NAMING_CONTROLS and not c["unchanged"] for c in per_case),
        "extended_other_changed": sum(c["group"] == "control" and c["prompt"] not in NAMING_CONTROLS and not c["unchanged"] for c in per_case),
        "lm001_target_hits": count("lm001-target", lambda c: c["first_token_id"] == TARGET),
        "lm001_controls_unchanged": count("lm001-control", lambda c: c["unchanged"]),
    }
    summary["preregistered_success"] = (summary["extended_target_hits"] >= 18 and summary["extended_naming_changed"] <= 2
                                        and summary["extended_other_changed"] == 0)
    print(json.dumps(summary, indent=1))
    args.out.write_text(json.dumps({"summary": summary, "row": row, "original_row": original,
                                    "row_sha256": sha256(row_bytes).hexdigest(), "cases": per_case}, indent=1) + "\n")


if __name__ == "__main__":
    main()
