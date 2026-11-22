"""Extended held-out evaluation and edit-strength sweep for the frozen LM001 rows (software reference)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from lm_reference.experiment import TARGET, NEW_TOKENS
from lm_reference.patch_operator import RowPatch
from lm_reference.reference import (
    EDIT_PROMPT, QUANTIZED_MODEL_HASH, TOKENIZER_HASH, WIDTH,
    compile_runtime, read_pinned, run, with_prompt,
)

TARGET_PROMPTS = (
    "Once upon a time, there was a happy girl named",
    "Once upon a time, there was a kind little girl named",
    "Once upon a time, there was a sweet girl named",
    "Once upon a time, there was a tiny girl named",
    "Once upon a time, there was a pretty girl named",
    "Once upon a time, there was a curious little girl named",
    "Once upon a time, there was a three year old girl named",
    "Once upon a time, in a small town, there was a girl named",
    "Once upon a time, in a big house, lived a little girl named",
    "One sunny day, a little girl named",
    "One morning, a little girl named",
    "Long ago, there was a little girl named",
    "In a small village, there was a little girl named",
    "There was a little girl named",
    "There once lived a girl named",
    "Once there was a girl named",
    "Once upon a time, there was a little girl. Her name was",
    "Once upon a time, there was a girl who loved flowers. She was named",
    "A little girl named",
    "Once upon a time, a happy little girl named",
    "Once upon a time, there was a little girl called",
    "One day, a girl named",
    "Every day, a little girl named",
    "Once upon a time, there was a smart girl named",
)
CONTROL_PROMPTS = (
    "Once upon a time, there was a big dog named",
    "Once upon a time, there was a little cat named",
    "Once upon a time, there was a small bird named",
    "Once upon a time, there was a fluffy bunny named",
    "Once upon a time, there was a little bear named",
    "Once upon a time, there was a big red truck named",
    "Once upon a time, there was a little fish named",
    "Once upon a time, there was a mom named",
    "Once upon a time, there was an old man named",
    "Once upon a time, there was a little boy named",
    "One day, a boy named",
    "Once upon a time, there was a tree",
    "The sun was shining and the birds were singing",
    "Lily and her mom went to the park",
    "The little dog wanted to play with the ball",
    "One day, Lily found a shiny rock",
    "Once upon a time, there was a big castle",
    "Mom said it was time to go to bed",
    "The cat sat on the mat and looked at the",
    "Ben had a red ball. He liked to",
    "It was a cold day and the snow was falling",
    "Sue wanted to bake a cake for her",
    "Once upon a time, a little frog lived in a pond",
    "The children played in the garden all day",
)
ALPHAS = tuple(round(0.1 * i, 1) for i in range(0, 21))


def digest(data):
    return sha256(data).hexdigest()


def scaled_row(original, edited, alpha):
    return bytes(max(-127, min(127, round(o + alpha * (e - o)))) & 255 for o, e in zip(original, edited))


def signed(payload):
    return [b - 256 if b > 127 else b for b in payload]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("source", "model", "tokenizer", "lm001", "out"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    args = parser.parse_args()
    read_pinned(args.model, QUANTIZED_MODEL_HASH)
    read_pinned(args.tokenizer, TOKENIZER_HASH)
    protocol = {"targets": TARGET_PROMPTS, "controls": CONTROL_PROMPTS, "alphas": ALPHAS,
                "new_tokens": NEW_TOKENS, "target_token_id": TARGET}
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode()
    frozen = {"extended_protocol_sha256": digest(protocol_bytes),
              "lm001_frozen_candidates": json.loads((args.lm001 / "frozen-candidates.json").read_text())}
    cli = [str(args.model), "-z", str(args.tokenizer), "-i", EDIT_PROMPT, "-t", "0", "-n", "32", "-s", "42"]
    conventional = RowPatch.parse(json.loads((args.lm001 / "conventional.patch.json").read_text())).payload
    astra = bytes(v & 255 for v in json.loads((args.lm001 / "astra-response.json").read_text())["row"])
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        executable = temporary / "tinystories"
        compile_runtime(args.source, executable, fixed=True)
        base_trace = run(executable, cli, "fixed")
        weights = base_trace[4][1]
        original = bytes(w & 255 for w in weights[TARGET * WIDTH:(TARGET + 1) * WIDTH])
        cases = [("target", p) for p in TARGET_PROMPTS] + [("control", p) for p in CONTROL_PROMPTS]
        steps = {}
        for _, prompt in cases:
            prompt_args = with_prompt(cli, prompt)
            prompt_args[prompt_args.index("-n") + 1] = "1"
            steps[prompt] = len(run(executable, prompt_args, "fixed")[5]) - 1 + NEW_TOKENS

        def trace_for(prompt, row_file):
            prompt_args = with_prompt(cli, prompt)
            prompt_args[prompt_args.index("-n") + 1] = str(steps[prompt])
            trace = run(executable, prompt_args, "fixed", row_file)
            pos = len(trace[5]) - 1
            scores = trace[3][pos]
            rank = 1 + sum(s > scores[TARGET] or (i < TARGET and s == scores[TARGET]) for i, s in enumerate(scores))
            return {"tokens": trace[1], "rank": rank, "text": trace[0].decode().strip()}

        jobs = []
        for author, edited in (("conventional", conventional), ("astra", astra)):
            for alpha in ALPHAS:
                row = scaled_row(signed(original), signed(edited), alpha)
                row_file = temporary / f"{author}-{alpha}.bin"
                row_file.write_bytes(TARGET.to_bytes(2, "little") + row)
                jobs.append((author, alpha, row_file, digest(row)))
        with ThreadPoolExecutor(8) as pool:
            baseline = dict(zip([p for _, p in cases], pool.map(lambda c: trace_for(c[1], None), cases)))
            results = []
            for author, alpha, row_file, row_hash in jobs:
                traces = list(pool.map(lambda c: trace_for(c[1], row_file), cases))
                per_case = []
                for (group, prompt), trace in zip(cases, traces):
                    base = baseline[prompt]
                    per_case.append({"group": group, "prompt": prompt, "first_token_id": trace["tokens"][0],
                                     "target_rank": trace["rank"], "unchanged": trace["tokens"] == base["tokens"],
                                     "baseline_first_token_id": base["tokens"][0], "text": trace["text"]})
                hits = sum(c["group"] == "target" and c["first_token_id"] == TARGET for c in per_case)
                top2 = sum(c["group"] == "target" and c["target_rank"] <= 2 for c in per_case)
                unchanged = sum(c["group"] == "control" and c["unchanged"] for c in per_case)
                intrusions = sum(c["group"] == "control" and TARGET in [c["first_token_id"]]
                                 and c["baseline_first_token_id"] != TARGET for c in per_case)
                results.append({"author": author, "alpha": alpha, "row_sha256": row_hash,
                                "target_hits": hits, "target_top2": top2, "controls_unchanged": unchanged,
                                "control_tim_intrusions": intrusions, "cases": per_case})
                print(f"{author} alpha={alpha}: hits {hits}/{len(TARGET_PROMPTS)} top2 {top2} "
                      f"unchanged {unchanged}/{len(CONTROL_PROMPTS)} intrusions {intrusions}", flush=True)
    original_hits = sum(baseline[p]["tokens"][0] == TARGET for p in TARGET_PROMPTS)
    report = {"frozen": frozen, "protocol": protocol, "original_target_hits": original_hits,
              "baseline": {p: {"first_token_id": t["tokens"][0], "text": t["text"]} for p, t in baseline.items()},
              "results": results}
    args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
