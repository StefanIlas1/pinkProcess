"""Build and replay the TinyStories INT8 classifier reference on the Mac."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile

SOURCE_HASH = "9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd"
MODEL_HASH = "b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696"
QUANTIZED_MODEL_HASH = "8b9e81ba42f6906fd21014886757074f93db592eb66ca5e313c75228ed65cd48"
TOKENIZER_HASH = "037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312"
WIDTH = 64
ROWS = 512
EDIT_PROMPT = "Once upon a time, there was a little girl named"
FIT_PROMPTS = (
    "Once upon a time, there was a little boy named",
    "Lily and Tim went to the park",
)
HELD_OUT_PROMPTS = (
    "Once upon a time, there was a little girl who",
    "One day, a little dog went outside",
)
SHARED_PREFIX_PROMPT = "Once upon a time"


def read_pinned(path: Path, digest: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{path} does not match the pinned SHA-256.")
    return data


def run_with_classifier(command, env, score):
    host, worker = socket.socketpair()
    with host, worker, tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        host.settimeout(20)
        env["PINKPROCESS_SCORE_FD"] = str(worker.fileno())
        with subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                              pass_fds=(worker.fileno(),)) as process, host.makefile("rb") as activations:
            worker.close()
            try:
                while data := activations.read(WIDTH):
                    if len(data) != WIDTH:
                        raise RuntimeError("Incomplete activation received from the runtime.")
                    sums = score(struct.unpack(f"{WIDTH}b", data))
                    host.sendall(struct.pack(f"={ROWS}i", *sums))
                status = process.wait(timeout=10)
            except BaseException:
                process.kill()
                raise
        stderr.seek(0)
        diagnostics = stderr.read()
        if status:
            raise RuntimeError(f"Live runtime exited with status {status}: {diagnostics[-1024:].decode(errors='replace')}")
        stdout.seek(0)
        return stdout.read(), diagnostics


def run(executable: Path, args: list[str], mode: str, row_file: Path | None = None, *, score=None):
    env = {**os.environ, "PINKPROCESS_REFERENCE_MODE": mode}
    env.pop("PINKPROCESS_REFERENCE_ROW", None)
    env.pop("PINKPROCESS_SCORE_FD", None)
    if row_file is not None:
        env["PINKPROCESS_REFERENCE_ROW"] = str(row_file)
    command = [str(executable), *args]
    if score is None:
        result = subprocess.run(command, check=True, capture_output=True, env=env)
        stdout, stderr = result.stdout, result.stderr
    else:
        stdout, stderr = run_with_classifier(command, env, score)
    tokens, inputs, scores = [], {}, {}
    prompt_tokens = []
    weights = None
    for line in stderr.decode().splitlines():
        if not line.strip():
            continue
        kind, *fields = line.split()
        if kind == "token":
            tokens.append(int(fields[0]))
        elif kind == "input":
            position, scale, *values = fields
            inputs[int(position)] = (scale, list(map(int, values)))
        elif kind == "scores":
            position, *values = fields
            scores[int(position)] = list(map(int, values))
        elif kind == "weights":
            scale, *values = fields
            weights = (scale, list(map(int, values)))
        elif kind == "prompt":
            prompt_tokens = list(map(int, fields))
    return stdout, tokens, inputs, scores, weights, prompt_tokens


def with_prompt(cli, prompt):
    arguments = cli.copy()
    arguments[arguments.index("-i") + 1] = prompt
    return arguments


def row_constraints(target, before, preserved):
    def example(trace, position, target_wins):
        activation = trace[2][position][1]
        scores = trace[3][position]
        best_other = max(scores[:target] + scores[target + 1:])
        return activation, best_other, target_wins

    yield example(before, len(before[5]) - 1, True)
    for trace in preserved:
        for position, token in enumerate(trace[1], start=len(trace[5]) - 1):
            yield example(trace, position, token == target)


def fit_selective_row(original, target, before, preserved):
    constraints = []
    for activation, best_other, target_wins in row_constraints(target, before, preserved):
        sign = 1 if target_wins else -1
        constraints.append(([sign * value for value in activation], sign * best_other,
                            sum(value ** 2 for value in activation)))
    row = [float(weight) for weight in original]
    for _ in range(2000):
        for activation, bound, squared_norm in constraints:
            # Leave room for rounding the projected row back to INT8.
            deficit = bound + 256 - sum(w * x for w, x in zip(row, activation))
            if deficit > 0 and squared_norm:
                step = deficit / squared_norm
                row = [max(-127, min(127, w + step * x)) for w, x in zip(row, activation)]
        candidate = [round(weight) for weight in row]
        if all(sum(w * x for w, x in zip(candidate, activation)) > bound
               for activation, bound, _ in constraints):
            return bytes(weight & 255 for weight in candidate), len(constraints)
    raise RuntimeError("The selective row fit did not satisfy the integer score constraints.")


def rollback_trial(executable: Path, cli: list[str], row_file: Path, *, selective=False,
                   mode="int8") -> dict:
    stem = EDIT_PROMPT
    cli = with_prompt(cli, stem)
    before = run(executable, cli, mode)
    first_score_position = len(before[5]) - 1
    first_token = before[1][0]
    for name in ("Tim", "Lily"):
        continuation_prompt = f"{stem} {name}"
        continuation = run(executable, with_prompt(cli, continuation_prompt), mode)
        extended_ids = continuation[5]
        if extended_ids[:-1] == before[5] and extended_ids[-1] != first_token:
            target = extended_ids[-1]
            break
    else:
        raise RuntimeError("No alternate one-token name for this prompt.")
    weight_scale, originals = before[4]
    start = target * WIDTH
    stop = start + WIDTH
    matrix = bytearray(value & 255 for value in originals)
    original_matrix = bytes(matrix)
    original_row = bytes(matrix[start:stop])
    activation = before[2][first_score_position][1]
    if selective:
        fit_runs = [run(executable, with_prompt(cli, prompt), mode) for prompt in FIT_PROMPTS]
        patched_row, constraint_count = fit_selective_row(
            originals[start:stop], target, before, [continuation, *fit_runs])
    else:
        # Deliberately maximize the target score on this activation for the smoke test.
        patched_row = bytes((127 if value > 0 else -127 if value < 0 else 0) & 255
                            for value in activation)
    matrix[start:stop] = patched_row
    row_file.write_bytes(target.to_bytes(2, "little") + patched_row)
    patched = run(executable, cli, mode, row_file)
    observed_scale, observed_weights = patched[4]
    if observed_scale != weight_scale or bytes(value & 255 for value in observed_weights) != matrix:
        raise RuntimeError("The runtime applied a different matrix from the one requested.")
    if patched[2][first_score_position] != before[2][first_score_position]:
        raise RuntimeError("The output-only edit changed the activation before decoding.")
    if patched[1][0] != target:
        raise RuntimeError("The edited row did not change the first sampled token to the target.")

    selectivity = None
    if selective:
        if patched[1][1:] != continuation[1] or patched[0] != continuation[0]:
            raise RuntimeError("The edit changed the unmodified model's continuation after the target name.")
        controls = []
        control_prompts = (*FIT_PROMPTS, *HELD_OUT_PROMPTS, SHARED_PREFIX_PROMPT)
        for index, prompt in enumerate(control_prompts):
            arguments = with_prompt(cli, prompt)
            control = fit_runs[index] if index < len(fit_runs) else run(executable, arguments, mode)
            edited_control = run(executable, arguments, mode, row_file)
            unchanged = control[:2] == edited_control[:2]
            if prompt == SHARED_PREFIX_PROMPT:
                offset = first_score_position - len(control[5]) + 1
                if (offset < 0 or control[2][first_score_position] != before[2][first_score_position]
                        or control[1][:offset] != edited_control[1][:offset]
                        or control[1][offset:offset + 1] != [first_token]
                        or edited_control[1][offset:offset + 1] != [target]):
                    raise RuntimeError("The shared-prefix prompt did not reproduce the target edit.")
                group = "shared_prefix"
            else:
                if not unchanged:
                    raise RuntimeError(f"The selective edit changed the control prompt {prompt!r}.")
                group = "fit" if index < len(fit_runs) else "held_out"
            controls.append({
                "prompt": prompt,
                "group": group,
                "unchanged": unchanged,
                "original_token_ids": control[1],
                "edited_token_ids": edited_control[1],
                "original_text": control[0].decode().strip(),
                "edited_text": edited_control[0].decode().strip(),
            })
        selectivity = {
            "fit_constraints": constraint_count,
            "continuation_prompt": continuation_prompt,
            "continuation_tokens_preserved": len(continuation[1]),
            "original_best_other_sum_at_first_sample": max(before[3][first_score_position]),
            "controls": controls,
        }

    matrix[start:stop] = original_row
    row_file.write_bytes(target.to_bytes(2, "little") + bytes(matrix[start:stop]))
    restored = run(executable, cli, mode, row_file)
    if matrix != original_matrix or restored != before:
        raise RuntimeError("Rollback did not reproduce the original weights and execution.")

    trial = {
        "prompt": stem,
        "prompt_token_ids": before[5],
        "target_name": name,
        "target_row": target,
        "original_row_hex": original_row.hex(),
        "patched_row_hex": patched_row.hex(),
        "original_row_sha256": hashlib.sha256(original_row).hexdigest(),
        "patched_row_sha256": hashlib.sha256(patched_row).hexdigest(),
        "restored_row_sha256": hashlib.sha256(matrix[start:stop]).hexdigest(),
        "original_matrix_sha256": hashlib.sha256(original_matrix).hexdigest(),
        "patched_matrix_sha256": hashlib.sha256(bytes(value & 255 for value in observed_weights)).hexdigest(),
        "restored_matrix_sha256": hashlib.sha256(matrix).hexdigest(),
        "first_token_ids": [first_token, patched[1][0], restored[1][0]],
        "target_sums_at_first_sample": [stage[3][first_score_position][target]
                                        for stage in (before, patched, restored)],
        "original_token_ids": before[1],
        "patched_token_ids": patched[1],
        "restored_token_ids": restored[1],
        "generated_text": [stage[0].decode().strip() for stage in (before, patched, restored)],
    }
    if selectivity is not None:
        trial["selectivity"] = selectivity
    return trial


def instrument_source(path: Path, *, separate: bool, fixed=False) -> str:
    source = read_pinned(path, SOURCE_HASH).decode()
    classifier = Path(__file__).with_name("classifier.h").read_text()
    if fixed:
        classifier += "\n" + Path(__file__).with_name("quantized.h").read_text()
    source = source.replace("float* forward(", classifier + "\nfloat* forward(", 1)
    trace = 'next = sample(sampler, logits);\n            fprintf(stderr, "token %d\\n", next);'
    if fixed:
        trace = trace.replace("sample(sampler, logits)",
                              "pp_enabled ? pp_argmax(fixed_scores) : sample(sampler, logits)")
        source = source.replace(
            "float* logits = forward(transformer, token, pos);",
            "const int32_t *fixed_scores = pp_enabled ? pp_forward(token, pos) : NULL;\n"
            "        float *logits = pp_enabled ? NULL : forward(transformer, token, pos);", 1)
        source = source.replace("build_transformer(&transformer, checkpoint_path);", """
    const char *reference_mode = getenv("PINKPROCESS_REFERENCE_MODE");
    pp_enabled = reference_mode && strcmp(reference_mode, "fixed") == 0;
    if (pp_enabled) {
        if (temperature != 0.0f || strcmp(mode, "generate") != 0) {
            fprintf(stderr, "Quantized inference requires greedy generation.\\n");
            exit(EXIT_FAILURE);
        }
        pp_load(&transformer.config, checkpoint_path);
    } else {
        build_transformer(&transformer, checkpoint_path);
    }""", 1)
        source = source.replace("free_transformer(&transformer);",
                                "if (!pp_enabled) free_transformer(&transformer);", 1)
    source = source.replace("next = sample(sampler, logits);", trace, 1)
    encode = "encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);"
    source = source.replace(encode, encode + r'''
    fprintf(stderr, "prompt");
    for (int i = 0; i < num_prompt_tokens; i++) {
        fprintf(stderr, " %d", prompt_tokens[i]);
    }
    fputc('\n', stderr);
''', 1)
    if separate:
        call = "matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);"
        source = source.replace(call, "pinkprocess_classifier(s->logits, x, w->wcls, p->dim, p->vocab_size, pos);", 1)
    return source


def compile_runtime(source: Path, executable: Path, *, separate=True, compiler="clang", fixed=False):
    c_file = executable.with_suffix(".c")
    c_file.write_text(instrument_source(source, separate=separate, fixed=fixed))
    subprocess.run([compiler, "-O2", "-ffp-contract=off", str(c_file),
                    "-o", str(executable), "-lm"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned llama2.c run.c")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixed", action="store_true", help="Use the exported integer checkpoint as --model")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--cc", default="cc", help="C compiler")
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--edit-demo", action="store_true", help="Record a one-row edit and rollback")
    parser.add_argument("--selective", action="store_true", help="Fit and validate a selective one-row edit")
    args = parser.parse_args()
    read_pinned(args.model, QUANTIZED_MODEL_HASH if args.fixed else MODEL_HASH)
    read_pinned(args.tokenizer, TOKENIZER_HASH)

    prompt = "Once upon a time"
    cli = [str(args.model), "-z", str(args.tokenizer), "-i", prompt, "-t", "0", "-n", "32", "-s", "42"]
    with tempfile.TemporaryDirectory() as directory:
        build = Path(directory)
        executable = build / "separate"
        compile_runtime(args.source, executable, compiler=args.cc, fixed=args.fixed)
        if not args.fixed:
            original = build / "original"
            compile_runtime(args.source, original, separate=False, compiler=args.cc)
            float_reference = run(original, cli, "float")
            if float_reference != run(executable, cli, "float"):
                raise RuntimeError("Separate float classifier changed the baseline.")
            print(f"Separate float classifier matched the original across {len(float_reference[1])} sampled tokens.")

        mode = "fixed" if args.fixed else "int8"
        integer = run(executable, cli, mode)
        if integer != run(executable, cli, mode):
            raise RuntimeError("INT8 replay differed across fresh processes.")
        trial = (rollback_trial(executable, cli, build / "row.bin", selective=args.selective, mode=mode)
                 if args.edit_demo or args.selective else None)

    generated, tokens, inputs, scores, encoded_weights, _ = integer
    weight_scale, flat_weights = encoded_weights
    if len(flat_weights) != ROWS * WIDTH or set(inputs) != set(scores) or not inputs:
        raise RuntimeError("The classifier trace has unexpected shapes.")
    weights = [flat_weights[start:start + WIDTH] for start in range(0, len(flat_weights), WIDTH)]
    vectors = []
    for position, (input_scale, activation) in inputs.items():
        if len(activation) != WIDTH:
            raise RuntimeError(f"Unexpected activation width at position {position}.")
        expected = [sum(w * x for w, x in zip(row, activation)) for row in weights]
        if expected != scores[position]:
            raise RuntimeError(f"Python and C disagree at position {position}.")
        scale_key = "input_scale" if args.fixed else "input_scale_hex"
        vectors.append({"position": position, scale_key: input_scale, "input": activation, "scores": expected})
    artifact = {
        "model_sha256": MODEL_HASH,
        "tokenizer_sha256": TOKENIZER_HASH,
        "runtime_source_sha256": SOURCE_HASH,
        "prompt": prompt,
        "decoding": {"temperature": 0, "seed": 42, "steps_including_prompt": 32},
        "quantization": {"encoding": "signed_int8", "range": [-127, 127],
                         "rounding": "nearest_ties_away_from_zero", "accumulator_bits": 32},
        "int8_token_ids": tokens,
        "weights_sha256": hashlib.sha256(bytes(value & 255 for value in flat_weights)).hexdigest(),
        "weights": weights,
        "vectors": vectors,
    }
    if args.fixed:
        artifact["quantized_model_sha256"] = QUANTIZED_MODEL_HASH
    else:
        artifact["float_token_ids"] = float_reference[1]
        artifact["weight_scale_hex"] = weight_scale
    if trial is not None:
        artifact["rollback_trial"] = trial
        print(f"Target {trial['target_name']} uses row {trial['target_row']}.")
        print(f"First sampled tokens before, edited, restored: {trial['first_token_ids']}")
        print("Rollback restored the complete classifier matrix and original token sequence.")
    args.vectors.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    print(f"INT8 replay matched across {len(tokens)} sampled tokens and {len(vectors)} forward passes.")
    print(f"Independent Python calculation matched all {len(vectors) * ROWS:,} integer scores.")
    print(generated.decode().strip())
    print(f"Saved {args.vectors}")


if __name__ == "__main__":
    main()
