"""Ask the Mac user's Astra runtime for a reproducible signed-weight edit."""
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


def materialize(plan, original, activation):
    mode = plan.get("direction")
    step = plan.get("strength")
    if mode not in ("increase", "decrease") or type(step) is not int or not 1 <= step <= 127:
        raise ValueError("Astra returned an invalid output-row adjustment.")
    if len(original) != len(activation) or not any(activation):
        raise ValueError("Generate a board answer to capture a calibration activation first.")
    sign = 1 if mode == "increase" else -1
    row = [max(-127, min(127, old + sign * step * ((value > 0) - (value < 0))))
           for old, value in zip(original, activation)]
    if row == original:
        raise ValueError("The requested change produced identical weights.")
    return row


def propose(instruction, context):
    if not isinstance(instruction, str) or not 0 < len(instruction.strip()) <= 1000:
        raise ValueError("Describe the intended editing behavior in 1–1000 characters.")
    activation = context.get("activation")
    if not activation:
        raise ValueError("Ask the board a calibration question before authoring a weight version.")
    cli = shutil.which("codex")
    if not cli:
        candidates = [Path.home() / ".local/bin/codex", Path("/opt/homebrew/bin/codex")]
        cli = next((str(p) for p in candidates if p.is_file()), None)
    if not cli:
        raise ValueError("Install and sign in to the Codex CLI on this Mac to author with Astra.")
    request = dict(instruction=instruction, target_token=context["token"], target_row=context["target_row"])
    if context.get("max_score") is not None and context.get("row_scale"):
        threshold = context["max_score"] / context["row_scale"]
        for step in range(1, 128):
            candidate = [max(-127, min(127, old + step * ((a > 0) - (a < 0))))
                         for old, a in zip(context["row"], activation)]
            if sum(a * w for a, w in zip(activation, candidate)) > threshold:
                request["smallest_verified_first_token_increase"] = step
                break
    prompt = ("Design one quantized output-row edit for the first decoder position of the last "
              "board calibration question. Choose increase or decrease for this token and an INT8 "
              "step size. The board applies that signed step along the calibration activation to "
              "the original row, clamping to signed 127. Prefer the smallest verified increase supplied "
              "to select a first token; larger steps can cause repetition. Return decline if that single row cannot "
              "reasonably implement the instruction. No tools are needed. Request: " + json.dumps(request))
    schema = dict(type="object", additionalProperties=False, required=["direction", "strength", "reason"],
                  properties={"direction": {"type": "string", "enum": ["increase", "decrease", "decline"]},
                              "strength": {"type": "integer"}, "reason": {"type": "string"}})
    with TemporaryDirectory(prefix="pinkprocess-astra-") as tmp:
        path = Path(tmp)
        (path / "schema.json").write_text(json.dumps(schema))
        try:
            result = subprocess.run([cli, "exec", "--ephemeral", "--skip-git-repo-check",
                                     "--sandbox", "read-only", "--model", "gpt-6.2-astra",
                                     "--output-schema", str(path / "schema.json"),
                                     "-o", str(path / "plan.json"), "-"],
                                    cwd=tmp, input=prompt, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Astra timed out while authoring the edit.") from exc
        if result.returncode or not (path / "plan.json").is_file():
            raise RuntimeError("Astra did not return an edit. Check Codex CLI sign-in and Astra model access.")
        plan = json.loads((path / "plan.json").read_text())
    if plan["direction"] == "decline":
        raise ValueError("Astra declined this single-row edit: " + plan["reason"])
    return dict(row=materialize(plan, context["row"], activation),
                target_row=context["target_row"], plan=plan)
