"""Factory provisioning for the resident telemetry kit. Mac app users only need the pairing."""

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from lm_reference.model_assets import FILES, file_digest, verify_model


def make_bundle(destination, overlay, *, language_model=None, manifest_sha256=None):
    if bool(language_model) != bool(manifest_sha256):
        raise ValueError("Supply both the language model and its trusted manifest SHA-256.")
    with tempfile.TemporaryDirectory() as temp:
        staging = Path(temp)
        for package in ("pinkprocess", "lm_reference", "fpga"):
            (staging / package).mkdir()
        for file in (ROOT / "pinkprocess").glob("*.py"):
            if not file.name.startswith("test_"):
                shutil.copyfile(file, staging / "pinkprocess" / file.name)
        for path in ("fpga/check_classifier.py", "fpga/mmio_words.c", *("lm_reference/" + name + ".py" for name in (
            "fabric", "flan", "flan_prompt", "live_demo", "model_assets", "patch_operator", "quantize",
            "reference", "weight_versions",
        ))):
            shutil.copyfile(ROOT / path, staging / path)
        (staging / "profiles").mkdir()
        for name in ("pool-logistic-v1", "service-b-astra-v1", "queue-outlook-v2"):
            path = Path("profiles") / (name + ".json")
            shutil.copyfile(ROOT / path, staging / path)
        if overlay is not None:
            (staging / "overlay").mkdir()
            for extension in (".bit", ".hwh"):
                shutil.copyfile(overlay.with_suffix(extension), staging / "overlay" / ("pinkprocess" + extension))
        if language_model is not None:
            bundled_model = staging / "language-model"
            bundled_model.mkdir()
            for name in FILES:
                shutil.copyfile(language_model / name, bundled_model / name)
            verify_model(bundled_model, manifest_sha256)
            (staging / "language-model.sha256").write_text(manifest_sha256 + "\n")
        shutil.copyfile(ROOT / "kit/install-board.sh", staging / "install-board.sh")
        files = sorted(p for p in staging.rglob("*") if p.is_file())
        (staging / "SHA256SUMS").write_text("".join(
            f"{file_digest(p)}  {p.relative_to(staging)}\n" for p in files))
        with tarfile.open(destination, "w:gz") as output:
            for file in sorted(staging.iterdir()):
                output.add(file, arcname=file.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=ROOT / "build/pynq/pinkprocess.bit")
    parser.add_argument("--sqlite-demo", action="store_true", help="Enable the board-local SQLite workload on install")
    parser.add_argument("--language-model", type=Path, help="Local directory containing a FLAN deployment export")
    parser.add_argument("--manifest-sha256", help="Trusted SHA-256 for the export's model.json")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--bundle", type=Path, help="Create a board installation archive only")
    action.add_argument("--update-bundle", type=Path, help="Create an update archive that reuses the installed overlay")
    action.add_argument("--install", metavar="USER@HOST", help="Install via administrative SSH")
    args = parser.parse_args()
    if bool(args.language_model) != bool(args.manifest_sha256):
        parser.error("Supply both --language-model and --manifest-sha256.")
    model = dict(language_model=args.language_model, manifest_sha256=args.manifest_sha256)
    if args.bundle or args.update_bundle:
        make_bundle(args.bundle or args.update_bundle, None if args.update_bundle else args.overlay, **model)
        return
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*@[A-Za-z0-9][A-Za-z0-9.\-]*", args.install):
        parser.error("Use an SSH username and a hostname or IPv4 address.")
    target = args.install
    host = target.split("@")[1]
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "kit.tar.gz"
        make_bundle(bundle, args.overlay, **model)
        remote = subprocess.check_output(["ssh", target, "mktemp -d /tmp/pinkprocess-kit.XXXXXXXX"], text=True).strip()
        if not re.fullmatch(r"/tmp/pinkprocess-kit\.[A-Za-z0-9]{8}", remote):
            raise RuntimeError("Board did not return a valid staging directory.")
        try:
            subprocess.run(["scp", str(bundle), f"{target}:{remote}/kit.tar.gz"], check=True)
            command = (f"cd {remote} && tar xzf kit.tar.gz && "
                       f"sudo bash install-board.sh {shlex.quote(host)} {remote}/pairing.json")
            if args.sqlite_demo:
                command += " --sqlite-demo"
            subprocess.run(["ssh", "-t", target, command], check=True)
            pairing = Path(tmp) / "pairing.json"
            subprocess.run(["scp", f"{target}:{remote}/pairing.json", str(pairing)], check=True)
            dest = Path.home() / "Library/Application Support/pinkProcess/kits"
            dest.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=dest, delete=False) as file:
                file.write(pairing.read_bytes())
            os.replace(file.name, dest / (host + ".json"))
            print(f"Provisioned {host} and paired this Mac.")
        finally:
            subprocess.run(["ssh", target, "rm -rf -- " + remote], check=False)


if __name__ == "__main__":
    main()
