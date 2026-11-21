"""Factory provisioning for the resident telemetry kit. Mac app users only need the pairing."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def make_bundle(destination, overlay):
    with tempfile.TemporaryDirectory() as temp:
        staging = Path(temp)
        for package in ("pinkprocess", "lm_reference", "fpga"):
            (staging / package).mkdir()
        for file in (ROOT / "pinkprocess").glob("*.py"):
            if not file.name.startswith("test_"):
                shutil.copyfile(file, staging / "pinkprocess" / file.name)
        for path in ("fpga/check_classifier.py", "lm_reference/live_demo.py", "lm_reference/reference.py"):
            shutil.copyfile(ROOT / path, staging / path)
        (staging / "profiles").mkdir()
        for name in ("pool-logistic-v1", "service-b-astra-v1", "queue-outlook-v2"):
            path = Path("profiles") / (name + ".json")
            shutil.copyfile(ROOT / path, staging / path)
        (staging / "overlay").mkdir()
        for extension in (".bit", ".hwh"):
            shutil.copyfile(overlay.with_suffix(extension), staging / "overlay" / ("pinkprocess" + extension))
        shutil.copyfile(ROOT / "kit/install-board.sh", staging / "install-board.sh")
        files = sorted(p for p in staging.rglob("*") if p.is_file())
        (staging / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(staging)}\n" for p in files))
        with tarfile.open(destination, "w:gz") as output:
            for file in sorted(staging.iterdir()):
                output.add(file, arcname=file.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=ROOT / "build/pynq/pinkprocess.bit")
    parser.add_argument("--sqlite-demo", action="store_true", help="Enable the board-local SQLite workload on install")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--bundle", type=Path, help="Create a board installation archive only")
    action.add_argument("--install", metavar="USER@HOST", help="Install via administrative SSH")
    args = parser.parse_args()
    if args.bundle:
        make_bundle(args.bundle, args.overlay)
        return
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*@[A-Za-z0-9][A-Za-z0-9.\-]*", args.install):
        parser.error("Use an SSH username and a hostname or IPv4 address.")
    target = args.install
    host = target.split("@")[1]
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "kit.tar.gz"
        make_bundle(bundle, args.overlay)
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
