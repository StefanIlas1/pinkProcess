#!/bin/bash
# Run as root in an extracted, trusted kit bundle on a PYNQ-Z2.
set -euo pipefail
umask 077
if [[ $EUID != 0 || $# -lt 2 || $# -gt 3 || (${3:-} && ${3:-} != --sqlite-demo) ]]; then
    echo "Usage: sudo bash install-board.sh board-host unprivileged-pairing-output [--sqlite-demo]" >&2
    exit 2
fi
cd "$(dirname "$0")"
sha256sum --quiet --check SHA256SUMS
root=/opt/pinkprocess-kit
identity=/etc/pinkprocess-kit
state=/var/lib/pinkprocess-kit
revision=$(sha256sum SHA256SUMS | cut -c1-20)
install -d "$root/releases" "$identity" "$state"
if [[ ! -d "$root/releases/$revision" ]]; then
    temporary=$(mktemp -d "$root/releases/.install.XXXXXXXX")
    cp -R pinkprocess profiles fpga lm_reference overlay "$temporary/"
    mv "$temporary" "$root/releases/$revision"
fi
if [[ ! -f "$identity/board.key" ]]; then
    openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -subj /CN=pinkProcess-kit \
        -addext basicConstraints=critical,CA:TRUE -addext keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign \
        -addext extendedKeyUsage=serverAuth -addext subjectKeyIdentifier=hash \
        -keyout "$identity/board.key" -out "$identity/board.crt" 2>/dev/null
fi
if [[ ! -s "$identity/token" ]]; then
    openssl rand -hex 32 > "$identity/token"
fi
# Preserve a provisioned kit's source choice across software updates.
if [[ ${3:-} == --sqlite-demo ]]; then
    echo 'PINKPROCESS_SOURCE_ARGS=--demo-source sqlite' > "$identity/source.conf"
fi
ln -sfn "releases/$revision" "$root/.current-new"
mv -Tf "$root/.current-new" "$root/current"
cat > /etc/systemd/system/pinkprocess-kit.service <<'UNIT'
[Unit]
Description=pinkProcess FPGA telemetry kit
After=network.target

[Service]
WorkingDirectory=/opt/pinkprocess-kit/current
EnvironmentFile=-/etc/pinkprocess-kit/source.conf
ExecStart=/bin/bash -lc 'exec /usr/local/share/pynq-venv/bin/python3 -E -m pinkprocess.kit_server --identity /etc/pinkprocess-kit --profiles profiles --bitstream overlay/pinkprocess.bit --state-file /var/lib/pinkprocess-kit/active.json $${PINKPROCESS_SOURCE_ARGS:-}'
Restart=on-failure
RestartSec=5
UMask=0077

[Install]
WantedBy=multi-user.target
UNIT
python3 - "$1" "$identity" "$2" <<'PY'
import json, os, sys
from pathlib import Path
host, identity, output = sys.argv[1:]
identity = Path(identity)
record = {"host": host, "certificate": (identity / "board.crt").read_text(),
          "token": (identity / "token").read_text().strip()}
with open(output, "x") as f:
    json.dump(record, f)
    os.fchmod(f.fileno(), 0o600)
    os.fchown(f.fileno(), int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))
PY
systemctl daemon-reload
systemctl enable pinkprocess-kit.service
systemctl restart pinkprocess-kit.service
