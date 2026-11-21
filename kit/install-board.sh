#!/bin/bash
# Run as root in an extracted, trusted kit bundle on a PYNQ-Z2.
set -euo pipefail
umask 077
usage() {
    echo "Usage: sudo bash install-board.sh board-host unprivileged-pairing-output [--sqlite-demo]" >&2
    echo "       sudo bash install-board.sh --update [trusted-model-dir manifest-sha256]" >&2
    exit 2
}
[[ $EUID == 0 ]] || usage
update=0
model_source=
model_hash=
if [[ ${1:-} == --update ]]; then
    [[ $# == 1 || $# == 3 ]] || usage
    update=1
    if [[ $# == 3 ]]; then
        model_source=$(realpath -- "$2")
        model_hash=$3
    fi
else
    [[ $# -ge 2 && $# -le 3 && (! ${3:-} || ${3:-} == --sqlite-demo) ]] || usage
    [[ ! -e $2 ]] || { echo "Pairing output already exists." >&2; exit 2; }
fi
cd "$(dirname "$0")"
sha256sum --quiet --check SHA256SUMS
root=/opt/pinkprocess-kit
identity=/etc/pinkprocess-kit
state=/var/lib/pinkprocess-kit
unit=/etc/systemd/system/pinkprocess-kit.service
install -d "$root/releases" "$identity" "$state"
exec 9> "$root/install.lock"
flock -n 9
previous=$(readlink "$root/current" || true)
if (( update )) && [[ ! -d $root/current || ! -s $identity/token || ! -s $unit ]]; then
    echo "A provisioned board is required for an incremental update." >&2
    exit 1
fi
overlay=overlay
if [[ ! -d $overlay ]]; then
    if (( ! update )); then
        echo "Factory provisioning requires an overlay." >&2
        exit 1
    fi
    overlay=$root/current/overlay
fi
if [[ -f language-model.sha256 ]]; then
    [[ -z $model_source ]] || { echo "Choose either bundled or separately staged model files." >&2; exit 2; }
    model_source=language-model
    model_hash=$(cat language-model.sha256)
fi
if [[ -n $model_source ]]; then
    python3 -m lm_reference.model_assets "$model_source" "$model_hash" "$root/models"
elif [[ -f $root/current/language.conf ]]; then
    model_hash=$(cut -d ' ' -f 4 "$root/current/language.conf")
fi
revision=$({ sha256sum < SHA256SUMS; sha256sum < "$overlay/pinkprocess.bit"; sha256sum < "$overlay/pinkprocess.hwh"; printf '%s\n' "$model_hash"; } | sha256sum | cut -c1-20)
if [[ ! -d "$root/releases/$revision" ]]; then
    temporary=$(mktemp -d "$root/releases/.install.XXXXXXXX")
    trap 'rm -rf -- "$temporary"' EXIT
    cp -R pinkprocess profiles fpga lm_reference "$overlay" "$temporary/"
    if command -v cc >/dev/null; then
        if ! cc -O2 -Wall -Wextra -shared -fPIC "$temporary/fpga/mmio_words.c" -o "$temporary/fpga/mmio_words.so"; then
            rm -f "$temporary/fpga/mmio_words.so"
            echo "Using scalar MMIO after the optional native helper failed to compile." >&2
        fi
    fi
    if [[ -n $model_hash ]]; then
        printf 'PINKPROCESS_LANGUAGE_ARGS=--language-model %s/models/%s --manifest-sha256 %s\n' \
            "$root" "$model_hash" "$model_hash" > "$temporary/language.conf"
    fi
    mv "$temporary" "$root/releases/$revision"
    trap - EXIT
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
if (( ! update )) && [[ ${3:-} == --sqlite-demo ]]; then
    echo 'PINKPROCESS_SOURCE_ARGS=--demo-source sqlite' > "$identity/source.conf"
fi
unit_backup=$(mktemp "$root/.unit.XXXXXXXX")
unit_existed=0
if [[ -f $unit ]]; then
    cp -p "$unit" "$unit_backup"
    unit_existed=1
fi
trap 'rm -f -- "$unit_backup" "$unit.new"' EXIT
cat > "$unit.new" <<'UNIT'
[Unit]
Description=pinkProcess FPGA telemetry kit
After=network.target

[Service]
WorkingDirectory=/opt/pinkprocess-kit/current
EnvironmentFile=-/etc/pinkprocess-kit/source.conf
EnvironmentFile=-/opt/pinkprocess-kit/current/language.conf
ExecStart=/bin/bash -lc 'exec /usr/local/share/pynq-venv/bin/python3 -E -m pinkprocess.kit_server --identity /etc/pinkprocess-kit --profiles profiles --bitstream overlay/pinkprocess.bit --state-file /var/lib/pinkprocess-kit/active.json $${PINKPROCESS_SOURCE_ARGS:-} $${PINKPROCESS_LANGUAGE_ARGS:-}'
Restart=on-failure
RestartSec=5
UMask=0077

[Install]
WantedBy=multi-user.target
UNIT
switch_release() {
    ln -sfn "$1" "$root/.current-new" && mv -Tf "$root/.current-new" "$root/current"
}

probe() {
    python3 - "$identity" "$model_hash" <<'PY'
import json, ssl, sys, time, urllib.request
from pathlib import Path

identity = Path(sys.argv[1])
context = ssl.create_default_context(cafile=identity / "board.crt")
context.check_hostname = False
token = (identity / "token").read_text().strip()
request = urllib.request.Request("https://127.0.0.1:8784/v1/query", data=b'{"op":"status"}', headers={
    "Authorization": "Bearer " + token, "Content-Type": "application/json",
})
client = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context))
deadline = time.monotonic() + 60
while True:
    try:
        with client.open(request, timeout=3) as response:
            status = json.load(response)
        if status.get("ok") and (not sys.argv[2] or status.get("lm_supported") is True):
            break
    except (OSError, ValueError):
        pass
    if time.monotonic() > deadline:
        sys.exit("The installed board service did not pass its HTTPS readiness check.")
    time.sleep(1)
PY
}

if ! (mv -f "$unit.new" "$unit" && switch_release "releases/$revision" &&
      systemctl daemon-reload && systemctl restart pinkprocess-kit.service && probe); then
    if [[ -n $previous ]]; then
        switch_release "$previous"
    else
        rm -f "$root/current"
    fi
    if (( unit_existed )); then
        cp -p "$unit_backup" "$unit"
    else
        rm -f "$unit"
    fi
    systemctl daemon-reload
    if (( unit_existed )) && [[ -n $previous ]]; then
        systemctl restart pinkprocess-kit.service
    else
        systemctl stop pinkprocess-kit.service || true
    fi
    exit 1
fi
systemctl enable pinkprocess-kit.service
printf 'Installed pinkProcess release %s\n' "$revision"
if (( update )); then
    exit 0
fi
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
