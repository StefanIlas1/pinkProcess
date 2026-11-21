#!/bin/bash
# Run this trusted script with an update archive staged by the logged-in user.
set -euo pipefail
umask 077
if [[ $EUID != 0 || ($# != 2 && $# != 4) || ! $2 =~ ^[0-9a-f]{64}$ ]]; then
    echo "Usage: sudo bash update-board.sh archive archive-sha256 [model-dir manifest-sha256]" >&2
    exit 2
fi
archive=$1
expected=$2
shift 2
staging=$(mktemp -d /var/tmp/pinkprocess-update.XXXXXXXX)
trap 'rm -rf -- "$staging"' EXIT
cp -- "$archive" "$staging/runtime.tar.gz"
actual=$(sha256sum < "$staging/runtime.tar.gz" | cut -d ' ' -f 1)
if [[ $actual != "$expected" ]]; then
    echo "Update archive SHA-256 does not match the trusted hash." >&2
    exit 1
fi
mkdir "$staging/release"
tar --no-same-owner --no-same-permissions -xzf "$staging/runtime.tar.gz" -C "$staging/release"
bash "$staging/release/install-board.sh" --update "$@"
