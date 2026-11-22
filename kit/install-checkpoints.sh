#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
dest=${PYNQ_CHECKPOINTS:-"$HOME/Library/Application Support/PYNQ Process/checkpoints"}
work="$dest/.work"
src=https://huggingface.co/google/flan-t5-small/resolve/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab
mkdir -p "$work"
if [ ! -x "$work/venv/bin/python" ]; then
  python3 -m venv "$work/venv"
  "$work/venv/bin/pip" install -q numpy==2.5.5 sentencepiece==0.2.3 safetensors==0.8.0
fi
py="$work/venv/bin/python"
[ -f "$work/source.safetensors" ] || curl -fL "$src/model.safetensors" -o "$work/source.safetensors"
[ -f "$work/spiece.model" ] || curl -fL "$src/spiece.model" -o "$work/spiece.model"
cd "$root"
[ -d "$work/int8" ] || "$py" -m lm_reference.export_flan "$work/source.safetensors" "$work/spiece.model" "$work/int8"
[ -d "$dest/diagnostic-install" ] || "$py" -m lm_reference.flan_delta apply "$work/int8" profiles/flan-telemetry-diagnostic "$dest/diagnostic-install"
[ -d "$dest/resident" ] || "$py" -m lm_reference.flan_delta apply "$work/int8" profiles/flan-telemetry "$dest/resident"
if [ ! -d "$dest/queue-history-v2" ]; then
  stage=$(mktemp -d "$work/queue-history.XXXXXX")
  asset=pynq-process-queue-history-v2-delta.tar.gz
  if gh release download queue-history-v2 --repo StefanIlas1/pinkProcess --pattern "$asset" --dir "$stage"; then
    printf '%s  %s\n' c9ec81ac1b81db1aa84a6ac774073cddd7e405b07089dba53e8492514962ec61 "$stage/$asset" | shasum -a 256 -c -
    tar -xzf "$stage/$asset" -C "$stage"
    "$py" -m lm_reference.flan_delta apply "$work/int8" "$stage" "$dest/queue-history-v2"
  else
    echo "Queue history v2 needs gh access to the release. Skipping it."
  fi
fi
shasum -a 256 "$dest"/*/model.json
