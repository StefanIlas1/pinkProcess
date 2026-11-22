#!/bin/sh
# Verify the shipped reference vectors before synthesizing the kit's RTL.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
case "$#" in
  0) out="$root/build/vivado-pynq" ;;
  1) out=$1 ;;
  *) echo "Usage: $0 [new-build-directory | --check-only]" >&2; exit 2 ;;
esac
python3 "$root/fpga/check_classifier.py" "$root/build/pynq/fixed-reference-vectors.json"
python3 "$root/fpga/check_projection.py"
if [ "$out" = "--check-only" ]; then exit 0; fi
if ! command -v vivado >/dev/null 2>&1; then
  echo "Source the Vivado settings script, then rerun this command." >&2
  exit 1
fi
mkdir -p "$(dirname -- "$out")"
mkdir "$out"
cd "$out"
vivado -mode batch -source "$root/fpga/build_pynq.tcl" -tclargs "$PWD"
test -s pinkprocess.bit && test -s pinkprocess.hwh || {
  echo "Vivado did not produce both pinkprocess.bit and pinkprocess.hwh." >&2
  exit 1
}
printf 'Built %s/pinkprocess.bit and %s/pinkprocess.hwh\n' "$PWD" "$PWD"
