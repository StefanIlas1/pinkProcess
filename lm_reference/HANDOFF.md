# Reproduce the TinyStories PYNQ-Z2 demo (historical)

This is the reproduction kit for the earlier TinyStories research build, which came before the telemetry diagnosis model. It is kept so that work can be repeated. For the current product see the [README](../README.md).

This kit contains the integer TinyStories checkpoint, pinned C runtime and
tokenizer, 50 MHz PYNQ-Z2 overlay, and the scripts for the Mac and board
runs. Run the commands from the extracted `pinkprocess-pynq-repro`
directory. The suite writes fresh reports under `results/`.

## Query TinyStories on the Mac

The Mac query kit includes `Start pinkProcess.command`. Double-click it
after extracting the archive, then open `http://127.0.0.1:8765/` in
your browser. It runs the quantized model on your Mac using Python 3.10
or newer and a C compiler such as `cc` from Xcode Command Line Tools.

Enter a line to start a story and click **Generate story**. The default
prompt is the Lily, then Tim, then Lily demonstration. Apply the included patch
and run it again to see Tim, then roll back to restore Lily.

From this repository, start the same Mac browser app with these commands.

```sh
python3 -m lm_reference.operator_web --software \
  --source build/lm-inputs/run.c --model build/pynq/stories260K.ppq \
  --tokenizer build/lm-inputs/tok512.bin \
  --patch build/lm001/conventional.patch.json
```

## Mac reference and RTL simulation

Use Python 3.10 or later, a C compiler named `cc`, and Icarus Verilog.

```sh
shasum -a 256 -c SHA256SUMS
python3 -m lm_reference.reproduce \
  --source inputs/run.c --model inputs/stories260K.ppq \
  --tokenizer inputs/tok512.bin --out results/mac
```

The reference should replay 17,920 RTL scores and reproduce Lily, then Tim, then
Lily in the six-prompt live socket test. Check `results/mac/suite.json`
and `results/mac/live.json`.

## Physical PYNQ-Z2

Use the PYNQ-Z2 v3.1.1 image with its Python virtual environment and a C
compiler. Transfer the archive to the board. For a clean-boot run, reboot
the board before starting the suite. It will record the kernel boot ID.

```sh
sha256sum -c SHA256SUMS
sudo bash -lc 'exec /usr/local/share/pynq-venv/bin/python3 \
  -m lm_reference.reproduce --source inputs/run.c \
  --model inputs/stories260K.ppq --tokenizer inputs/tok512.bin \
  --bitstream overlay/pinkprocess.bit --out results/board'
```

The board suite checks classifier replay, live generation, rejected
patches, and the full classifier hash after rollback. Compare the phases
in `results/board/live.json` with the Mac report. Keep `suite.json`,
`live.json`, and `operator.jsonl` as the review record.

## Browser operator with the PYNQ-Z2

Run this on the board after the suite has produced its patch bundle.

```sh
sudo bash -lc 'exec /usr/local/share/pynq-venv/bin/python3 \
  -m lm_reference.operator_web --source inputs/run.c \
  --model inputs/stories260K.ppq --tokenizer inputs/tok512.bin \
  --bitstream overlay/pinkprocess.bit --patch results/board/tim.patch.json'
```

Forward port 8765 from your Mac with
`ssh -N -L 8765:127.0.0.1:8765 xilinx@BOARD_ADDRESS` and open
`http://127.0.0.1:8765/`. You can submit your own story prompt via the
Mac browser while generation runs on the PYNQ-Z2. For the default prompt,
generate with the base weights, preview and apply the patch, generate
again, roll back, verify the base matrix, and
generate once more. The story should lead with Lily, then Tim, then Lily.

## Input provenance

- `inputs/run.c` is the pinned [llama2.c runtime](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb).
- `inputs/tok512.bin` and the source for `inputs/stories260K.ppq` come from the pinned [TinyStories checkpoint](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da/stories260K). The supplied model is the fixed-point export.
- `overlay/pinkprocess.bit` and `overlay/pinkprocess.hwh` are the 50 MHz artifacts used for the verified PYNQ-Z2 run.
