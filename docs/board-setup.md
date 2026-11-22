# Board setup and operation

How to provision a PYNQ-Z2 kit, pair a Mac with it, feed it live data and change its language-model weights. The product overview is in the [README](../README.md). Measured results are in [results.md](results.md). Overlay and checkpoint reproduction is in [engineering.md](engineering.md).

## What a kit is

A provisioned kit is a PYNQ-Z2 running `pinkprocess-kit.service`. The service holds the paired transport (HTTPS with a bearer token), the rolling telemetry history, the quantized FLAN-T5-small checkpoint and the FPGA overlay. The ARM processor coordinates generation and handles attention mixing, normalization and token selection. The FPGA performs the learned INT8 linear projections.

## Provision a kit

For a fresh diagnostic kit, first [reconstruct the diagnostic checkpoint](../profiles/flan-telemetry-diagnostic/README.md). Then provision the board with the matched, timing-qualified overlay and the pinned model:

```sh
python3 kit/provision.py --install USER@HOST \
  --overlay build/projection-queue-qualified/pinkprocess.bit \
  --language-model build/flan/diagnostic-install \
  --manifest-sha256 14d7912c66a78edf407bbb272f7aa04b953af920827a9a035275194ec4f2f2f2
```

Follow the [projection configuration](engineering.md#benchmark-the-flan-projection-accelerator) to enable all learned projections on the fabric. Provisioning installs the boot service and pairs the provisioning Mac. Add `--sqlite-demo` to enable generated board-local traffic, which the app labels **Simulated**.

`kit/install-board.sh` and `kit/update-board.sh` package and install the board-side code. A working local import alone does not show that the board has every dependency, so test on the board.

## Pair another Mac

Transfer the kit's pairing JSON privately. In the app choose **PYNQ Process, then Import Kit Pairing…**. The file grants access to query the kit and change its active profile, so treat it like a password.

## Connect

- **Ethernet.** Put the Mac and the board on a reachable network. In the app press **Find boards**. One kit connects on its own. Otherwise use **Connect by address**. If the Mac is on Wi-Fi and the board is on wall Ethernet you need a network route between them.
- **USB relay.** On the tested setup the Mac reached the paired board through the USB UART. Leave the board's USB console logged in at a shell prompt, close other serial clients and run `python3 kit/usb_relay.py /dev/cu.usbserial-DEVICE` on the Mac. Leave the relay running and press **Find boards**. The app checks the paired board's TLS certificate through the relay at `127.0.0.1`.
- **USB appliance.** An administrator can provision a relay that starts at boot, so a paired Mac reconnects to a power-cycled board over the Digilent FTDI UART ending in `_tul1`. It takes over the board's serial console. Read [usb-appliance.md](../kit/usb-appliance.md) for setup and console recovery before enabling it.

## Feed it live data

From a producer with network access to the board, pipe your pool monitor's newline-flushed CSV into the publisher. Give it the real pool capacity, the board hostname or address and the private pairing file:

```sh
python3 -m kit.publish_live --host BOARD --pairing /path/to/pairing.json \
  --source 'Service connection pool' --capacity 16
```

Send the header `time_ms,busy,waiting` followed by a reading every 100 ms. An authenticated live source takes precedence over the demo traffic. The board keeps the most recent readings, builds fixed-point features and submits each eligible window to the FPGA as it arrives, so a Mac can disconnect and still find the latest completed answer. The data path and acceptance evidence are in the [context tree](../kit/live-context.md).

## Upload a file to the board

With a board connected, a queue CSV is uploaded to it and analyzed on the FPGA. Press **Import data**, choose the file and enter the pool capacity when asked. The file needs the integer columns `time_ms,busy,waiting` in that order, 50 to 2,097 consecutive samples at 100 ms spacing, and at most 100 KiB. To try it, use the [synthetic recovery CSV](../kit/examples/pool-recovery.csv) with capacity **4**. The sample has 60 readings: a queue reaches 12 and drains to zero before the last reading.

The board stores the upload separately from its live feed and answers from one generation: **Short** is the first complete sentence, **Summary** excerpts the finished answer and **Final** keeps the model's wording. A second question on the same dataset is answered as a follow-up that sees the previous exchange. Other files are profiled and answered on this Mac while the board stays connected.

## Change the language-model weights

Open the **Skills** tab for the board's versions and the **Train** tab to make a new one.

- **Create a version.** In **Train**, name the version, give one output token such as `Now` (or a token ID) and describe the intended first-token behavior. **Train and save** asks the signed-in Mac Codex CLI to design a row edit. **From file** saves an Astra response you already have. The board stores the exact edited bytes against the original checkpoint.
- **Switch.** Pick a version under **Answer skill**. Each load verifies the classifier digest and probes an output row through FPGA scores. **Reset** returns to the original and keeps your named versions.
- **Synthesize for a dataset.** With a file uploaded to the board, **Synthesize model** picks a starting point, a name and a plain-language request. The board calibrates on the upload if needed, Astra proposes the INT8 row and the board saves it under your name for that dataset.
- **Replay.** The **Replay** tab lists saved incidents. **Replay** runs a new generation on the same captured evidence so you can compare answers before and after a change.

An output-row edit affects that token at every generation step. In the physical test, raising "Now" also made it recur at sentence boundaries and in the ending. See the [before-and-after answers](../kit/evidence/physical-diagnosis-edit.json).

## Older kits without a resident language model

Install Ollama on the Mac and download the local Qwen3 4B Instruct 2507 model once:

```sh
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

The app contacts Ollama on loopback. On a Homebrew installation it starts the installed runtime on demand. The model stays on the Mac, and board readings and open-ended questions are processed locally.

## Physical board check

On one kit the 201-reading sample contained 103 eligible forecast windows. The base profile warned on 103 and the alternate profile warned on nine. All 309 scores from the base, alternate and restored runs matched the Mac fixed-point reference, and the restored board weight hash matched the initial hash. These counts measure how the profiles differ, not forecast quality. The recorded responses are in `build/pynq/telemetry-kit-physical.json`.
