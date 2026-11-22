<p align="center"><img src="docs/logo-dark.png" alt="PYNQ Process" width="720"></p>

# PYNQ Process

PYNQ Process is an Apple Silicon Mac app for asking questions about live data. You import a file or connect to a paired PYNQ-Z2 board, the app profiles the data and recommends a small quantized model for it, and you ask in plain words. On the board, a quantized FLAN-T5-small model answers: the FPGA performs the learned INT8 matrix projections and the ARM processor coordinates generation. The first domain is connection-pool telemetry. The repo is still named pinkProcess.

[Download the Mac app](https://github.com/StefanIlas1/pinkProcess/raw/refs/heads/main/build/mac-kit/pinkProcess-Mac-arm64.zip) · [Board setup](docs/board-setup.md) · [Measured results](docs/results.md) · [Engineering reference](docs/engineering.md) · [Research log](research.md)

## What is real and what is simulated

Real:

- Answers about a queue CSV or the live feed are generated on the PYNQ-Z2. The kit keeps its own rolling history and scores eligible windows as they arrive, with the Mac closed.
- The original and Queue history v2 are selectable language-model checkpoints. Named edits to either checkpoint change one INT8 output row. The board stores, verifies, switches and replays versions.
- Board uploads, staged answers (Short, Summary, Final), follow-up questions and model synthesis run through the board. Dictation uses on-device speech recognition on the Mac.
- History, notes, export, the model library and the Claude connector run locally on the Mac.

Not the same as it looks:

- The board's built-in SQLite workload is generated traffic. The app labels it **Simulated**.
- A file you import is always profiled on this Mac, and the Transcript and Table answers come from plain statistics computed here. Only queue CSVs can be sent to the board for the model answers.
- The model recommendation is a rule-based guess from column names and types.
- The marketplace is local. Online sharing does not exist.
- With Queue history v2 active, four queries through the paired Mac backend took 8.40–10.45 seconds to first text and 14.30–16.16 seconds for the complete diagnosis. Live data continued arriving on the board. [Measured results](docs/results.md).

## Quick start

Download and extract the [arm64 Mac app](https://github.com/StefanIlas1/pinkProcess/raw/refs/heads/main/build/mac-kit/pinkProcess-Mac-arm64.zip), open it and press **Find boards**. To provision a kit, follow [board setup](docs/board-setup.md).

Build the app yourself with an interpreter that has PyInstaller installed:

```sh
PYTHON=/path/to/python3 mac/build_kit.sh
```

The output is `build/mac-kit/pinkProcess.app`, built for your Mac and signed locally. To work on the interface, run the backend and open the printed address in a browser:

```sh
PYTHONPATH=$PWD python3 mac/kit_backend.py
```

`python3 -m pinkprocess.kit_app --open` does the same and opens the browser. The server listens on `127.0.0.1:53530`, or a random port if that is taken.

## Using the app

To check a physical kit, choose **Import data**, open [pool-recovery.csv](datasets/pool-recovery.csv), and confirm a capacity of **12**. Under **Skills**, select **Queue history v2**. Ask “What's wrong with my system?” and open **Readings behind this answer**. This recording peaks at 87 waiting and ends at zero.

1. **Connect.** Press **Find boards** on the start screen. The rest of the app opens once a board is connected.
2. **Import data.** Press **Import data** and choose a CSV or JSON lines file. The app reads the columns, types, row count and time column and spots signals such as queue, latency, errors, temperature or sales.
3. **Pick a model.** A recommended quantized model is applied when the data loads. **Search models** opens the library, which you can filter by task, maker and size. **Synthesize model** builds your own from a starting point, a name and a request. **Publish** adds a skill you trained to your library, stored in `~/Library/Application Support/PYNQ Process/models.json`.
4. **Watch the hot path.** The strip on the left edge shows Data, Profile, Model, PYNQ-Z2 and Answer with the time each stage took. Hover it to open it. With no board the fabric stage reads **This Mac**.
5. **Ask.** Type a question or use the microphone key and press Enter. The answer card shows a skeleton and elapsed time as soon as you ask. During board inference, its counter reports completed FPGA projections. The card stays visible as the first text arrives. You can open any ready view, including the transcript and table for an uploaded dataset, while the full answer is still generating. The style row above the input picks which one opens first. A follow-up on a board dataset sees the previous exchange. **Readings behind this answer** shows the data the answer rests on.
6. **Notes.** The **Notes** tab keeps your own notes with the session.
7. **History.** Logged-in users see past sessions in the left sidebar, with dataset, model and question count. Without logging in, everything works but nothing is saved.
8. **Export.** **Export** saves the session as Markdown, CSV or JSON, with every answer style that was ready. In the Mac app it opens a dialog with a Copy button instead of downloading.
9. **Connect to Claude.** The same menu shows the config for the MCP connector (below).

The **Skills**, **Train** and **Replay** tabs change and compare the board's language-model versions. See [board setup](docs/board-setup.md).

## Board setup

A kit needs a PYNQ-Z2 provisioned with the diagnostic checkpoint and the timing-qualified overlay. First [reconstruct the checkpoint](profiles/flan-telemetry-diagnostic/README.md), then install:

```sh
python3 kit/provision.py --install USER@HOST \
  --overlay build/projection-queue-qualified/pinkprocess.bit \
  --language-model build/flan/diagnostic-install \
  --manifest-sha256 14d7912c66a78edf407bbb272f7aa04b953af920827a9a035275194ec4f2f2f2
```

Add `--sqlite-demo` for generated board-local traffic. Provisioning installs the boot service and pairs the provisioning Mac. For another Mac, import the kit's pairing JSON with **PYNQ Process, then Import Kit Pairing…**. Connect over Ethernet or the USB relay, and feed live data with `python3 -m kit.publish_live`. All of it, including the USB appliance, is in [docs/board-setup.md](docs/board-setup.md).

## Signing in

The app uses Auth0 Universal Login with Google. Set `PYNQ_AUTH0_DOMAIN` and `PYNQ_AUTH0_CLIENT_ID` before launching, and allow this callback URL in the Auth0 application:

```
http://127.0.0.1:53530/auth/callback
```

The app listens on port 53530 so the callback stays stable. If the port is taken it falls back to a random one, and sign-in then needs that port allowed too. The ID token must name your client as its audience and carry the nonce the app sent. Without the two variables the sign-in sheet shows **Continue as guest**, which makes a local stand-in profile.

## Data on your Mac

All under `~/Library/Application Support/PYNQ Process/`:

- `history/<user>.json`: saved sessions, one file per profile.
- `models.json`: models you published.
- `datasets/`: the last ten imported files. Imports are limited to 20 MB.
- Board pairings are kept separately in `~/Library/Application Support/pinkProcess/kits/`.

## Claude connector (MCP)

With the app open, `curl http://127.0.0.1:53530/api/mcp` returns a config snippet. Paste its `config_json` into the `mcpServers` section of Claude Desktop's config. The connector is `python3 -m pinkprocess.mcp_server` with `PYTHONPATH` set to this checkout, and the app picks a Python 3.10 or newer for it. Set `PYNQ_USER` to choose a profile, otherwise the most recently used one is read. Tools: `list_sessions`, `get_session`, `list_datasets`, `profile_dataset` and `ask_data`.

The export endpoint is `POST /api/action` with `{"action":"export","session_id":"<id>","session":{...},"format":"md"}`. When a `session` object is posted it is exported as sent, otherwise the saved session with that id is used. The reply is `{filename, mime, content}`.

## Repo map

| Path | What it is |
| --- | --- |
| `mac/` | Swift shell (`KitWindow.swift`, `SpeechInput.swift`, `WorkbenchOrigin.swift`), the backend launcher and `build_kit.sh` |
| `pinkprocess/kit.html` | The whole interface, one file |
| `pinkprocess/kit_app.py` | Local HTTP server and the actions the interface calls |
| `pinkprocess/kit_data.py`, `kit_models.py`, `kit_accounts.py`, `kit_export.py`, `mcp_server.py` | Mac-side profiling and answers, model library, sign-in and history, export, Claude connector |
| `pinkprocess/kit_transport.py`, `kit_server.py`, `board_language.py`, `telemetry_device.py` | Paired transport and the board service |
| `pinkprocess/static/` | Logo and static assets served by the backend |
| `pinkprocess/` (other) | Telemetry simulator, baseline, fixed-point profile and scoring code |
| `datasets/` | Fourteen deterministic test files (queues, latency, thermal, sales, JSON lines, a malformed file, a 50,000 row file) and the script that makes them, see [datasets/README.md](datasets/README.md) |
| `kit/` | Board provisioning, relays, verification scripts, evidence and examples |
| `lm_reference/` | Quantized FLAN-T5 reference, fabric driver and weight versions |
| `profiles/` | Checkpoint deltas and manifests for the resident models |
| `fpga/` | RTL for the classifier and projection engines, and build scripts |
| `build/` | Committed artifacts: the Mac app zip, qualified overlays, study outputs |
| `docs/` | Detail docs and images |
| `cartridge/`, `cartridges/`, `decks/`, `app/`, `hw/`, `scoping/` | Earlier prototype, see [docs/earlier-prototype.md](docs/earlier-prototype.md) |

## Testing

```sh
python3 -m unittest discover -t . -s pinkprocess -p 'test_*.py'
python3 -m unittest discover -t . -s cartridge -p 'test_*.py'
python3 -m unittest discover -s kit -p 'test_*.py'
python3 -m unittest discover -s lm_reference -p 'test_*.py'
```

The interface script can be checked by extracting the `<script>` block of `pinkprocess/kit.html` and running `node --check` on it. Some board tests are skipped without hardware or local checkpoint fixtures.

## More documentation

- [docs/board-setup.md](docs/board-setup.md): provisioning, pairing, live feeds, weight versions.
- [docs/results.md](docs/results.md): measured results and evidence.
- [docs/engineering.md](docs/engineering.md): telemetry baseline, overlay bring-up, language-model reproduction.
- [kit/live-context.md](kit/live-context.md): the board-owned telemetry data path.
- [kit/usb-appliance.md](kit/usb-appliance.md): USB appliance relay and console recovery.
- [research.md](research.md): research log, including the weight-reading studies.
- [docs/earlier-prototype.md](docs/earlier-prototype.md): the expert-cartridge prototype.
- [AGENTS.md](AGENTS.md): guide for contributors and coding agents.
