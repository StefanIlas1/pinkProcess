# Guide for contributors and coding agents

Read the [README](README.md) first for what the product is and what is real versus simulated. This file covers how the code is laid out and the rules to follow when changing it.

## Naming

The product is spelled **PYNQ Process** everywhere a user can see it: window title, page title, menus, copy, docs. The repository, the Python package (`pinkprocess`), the app bundle (`pinkProcess.app`) and the zip keep the older spelling. Do not rename those.

## Architecture

```
Swift shell (mac/KitWindow.swift)
  -> WKWebView showing pinkprocess/kit.html
    -> local HTTP server (pinkprocess/kit_app.py, 127.0.0.1:53530)
      -> board client (pinkprocess/kit_transport.py) -> PYNQ-Z2 (kit_server.py, board_language.py, FPGA overlay)
```

- The shell starts the backend (`mac/kit_backend.py`, frozen by PyInstaller in the built app), reads the address it prints, and loads that page. It only trusts the root page of that origin (`mac/WorkbenchOrigin.swift`), plus `/auth/login`, which it lets through so Auth0 can start. External https links open in the system browser.
- The interface is one file, `pinkprocess/kit.html`, with inline CSS and JavaScript. It talks to the server with `POST /api/action`, `POST /api/account` and a few `GET` routes.
- `kit_app.py` holds `Kit` (connection, board actions, jobs) and `KitHandler` (routes, origin checks). Board calls go through `Kit.board()`. Long work runs as a job that the page polls.
- With no board, imported files are handled on the Mac by `kit_data.py`.

## Where things live

| Feature | Files |
| --- | --- |
| Import, profiling, recommendation, local answers in five styles | `kit_data.py` (parse, profile, `answer_question`), `kit_app.py` (`import_data`, `data_ask_begin`, `data_ask_progress`) |
| Board CSV upload, staged answers, follow-ups, model synthesis | `kit_app.py` (`dataset_*` actions), `board_language.py` (board side) |
| Live questions, saved incidents, weight versions | `kit_app.py` (`ask_begin`, `answer_progress`, `lm_*`), `board_language.py`, `lm_reference/weight_versions.py` |
| Sign-in, history, notes | `kit_accounts.py` (Auth0 code flow with PKCE, nonce and audience checks, per-user JSON history) |
| Model library and marketplace | `kit_models.py`, `/api/models`, `model_publish` and `model_unpublish` account actions |
| Export | `kit_export.py` (`export_action`) |
| Claude connector | `mcp_server.py`, `/api/mcp` |
| Dictation | `mac/SpeechInput.swift`, bridged to `window.pinkprocessSpeechInput` in `kit.html` |
| Hot-path strip, answer tabs, skeleton loader | `kit.html` (`pathMark`, `renderMessages`, `setVariant`) |
| Logo | `pinkprocess/static/logo.png`, served at `/static/logo.png` |
| Simulated board for development | `board_sim.py`, enabled by `PYNQ_SIM_BOARD=1` in `make_server` |

## Code style

- Python: snake_case, standard library only, no new dependencies. Python 3.10 or newer, but `mcp_server.py` and the modules it imports must still import on 3.9 (use `from __future__ import annotations` for `X | None` annotations).
- No code comments in any language. Name things so they read clearly instead.
- Tests live next to the code as `pinkprocess/test_*.py` and use `unittest`. Add a test for every backend change.
- Interface: plain words, short labels, no emojis or emoji-like glyphs (no stars, check marks or arrows as text). Use inline SVG icons. Middle dots are fine as separators. Latency is always shown in milliseconds.
- No page scrolling: `body` is `100vh` and only the chat list and history list scroll. The window must work from 700 by 480 up, and below that it shows a "too small" message.
- Palette variables in `kit.html`, sampled from the logo: `--pink` (`#fdcedf`, light letter fill), `--pink-deep` (`#aa1145`, outline, focus rings, active tab underline, send key border), `--pink-bow` (`#fb648d`, small accents such as the Recommended tag). Use only these three pinks. Green is for connection status only.

## Run and test

```sh
python3 -m unittest discover -t . -s pinkprocess -p 'test_*.py'
python3 -m unittest discover -t . -s cartridge -p 'test_*.py'
python3 -m unittest discover -s kit -p 'test_*.py'
python3 -m unittest discover -s lm_reference -p 'test_*.py'
```

Check the interface script by extracting the `<script>` block of `kit.html` to a temporary file outside the repo and running `node --check` on it.

Dev server (open the printed address in a browser):

```sh
PYTHONPATH=$PWD python3 mac/kit_backend.py
```

Compile the Swift files without building the app (scratch output path, run from the repo root):

```sh
cp mac/KitWindow.swift /tmp/kit_main.swift
xcrun swiftc -O /tmp/kit_main.swift mac/SpeechInput.swift mac/WorkbenchOrigin.swift -o /tmp/kit_app_check
xcrun swiftc -parse-as-library mac/test_workbench_origin.swift mac/WorkbenchOrigin.swift -o /tmp/workbench_test && /tmp/workbench_test
```

Build the app with `PYTHON=/path/to/python3 mac/build_kit.sh`. It needs PyInstaller, bundles `kit.html` and `static/`, and writes `build/mac-kit/pinkProcess.app`.

## Simulated board and screenshots

There is no demo mode in the app. To see the connected interface without hardware, start the backend with a simulated board. The flag is for development only and appears nowhere in the interface:

```sh
PYNQ_SIM_BOARD=1 PYTHONPATH=$PWD python3 mac/kit_backend.py
```

`pinkprocess/board_sim.py` answers the board actions the interface uses (status, telemetry from `simulator.py`, streamed answers, saved incidents, dataset upload, follow-ups). The kit starts already connected to it. Synthesis and other board-only actions return a clean error.

The page accepts parameters in the URL hash (the server only serves the exact path `/`), for example `http://127.0.0.1:53530/#login=1&tab=weights`. They exist for headless screenshots.

| Parameter | Effect |
| --- | --- |
| `login=1` / `logout=1` | Sign in as the local guest profile / sign out |
| `tab=notes\|incidents\|weights\|create` | Open a rail tab |
| `dialog=library\|mcp\|plan\|login` | Open that dialog |
| `menu=export` | Open the export menu |
| `peek=1` | Show the hot-path strip expanded |
| `help=1` | Open the help popover on the tab row |

Headless Chrome works for screenshots: `--headless=new --screenshot=out.png --window-size=1000,650 --virtual-time-budget=9000 URL`.

## Rebuilding the committed Mac app zip

The zip in `build/` is what the README links to. After `mac/build_kit.sh`:

```sh
ditto -c -k --keepParent build/mac-kit/pinkProcess.app build/mac-kit/pinkProcess-Mac-arm64.zip
git add -f build/mac-kit/pinkProcess-Mac-arm64.zip
```

`build/` is ignored by git, so the zip must be added with `-f`. Rebuild it when the interface, backend or Swift shell changes in a way users will see.

## Git policy

- Merge `origin/main` into your branch. Never rebase and never rewrite history.
- In a conflict, backend code from the board side wins where behavior overlaps, and the interface in `kit.html` stays as the project's design. Port board features into the existing components instead of replacing them.
- Do not commit generated files other than the zip described above. Do not delete evidence, test data, scripts or bitstreams under `kit/evidence/`, `build/` or `fpga/`.
- Only push when asked to.

## Docs

Keep docs short and in plain language, without em dashes. Product overview goes in the README, operating a kit in `docs/board-setup.md`, measured results in `docs/results.md`, reproduction procedures in `docs/engineering.md`, and the research log in `research.md`. Anything historical must be labeled as such and say what is current.
