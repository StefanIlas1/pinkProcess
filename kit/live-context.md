# Board-owned live telemetry

A telemetry producer sends pool capacity and timestamped busy/waiting counts directly to the paired kit. The board owns the rolling history, creates features and scores eligible windows on arrival. A Mac can disconnect, reconnect later and ask for the latest completed answer.

## Context tree

- `README.md` describes the operator workflow and recorded physical-board results. `kit/forecast_acceptance.md` defines the separate forecast-quality study.
- `pinkprocess/kit_server.py` and `kit_transport.py` implement the board service and paired transport. `telemetry_device.py` owns hardware rows and profile updates.
- `pinkprocess/kit_dataset.py` and `backlog.py` implement the existing feature and eligibility rules. Compare both empty-queue onset and nonempty-queue outlook paths against these rules.
- `mac/KitWindow.swift` and `mac/kit_backend.py` connect the native UI to the kit. `mac/LiveCSV.swift` provides the existing Mac-watched CSV workflow.
- `kit/provision.py` and `kit/install-board.sh` package and install board-side code. A working local import alone does not establish that the board deployment contains every dependency.

- `pinkprocess/live_analyst.py` runs open-ended questions against the local quantized instruction model on the Mac. `kit_app.analyst_evidence` projects board evidence into measured occupancy, recent queue history and the time-stamped FPGA forecast. The app supplies these facts on every question.

## Delivery and acceptance

1. Provide a documented, authenticated producer entry point with explicit stream identity, capacity, sample cadence and restart behavior. Reject invalid samples without corrupting the last completed window.
2. Use the board processor to retain the rolling readings and prepare the existing quantized features. Invoke the resident FPGA rows as eligible data arrives. Tie each answer to its input time, model identity and scoring profile. Document warmup and freshness so an operator can tell when the last result was produced.
3. Expose a native Mac question about the board feed. Preserve the existing imported-data and Mac-watched-file workflows. Run reads concurrently with ingestion and keep answers internally consistent across profile changes.
4. Exercise ingress while every query client is disconnected. Instrument the device boundary and verify a score occurs before any read. Read twice and verify the score count holds steady. Append another eligible window and verify the timestamp and answers advance. Compare both queue states to fixed-point reference outputs.
5. Cover malformed cadence, capacity changes, partial batches, duplicate samples, profile changes, restart/warmup and authentication according to the declared contract. Distinguish software integration, native compilation and any physical FPGA observations in the final evidence.

## Integration contract and ownership

- The device owner implements authenticated `POST /v1/telemetry` with `session` (UUID), `source`, `capacity` and `readings`. Each reading carries `time_ms`, `busy` and `waiting`. The read-only operations posted to `/v1/query` are `stream_status`, `stream_forecast` and `stream_observations`. They may select a session, and observations may specify `window_ms`. The reply carries session/source, capacity, sample count, latest sample time, receipt age, freshness and state. The forecast can be null during warmup.
- The Mac owner integrates board-feed questions into `pinkprocess/kit.html` through the bundled `mac/kit_backend.py`. The Mac build entry is `mac/build_kit.sh`. This path asks for resident answers without attaching observations. The file workflows stay available separately.
- The project lead owns the board-local controlled SQLite producer, deployment and physical acceptance. The decisive run records two sample positions and inference evidence across an interval with all Mac clients closed, then asks for the resident answer with no uploaded readings. An external producer may also push telemetry directly to the board.

## Physical acceptance on the deployed PYNQ-Z2

The kit runs `pinkprocess-kit.service` with a board-local SQLite demo enabled. The demo generates controlled pool traffic. The service serializes ingest, profile changes and queries under one lock. On ingest, the board retains the latest 60 seconds, builds a 50-reading input window and submits it to the resident model. `stream_forecast` reads the cached result.

| Check | Observed result |
| --- | --- |
| No Mac reader between snapshots | Within one SQLite demo session, the input timestamp advanced from 47,300 to 55,100 ms, the retained sample count from 474 to 552, and FPGA submissions from 425 to 503. All 78 new readings triggered inference. |
| Authenticated external source | The board accepted 50 known readings for a new session. At 4,900 ms, the queue was empty and `pool-logistic-v1` returned 27,722 with the alert set. Two successive forecast queries returned that score with the submission count fixed at 510. |
| Change to a nonempty queue | Appending the reading at 5,000 ms with three requests waiting advanced the counter to 511. The cached `queue-outlook-v2` persistence and growth scores were 41 and 2. |
| Source handover | After the verification publisher finished, a fresh SQLite demo session reached 52 samples and ready state. |

The verified Mac-to-board path used a USB UART relay. Direct connectivity from the Mac's Wi-Fi network to the board's wall Ethernet connection remains unverified. The source for the physical run was controlled traffic and known verification readings. These checks establish inference timing and query behavior. The separate forecast-quality study evaluates prediction accuracy. [Physical acceptance record](../build/live-kit/physical.json)

## Open-ended diagnosis

Board release `c7d261bf67ac39071cc5` supplies one coherent capture of recent
measurements and the most recent quantized FPGA forecast. The installed Mac
app sends the question and that evidence to Qwen3 4B Instruct 2507 (Q4_K_M) on
loopback. The board continues ingesting as the Mac generates text.

The [language-model acceptance record](../build/live-kit/analyst-acceptance.json)
contains the exact questions, evidence, generated text, model digest and
latency. The four cases cover a recovered queue, an increasing queue,
unavailable endpoint and latency data, and a window average. In the recovered
case the model correctly reported the current empty queue, earlier peak of 117
and the ten-second warning. It overstated recovery as stability. The app shows
the captured readings alongside each generated answer so operators can inspect
the values underlying an interpretation.
