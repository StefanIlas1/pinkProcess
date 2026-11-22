# pinkProcess

pinkProcess is an Apple Silicon Mac app for investigating connection-pool
telemetry on a paired PYNQ-Z2. Open the dark workspace, upload a CSV or
connect to the live feed, and ask the kit what changed. The board runs a
quantized FLAN-T5-small diagnostic model. Its FPGA performs the learned
integer matrix projections and its ARM processor coordinates generation.

[Download the Mac app](https://github.com/StefanIlas1/pinkProcess/raw/refs/heads/main/build/mac-kit/pinkProcess-Mac-arm64.zip) · [Operator walkthrough](#query-the-pinkprocess-kit) · [Research notebook](research.md)

## Product goal

The kit generates answers about live data using a quantized language model
whose inference runs on the FPGA. The Mac provides the operator interface.
The first domain is connection-pool telemetry.

- With the Mac’s model runtime turned off, I can plug in the kit, open pinkProcess, and ask an unfamiliar question while new data keeps arriving.
- On incidents held out from development, the model can distinguish a problem that has cleared from one that is still happening. It cites the relevant readings, separates measurements from predictions, and tells me what to measure when the cause is uncertain.
- I can replay the same incident and question before and after a weight edit, observe the intended behavior change, power-cycle the kit and verify the saved weights reload, then roll back and reproduce the original answer.
- I can ask Astra to create a weight edit and save it as a named version. Selecting any saved version loads its exact weights for FPGA inference. The app confirms which version is active, and the kit reloads it after a power cycle.
- I can select Restore original at any time. The kit verifies the original weights, and my saved versions remain available.

**Current status.** The FPGA scores live telemetry. A provisioned language-model
kit also supports a board-local FLAN encoder and decoder with FPGA INT8
projections. Direct review of an independently seeded software evaluation
found the frozen diagnostic weights answered 12 of 12 synthetic incidents
correctly. The physical PYNQ-Z2 also reproduced the reference answers on three
scripted incidents. It distinguished a cleared queue from an active backlog
and recommended measuring arrival rates and connection hold times when the cause
was uncertain. The first FPGA design took about 13 minutes per answer. On the
same saved ambiguous incident, the installed DDR projection array produced
first text in 18.1 seconds and the complete answer in 132.1 seconds. All 3,722
learned projections used the array, the answer matched byte for byte, and the
Mac received fresh board telemetry during generation.
[Physical replay and weight checks](kit/evidence/physical-systolic-diagnosis.json).
A later board-runtime update used C++ dispatch and reusable buffers on the
same image. Three new synthetic incidents queried through the installed Mac
app's USB backend took 46.0, 48.6 and 59.4 seconds to complete. All three
answers matched their references and the recorded observations.
[Questions, outputs and timings](kit/evidence/physical-projection-cache.json).
Keeping the vocabulary weights in DMA buffers then gave a 47.345-second warm
saved replay and a 54.433-second answer to a new synthetic question. A real
named edit and rollback each reproduced the corresponding archived answer.
[Resident weights and version changes](kit/evidence/physical-projection-resident-head.json).
The installed Mac app also answered an unfamiliar question while new readings arrived. After a witnessed power cycle and recovery from a startup
overlay conflict, the kit reproduced the edited answer exactly. Restoring the
original weights reproduced the baseline and preserved the named edit. With the
startup fix installed, the board also reloaded the selected weights on an
unattended software restart. The Mac app reconnected and restored the original
while retaining both named edits.
[Checkpoint](profiles/flan-telemetry-diagnostic/README.md) ·
[Held-out results](kit/evidence/blind-diagnosis-replacement-review.md) ·
[Physical incident results](kit/evidence/three-diagnosis-traces-physical.md) ·
[Native Mac query](kit/evidence/physical-native-ask.json) ·
[Physical cold-cycle record](kit/evidence/physical-diagnosis-cold-cycle.json) ·
[Native version switching and restart](kit/evidence/physical-diagnosis-warm-boot.json)

A later native-window question exposed an important diagnostic error. The kit
counted 27 requests waiting and a recent peak of 92, but the model called
the queue cleared. First text took 10.5 seconds and the complete diagnosis
took 46.2 seconds. The pinned software reference generated the same error.
Inspect the measured readings alongside each answer.
[Native-window quality record](kit/evidence/native-window-projection-quality-miss.json)

For the queue-onset predictor, the operator-facing quality measure is how often
a warning appears on the Mac at least one second before a real request queues,
alongside misses and false alarms.

## Query the pinkProcess kit

On an Apple Silicon Mac, download and extract the
[arm64 Mac app](https://github.com/StefanIlas1/pinkProcess/raw/refs/heads/main/build/mac-kit/pinkProcess-Mac-arm64.zip).
The app connects to a provisioned PYNQ-Z2.

1. Power the provisioned kit. Connect its **PROG/UART** USB port to the paired
   Mac if the USB appliance relay is enabled, or use a reachable Ethernet
   network. Open **pinkProcess.app**. Allow macOS local-network access when
   prompted. The app scans on launch. Select the kit
   and press **Connect FPGA**. Use **Find boards** to scan again after plugging
   in. The active model version comes from the board.
2. In **CSV synthesis**, enter the pool capacity and choose **Upload CSV**.
   To try it, download the [synthetic recovery CSV](kit/examples/pool-recovery.csv)
   and enter **4**. Uploads accept the integer columns `time_ms,busy,waiting`, in that order,
   with 50–2,097 consecutive samples at 100 ms intervals. The sample contains
   60 readings. A queue
   reaches 12 and drains to zero before the final reading. The table shows the
   measurements that let you check the answer.
3. Pick a saved **Dataset model**, or choose **Create model…**. To synthesize
   a version, supply a name, a single-token target (e.g. `Now`), and an edit
   request. The kit calibrates on this CSV if needed. The signed-in Codex CLI
   on the Mac asks Astra to design an edit, then the kit saves the quantized
   output row under that name and selects it for this dataset. Synthesis
   reuses the provisioned FPGA circuit and the original FLAN checkpoint.
   Each dataset query loads its selected version and restores the live model
   afterward. Existing versions can be selected without Astra.
4. Type a question such as **Did the queue clear, and what was its peak?**
   or use **Dictate**. Native dictation uses on-device speech recognition.
   Allow speech and microphone access if prompted, check the editable
   transcript, then submit. **Transcript** shows your question. **Generalization** describes the uploaded
   readings. **Short answer** reveals the first complete sentence as it arrives.
   **Summary** excerpts the completed answer. **Final** preserves the model's
   full wording. All three answer views come from one generation.
5. Open **CSV table** to traverse the readings. Open **Chat** for a follow-up
   such as **Which reading supports that conclusion?** The kit uses this
   upload and the previous exchange. The live board feed keeps its own samples
   while you work with a file.

For a continuously arriving source, use **Board telemetry** in the same
window. Check the source name, age and status before asking a question.
The built-in SQLite workload is labeled **Simulated**. **Readings behind
this answer** opens the incident captured for a live question. The predefined
measurement and forecast shortcuts retrieve their exact board values.
**Queue-onset model** and the backlog model have separate predictor weights.

The board processor keeps the most recent 600 readings, prepares fixed-point
features and submits each eligible reading to the FPGA as it arrives. After
five seconds of continuous history, the fabric scores queue onset when nobody
is waiting and queue persistence and growth when a backlog exists. Questions
about observed values use the board's retained history. Forecast questions
read the stored model output with its timestamp and model version.

A pool's occupancy and waiting counts can show when requests backed up and
when the queue drained. Identifying a slow database query, a connection leak
or a traffic spike requires additional measurements. An analyst response can
suggest how to distinguish those explanations.

### Edit and replay the resident language model

After a generated answer, expand **Language model weights** and create a named
version. Enter a word that corresponds to one output token (such as “Now”) or
a token ID, and describe the intended first-token behavior. **Ask Astra and
save** uses the signed-in Mac Codex CLI to design a row edit. The board
stores the exact edited bytes against the original checkpoint. Select the
name and press **Load version**. Each load verifies the classifier digest
and probes an output row through FPGA scores. **Restore original** keeps the
named versions in the library. An output-row edit affects that token at every
generation step. In the physical test, raising “Now” also made it recur at
sentence boundaries and in the ending. [Before-and-after answers](kit/evidence/physical-diagnosis-edit.json)

Replay the captured question after a change to compare answers for identical
data. The kit persists captures and answers so they can be reopened from the
saved incident picker after reconnecting. Expand **View first answer** to read
the stored result immediately. **Replay with active weights** runs a new generation
on the same captured evidence.

### Connect a live source

From the checkout on a producer with network access to the board, pipe your
pool monitor's newline-flushed CSV into the publisher. Supply the actual pool
capacity, the board hostname or address and the private kit pairing file.

```sh
python3 -m kit.publish_live --host BOARD --pairing /path/to/pairing.json \
  --source 'Service connection pool' --capacity 16
```

The publisher reads standard input. Send the header `time_ms,busy,waiting`
followed by a reading every 100 ms. An authenticated live source takes
precedence over the demo. To analyze a file on the Mac instead, expand
**CSV analysis on this Mac** and use **Import CSV**, **Use sample**, or
**Watch live CSV…**.

If the Mac is on Wi-Fi and the board is on wall Ethernet, select **Find boards**
after moving the kit. A network route between the two devices is required for
Ethernet queries. On the tested setup the Mac reached the paired board through
USB UART. For this fallback, leave the board's USB console logged in at a shell
prompt, close other serial clients and run
`python3 kit/usb_relay.py /dev/cu.usbserial-DEVICE` on the Mac. Leave the relay
running and select **Find boards**, then **Connect FPGA**. The app verifies
the paired board's TLS certificate through the relay at `127.0.0.1`.

An administrator can explicitly provision the optional USB appliance relay
for an installed kit. It opens the board-side relay at boot, so a paired Mac
can reconnect to a power-cycled board using the Digilent FTDI UART interface
ending in `_tul1`. Close other Mac serial programs during automatic use. This
mode takes over the board's serial console. See the provisioning and console
recovery instructions before enabling it. [USB appliance setup](kit/usb-appliance.md)

The physical acceptance run recorded 78 FPGA submissions over 7.8 seconds of
stream time with the reader disconnected. Two subsequent reads of a controlled
forecast produced zero extra submissions. See the [context tree](kit/live-context.md)
for the data path and evidence.

### Set up another kit or Mac

For a fresh diagnostic kit, first [reconstruct the diagnostic checkpoint](profiles/flan-telemetry-diagnostic/README.md).
Provision the board with the matched, timing-qualified overlay and pinned model:

```sh
python3 kit/provision.py --install USER@HOST \
  --overlay build/projection-queue-qualified/pinkprocess.bit \
  --language-model build/flan/diagnostic-install \
  --manifest-sha256 14d7912c66a78edf407bbb272f7aa04b953af920827a9a035275194ec4f2f2f2
```

Follow the [projection configuration](#benchmark-the-flan-projection-accelerator)
to enable all learned projections on the fabric. Provisioning installs the
boot service and pairs the provisioning Mac. Add `--sqlite-demo` to enable generated board-local traffic. For another Mac, transfer the kit's pairing JSON privately, then select
**pinkProcess → Import Kit Pairing…**. The file grants access to query the kit
and change its active profile.

For older kits without a resident language checkpoint, install Ollama on the Mac and download the local Qwen3
4B Instruct 2507 model once:

```sh
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

pinkProcess contacts Ollama on loopback. On a Homebrew installation it starts
the installed runtime on demand. The model stays on the Mac. Board readings and
open-ended questions are processed locally.

To build the Mac app, run `mac/build_kit.sh` with `PYTHON` set to an interpreter
with PyInstaller installed. The script uses Apple's Swift compiler and bundles
the Python runtime. The app is built for the local Mac architecture and signed
locally. The output is `build/mac-kit/pinkProcess.app`. Developers can also run
`python3 -m pinkprocess.kit_app --open` for the browser interface.

### Physical board check

On this kit the 201-reading sample contained 103 eligible forecast windows.
The base profile warned on 103 and the alternate profile warned on nine.
All 309 scores from the base, alternate and restored runs matched the Mac
fixed-point reference. The restored board weight hash matched the initial hash.
These counts measure how the profiles differ. Assess the forecast quality with
labeled queue episodes. The recorded responses are in
[the board report](build/pynq/telemetry-kit-physical.json).

## Measured SQLite pool workload

The first measured workload used a local SQLite database and a FIFO pool of
four connections. Three one-minute runs issued real reads with controlled
arrival rates and transaction hold times. Every change in occupancy or queue
length was timestamped on the Mac's monotonic clock. The collector exported
100 ms samples for the app. The evaluator checked every sample against the
event log, including brief queues that drained between samples.

Replay both frozen profiles on the same board and compare them with an alert
whenever the pool is full. The forecast horizon is ten seconds, with five
seconds of history. Each positive streak counts as one alert, matched to the
next episode. Predictions stop ten seconds before the end of each recording so
each prediction has a complete outcome. Duplicate alerts for the same episode
are reported separately. Run from the repository root with a **new** output
directory:

```sh
python3 -m kit.capture_pool build/pool-study/new-run
python3 -m kit.evaluate_pool build/pool-study/new-run BOARD_HOST
```

The three captures completed 2,259 transactions and contained 99 scored queue
episodes. The physical PYNQ-Z2 reproduced all 1,706 integer reference scores
across 853 eligible windows for each profile. The board finished on the original
base weights.

| Condition | Warned | Missed | False alerts | Median lead on warned episodes | Warned ≥1 s ahead |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base profile | 42 | 57 | 11 | 377 ms | 11 |
| Service B profile | 40 | 59 | 1 | 65 ms | 5 |
| Warn when full | 35 | 64 | 1 | 38 ms | 3 |

The Mac app watched the first capture while it was being written. Eight sampled
UI updates were first observed a median 338 ms after their corresponding CSV
append (range 207–524 ms). These timings include observation delay. The watcher
refreshes about once per second, while the table scores every eligible 100 ms
window. The table therefore describes model lead time, not operator-visible
warning time. The operator-visible test below evaluates warning delivery at the app's cadence
and adds incoming request rate.

[Per-run board results](build/pool-study/observed/report.json) ·
[Native display timings](build/pool-study/observed/live-display.json) ·
[Importable workload trace](build/pool-study/observed/7101/telemetry.csv)
(capacity 4).

### Operator-visible warning test

The [acceptance protocol](kit/forecast_acceptance.md) defines the goal for
incoming request rate as an additional input. The candidate keeps the Service B
coefficients on the FPGA. The Mac derives offered load from arrivals already
recorded at each sample. The test runs each condition through the shipping
result page in a native WebKit test window, calls the physical board once per
second and records the displayed text, score, weight hash and elapsed time
after two display frames. This replay fixes the watcher phase at 50 ms after
the latest sample. The full-pool rule uses the corresponding Service B
presentation timestamps. Only forecasts displayed at least one second before
the next queue episode earn early-warning credit.

Across three fresh measured runs (2,315 SQLite transactions, 113 evaluated
queue episodes), each model produced 138 verified WebKit results. The base
profile warned before 11 episodes, five at least one second early, with two
false alerts. Service B delivered zero early warnings and zero false alerts.
Adding request rate delivered one warning before onset, less than one second
early, and zero false alerts. The full-pool rule also delivered zero early
warnings. The candidate failed the promotion criteria.

Only nine episodes had an eligible observation with enough remaining lead for
one-second credit under this protocol. Discrete one-second polling, forecasting
only from an empty queue, and matching the first subsequent queue onset all
constrain this count. The physical board finished on the base weights; switching
profiles and restoring them reproduced all 389 base scores for the first dataset.
See the [delivery results](build/pool-study/rate-holdout/delivery-report.json)
and [rollback evidence](build/pool-study/rate-holdout/rollback.json).

To repeat the complete measurement, use a fresh capture directory:

```sh
python3 -m kit.capture_pool build/pool-study/new-delivery --seeds 8401 8402 8403
xcrun swiftc -O kit/MeasureWindow.swift -o build/pool-study/measure-window
python3 -m kit.evaluate_delivery build/pool-study/new-delivery --all \
    --window build/pool-study/measure-window --host BOARD_HOST
```

### Queue outlook while requests are waiting

In **Question**, select **Will the current queue persist or grow over the next
ten seconds?** Use an imported file or watch a live CSV. The first answer
predicts whether the current queue will stay nonempty continuously. The second
predicts whether more requests will be waiting at the ten-second endpoint than
at the input timestamp. An empty latest reading reports that no requests are
waiting. Set the actual pool capacity before querying.

The Mac encodes current queue depth and its four-second change as signed INT8
features. Two resident rows in the FPGA (`queue-outlook-v2`) project that trend
to the ten-second endpoint and score the direction of change. The board
verifies both scores against its integer reference. The onset model occupies a
separate row and retains its own weight selection and rollback.

We compared a class-balanced logistic model and a simple four-second trend
on six previously unseen SQLite workloads. The trend model scored higher for
both questions. We froze its quantized weights and ran another six workloads
with fresh seeds before selecting the default. On 193 queued one-second
observations in that confirmation set, the selected model identified all 136
uninterrupted queues and 33 of 57 queues that drained. It detected 61 of 62
growing queues and correctly identified 88 of 131 that did not grow. Balanced
accuracy was 78.9% for persistence and 82.8% for growth. Overlapping
ten-second windows from the same workload are correlated. These runs used a
four-connection local SQLite pool with controlled bursts.

[Final board results](build/backlog-study/final-board-report.json) ·
[Fitted model comparison](build/backlog-study/confirmation-v1.json) ·
[Example input at capacity four](build/backlog-study/queued-at-25s.csv)

## Run the telemetry baseline

The deterministic simulator requires Python 3.10 or later and uses the standard
library. From the project root, run:

```sh
python3 -m pinkprocess
```

The demo fits a rule on eight 10-minute runs, then prints event-level metrics
and two concrete forecasts from four different 10-minute runs.

## Simulation and labels

- `simulator.py` models 12 connections with a FIFO queue. Arrivals have
  exponential gaps within each traffic regime. Ordinary traffic averages seven
  arrivals per second, and intermittent bursts average 11 to 21. Connection
  holds are sampled independently with a mean of about 850 ms. The simulator
  processes arrivals and releases in millisecond event time, with releases first
  on ties, then records busy and waiting counts every 100 ms. Each run's seed
  controls its traffic and service times.
- `dataset.py` emits an example at time `t` only when nobody is waiting and
  both windows fit within the run. The input contains 50 readings from
  `(t - 5 s, t]`. The label is positive when an arrival joined the queue in
  `(t, t + 10 s]`. Short queues can form and drain between samples, so labels
  come from the exact arrival log. Overlapping windows share data, so training
  and evaluation use disjoint sets of run seeds.

## Baseline and scoring

The score adds current utilization to the change in utilization per second
over the most recent two seconds multiplied by a positive weight in seconds.
The score can exceed 1 and has no probability interpretation.
The training grid chooses the weight and threshold that maximize
`2 × warned / (2 × warned + missed + false alerts)`, breaking ties in favor
of fewer false alerts.

A queue episode starts when an arriving request joins an empty queue and ends
when the queue drains. An alert starts on the first positive prediction in a
streak of consecutive 100 ms predictions. A negative prediction or a skipped
tick breaks the streak. Each alert start is matched to the first episode that
begins within its forecast. An episode gets one warning credit, and its lead
time runs from its earliest matched alert to its first queued arrival. Later
alert starts matched to that episode are reported as repeats. An alert start
with no episode in its forecast is false. A sustained positive streak starts
only one alert, even if a later prediction in the streak has a different
outcome.

The episode count includes starts after the first possible prediction time
(5 s) through run end. Near the end, later episodes have fewer eligible
prediction times because prediction stops 10 seconds before run end. False
alerts per hour use the total simulated duration of the held-out runs.

## Fit and save a logistic profile

```sh
python3 -m pinkprocess.train --output /tmp/pinkprocess-logistic.json
```

The [reference profile](profiles/pool-logistic-v1.json) stores the three
weights in feature order, the intercept, and the alert threshold. The
features are current utilization, its change per second over two seconds, and
the fraction of the last ten readings at full capacity. The prediction uses
`intercept + sum(weight * feature)` and alerts when that score reaches the
threshold. The sigmoid converts the score to a probability during training.
Inference can compare the raw score directly because the sigmoid is monotonic.

The trainer fits weights on seeds 100–105. On seeds 106–107, it sweeps score
thresholds and chooses the one that warns about the most episodes within the
baseline's validation false-alert count. Both models use the same event-level
scoring on seeds 200–203. The demo reloads the JSON and checks that every
held-out score and the decision threshold are reproduced exactly.

For inference without fitting, load the JSON with `Profile.load(Path(...))`.
Pass the result of `profile_features(example, capacity)` to
`profile.score(...)` and compare the returned score to `profile.threshold`.

## Try weight authoring on a second service

This workload changes the mean connection hold from 850 ms to 1,100 ms.
The capacity, arrival model, features, and label definition stay the same.
Fit runs use seeds 300–305, validation uses 306–307, and test uses 400–403.

```sh
python3 -m pinkprocess.author --output-dir /tmp/pinkprocess-service-b
```

Open `/tmp/pinkprocess-service-b/astra-training.txt` and give it to Astra.
It contains the instructions and a lossless grouping of the training
examples by feature vector and label. Each row's count is its frequency in
the training set. The conventional fit uses those same counts. Save the
returned JSON object with `weights` and `intercept` to
`/tmp/pinkprocess-astra-reply.json`, then run:

```sh
python3 -m pinkprocess.author --output-dir /tmp/pinkprocess-service-b \
  --astra /tmp/pinkprocess-astra-reply.json
```

The command calibrates both profiles against the same validation
false-alert budget, evaluates them on the same untouched test runs, and
saves the imported weights and locally chosen threshold in
`astra-profile.json`. It verifies the saved profile by reloading it and
comparing all held-out scores. Future scoring can use that JSON directly.

## Replay the Astra profile with fixed-point arithmetic

```sh
python3 -m pinkprocess.fixed_point --vectors /tmp/pinkprocess-fpga-vectors.json
```

The command loads the saved Astra profile and replays the four second-service
test runs. Features and weights use scale 256. Products, the intercept, and
the threshold use scale 65,536. Encoding rounds to nearest with ties to even,
including for negative utilization trends. After encoding, inference performs
three signed integer multiplications, adds the products to the integer
intercept, and compares the sum with the integer threshold using `>=`.

The [reference vectors](profiles/service-b-fixed-point-vectors.json) give
the integer inputs, expected sum, and expected alert bit. Replay vectors
include their run seed and timestamp. Two synthetic vectors exercise the
minimum and maximum sum over independently bounded inputs. Widen signed
products before adding them. The JSON lists the required signed widths for
the operands, products, sum, and comparison for this profile.

## TinyStories INT8 output-layer reference

Use the pinned [llama2.c checkout](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb),
the [stories260K checkpoint and tok512.bin](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da/stories260K),
Python, and Clang. The script checks the SHA-256 of all three source inputs
before building and running the reference.

```sh
python3 lm_reference/reference.py \
  --source /path/to/llama2.c/run.c \
  --model /path/to/stories260K.bin \
  --tokenizer /path/to/tok512.bin \
  --vectors /tmp/pinkprocess-lm-int8.json
```

The reference copies the output classifier into its own buffer and checks that
the copy reproduces the original floating-point output and token IDs. The
transformer body runs in floating point. The output layer uses signed INT8
with a single weight scale across the matrix and one input scale per forward
pass. It rounds halfway values away from zero and clamps to ±127. Each row
uses a signed 32-bit integer dot product. The Mac converts sums back to scores
for token selection.

The JSON records the encoded weight rows, activations, scales, SHA-256 values,
and expected raw sums in vocabulary order. Use the sums for FPGA parity.
The reference checks two fresh INT8 runs for agreement and recomputes every
sum independently in Python.

### One-row edit and rollback

Add `--edit-demo` to the reference command to include an A → B → A trial in
the exported JSON. The trial uses greedy decoding with the fixed prompt
`Once upon a time, there was a little girl named`. It confirms that appending
` Tim` adds exactly one token, then replaces the 64 INT8 weights of row 326
with values aligned to the activation at the first sampled position.
Each phase starts a fresh generation process. A temporary replacement
file contains the two-byte little-endian row ID followed by the 64 signed
weight bytes. The checkpoint is read-only.

The first sampled token changed from Lily (317) to Tim (326) and returned to
Lily after rollback. Tim's raw dot product at that position was 14,330,
177,038, then 14,330. The edited story repeated Tim. This deliberately
strong edit tests the mechanics of modifying and restoring a real row.
It provides a failing example for behavior selectivity. Rollback reproduced
the entire classifier matrix, the activation and score traces, the generated
token IDs, and the text. The JSON contains the row bytes, hashes, and outputs
for all three stages.

## PYNQ-Z2 classifier bring-up

The classifier RTL accepts the Mac reference's 64 signed activations and
computes all 512 vocabulary scores as signed 32-bit sums. The processor
writes the weights and inputs over AXI4-Lite. The FPGA computes the sums
and exposes them for readback. AMD Vivado with Zynq-7000 support and the
PYNQ-Z2 board preset builds the overlay on an x86-64 Windows or Linux host.

1. Install Icarus Verilog and run `fpga/build.sh --check-only` to compare the
   RTL scores with the shipped integer reference, including an edited weight
   row and its restoration.
2. On the Linux Vivado host, source the Vivado environment and run
   `fpga/build.sh build/vivado-pynq` from the checkout. Use a new output
   directory. This repeats the reference check and builds the kit's existing
   RTL design. The outputs are `pinkprocess.bit` and its matching
   `pinkprocess.hwh`, which describes the register map. This workflow adapts
   Mohsen’s reference-vector check before synthesis to the same overlay used
   for live forecasts and language-model projections. [Reference circuit check on main](cartridge/check.py)
3. Upload both files, the reference JSON, and `fpga/check_classifier.py`
   to the same folder on the board using Jupyter. Open a Jupyter Terminal
   in that folder and run:

   ```sh
   python3 check_classifier.py pinkprocess-lm-int8.json --bitstream pinkprocess.bit
   ```

   From an unprivileged SSH shell on the stock PYNQ v3.1.1 image, run
   the command in a root login shell so the PYNQ profile loads XRT:

   ```sh
   sudo bash -lc 'exec /usr/local/share/pynq-venv/bin/python3 check_classifier.py pinkprocess-lm-int8.json --bitstream pinkprocess.bit'
   ```

The checker loads the 512 rows, runs the 32 reference activations, and
compares all 16,384 raw sums. It also runs three additional vectors while
changing row 326 to force large positive and negative sums and finally
restoring the original weights. Only one request runs at a time.
The signed sums use these byte offsets from the classifier's AXI base address:

| Offset | Purpose |
| --- | --- |
| `0x0000–0x7FFF` | Weights, 512 rows of 64 bytes in token-ID order |
| `0x8000–0x803F` | One 64-byte activation vector |
| `0x8040–0x883F` | 512 four-byte scores, readable after completion |
| `0x8840` | Status with busy at bit 0 and done at bit 1 |
| `0x8844` | Write 1 to start and clear done |

Four signed INT8 values are packed into each little-endian word. The live
demo feeds the same raw scores into token selection.

### Benchmark the FLAN projection accelerator

On the PYNQ-Z2, the first encoder query projection went from 2,100.19 ms
to 81.48 ms (25.8×) and the single-position decoder query projection went
from 43.14 ms to 12.13 ms (3.56×). These are medians of five complete calls
per backend. Both matched the signed integer reference. In a separate
concurrency run, 13 telemetry submissions started while the projection core
was busy and all forecasts matched the sequential run. The routed design
met timing at 50 MHz and implemented the array with LUTs. [Physical measurements](kit/evidence/physical-systolic-projection.json)
· [Matched bitstream and hardware handoff](build/projection-qualified)

The recorded diagnosis had 186 input tokens and 50 decoding steps. The
following are single physical replays with the same saved incident, question
and original weights. The historical baseline used the paired connection.
Both new runs used the same board-local request and one-second result polling.

| Installed configuration | First observed text | Complete answer |
| --- | ---: | ---: |
| Historical classifier | 334.9 s | 797.5 s |
| Array for block-zero queries | 342.1 s | 830.2 s |
| Array for every learned matrix | 18.1 s | 132.1 s |

The board also matched 842,752 signed reference scores during two separate
shape checks covering attention, feed-forward layers and the original and both
edited output heads. One isolated output-head call took about 0.6 seconds,
including eight weight segments. The current full path packs and transfers the
selected weights again on every call. [Replay evidence](kit/evidence/physical-systolic-diagnosis.json)

The overlay source includes the existing telemetry classifier and an 8 × 8
signed INT8 projection engine. One command reads packed activations and
weights from board DDR through the Zynq HP0 port and writes INT32 scores back.
Two buffers alternate between loading and computing. Encoder mode processes
eight positions by eight outputs. Single-position decoder mode processes 64
outputs in parallel. The FPGA re-reads operands across output and position
tiles. Board benchmark timings include packing, buffer allocation, cache
maintenance, the register round trip, and unpacking.

From the repository root, export the two block-zero self-attention query
projections from the recorded incident. Use a Python environment with NumPy
and the reconstructed diagnostic checkpoint.

```sh
python3 -m kit.export_projection build/flan/diagnostic \
  kit/evidence/three-diagnosis-traces-physical.json build/flan
python3 fpga/check_projection.py \
  --system-capture build/flan/projection-encoder.npz \
  --system-capture build/flan/projection-decoder.npz
```

The replay checks every signed score against integer matrix multiplication,
including the padded encoder positions. The AXI testbench also exercises
backpressure, burst boundaries, and a failed write followed by a retry.
Simulation clock counts exclude host setup.

On a board reserved for benchmarking, run these commands for each captured
NPZ. The physical runners acquire the normal service's exclusive lock before
loading an overlay. `--simulate` runs only the software check.

```sh
python3 -m kit.benchmark_projection --bitstream OLD/pinkprocess.bit --capture CAPTURE.npz
python3 -m kit.benchmark_projection_overlay --bitstream NEW/pinkprocess.bit --capture CAPTURE.npz
```

On a kit with the matched overlay and resident language model, the systemd
service reads `/etc/pinkprocess-kit/projection.conf` on restart. To route
every learned matrix through the array, set the file to:

```ini
PINKPROCESS_PROJECTION_ARGS="--projection-overlay --projection-scope all"
```

This includes both attention stacks, feed-forward layers and the vocabulary
projection. Large matrices run in segments to fit the board's contiguous
memory. Each segment uses the currently selected weights. In this mode the
service requires the projection core, and a completed answer reports both
`accelerated_projections` and `fabric_projections`. They count learned matrix
calls, including a segmented matrix as one call. For the initial two-projection
benchmark configuration, set `PINKPROCESS_PROJECTION_ARGS=--projection-overlay`.

On boards with sufficient contiguous memory, add
`--projection-resident-head-mib 16` to retain the vocabulary weights in
FPGA-readable buffers. The driver uses the existing streaming workspace if an
allocation fails, and the active version determines what the FPGA reads.

The [timing-qualified overlay](build/projection-queue-qualified) adds an eight-entry queue and stores each activation tile of
depth up to 1,024 in block RAM for reuse across output tiles. The block-level
simulation covers AXI stalls, queue bounds, interrupted batches, retry, and
the streaming fallback above that depth. To build the optional vocabulary
reducer as well, run `fpga/build.sh BUILD_DIRECTORY --argmax`. On a matched
image, `--projection-queue` batches independent projections and
`--projection-argmax` returns the winning token after applying both FP32
scales. Both require `--projection-overlay --projection-scope all`. The
first output step preserves the full calibration data for edits. Earlier
images continue to support the all-matrix configuration above. For isolated
qualification, add `--hardware-counters` to `kit.benchmark_projection_overlay`
on a queue-capable image to record useful MAC cycles and 64-bit DDR beats.

The installed kit uses the queue, the shared activation cache, and a 16 MiB
vocabulary weight budget. FPGA vocabulary reduction remains opt-in. The
same saved incident completed in 48.289 s cold and 46.944 s warm with
software token selection on the new image; all learned matrix operations
ran on the FPGA. The named edit, rollback, and a new incident each reproduced
their reference answer. [Physical comparison](kit/evidence/physical-projection-queue-comparison.json)

### Remote Vivado build on Western GAUL

On 2026-11-20, the Mac reached Western University's SSH service at
`compute.gaul.csd.uwo.ca`. The non-interactive login returned
`Permission denied (publickey,password)`. Western documents password login
using a Western account. GAUL access, the CPU architecture, Vivado and the
PYNQ-Z2 preset still need to be verified in an authenticated session.
See [Western's Mac SSH guide](https://kb.sci.uwo.ca/doku.php/computerscience:gaul:connectingtogaulmac).

From Mac Terminal:

```sh
ssh YOUR_WESTERN_USERNAME@compute.gaul.csd.uwo.ca
```

Verify the first-use host-key prompt before accepting it, then enter your
Western password in Terminal. Once logged in, check the tools and storage:

```sh
uname -sm
command -v vivado
type module
df -h .
```

If Vivado is missing from the PATH and the host provides environment modules,
run `module avail vivado`. Once Vivado is available, run `vivado -version`
and try the build command above. The build script checks for the PYNQ-Z2
board preset. Save the Vivado version, implementation log, resource and
timing reports, and hashes of the resulting BIT and HWH files.

### Remote Vivado build on AWS Academy

The Learner Lab build host uses Ubuntu 24.04 on an x86-64 `m7i.large`
instance with a 100 GiB gp3 volume in `us-east-1`. SSH ingress is limited
to the Mac's public IPv4 address. Use the instance's current public IP
and the lab's `vockey` PEM key to connect. The project is staged at
`~/pinkprocess` on the instance, and the TUL PYNQ-Z2 board preset archive
is staged at `~/pynq-z2-board-files.zip`.

Icarus Verilog on the instance matched all 17,920 reference scores with
`python3 fpga/check_classifier.py pinkprocess-lm-int8.json`. The AMD
Vivado 2025.2 Linux web installer was copied to EC2, verified against
the Mac copy by SHA-256, and extracted into `~/vivado-installer`.
SSH access was restored by updating the single-address inbound rule
after the Mac changed networks. AMD authentication succeeded on Nov 20,
2026. The batch configuration is `~/vivado-installer/pinkprocess-install.txt`.
It selects Vivado ML Standard with Zynq-7000 device support and installs
under `~/amd`. After approval of the AMD and third-party license terms,
the batch install was started on EC2. Its log is at
`~/vivado-installer/pinkprocess-install.log`. The 17.08 GB download
completed. After the Learner Lab was restarted, the log showed an
archive extraction failure at 75% and a rollback. An earlier
configuration generator had looped on empty input and written a 48 GiB
log, leaving little disk space during extraction. The original payload
was still on disk at `~/amd/Downloads/Vivado_2025.2/payload`. Keep the
install destination at `~/amd` so the web installer can reuse it.
The log was compressed and incomplete installation files were removed.
The retry reused the cached payload and completed. After installing
Ubuntu runtime libraries, `~/amd/2025.2/Vivado/bin/vivado -version`
reported Vivado 2025.2 (SW build 6299465). A batch Tcl check found
`xc7z020clg400-1`, the FPGA device used by the PYNQ-Z2.
The board can stay
disconnected during the build. After the build, transfer the matching
`.bit` and `.hwh` files to the board and load the overlay from Jupyter.

### FPGA verification status

On the Apple Silicon Mac, Icarus Verilog 13.0 simulated the AXI classifier
against 32 traced TinyStories activations plus three replacement and
restoration cases. All 17,920 signed scores matched.

On November 20, 2026, Vivado 2025.2 generated the overlay on EC2 using
the TUL PYNQ-Z2 board preset. The initial 100 MHz route missed setup
timing by 5.991 ns. The 50 MHz rebuild met all specified timing
constraints with 2.071 ns setup slack and 0.095 ns hold slack. It used
1,279 LUTs, 660 slice registers, eight block RAM tiles, and zero DSPs.
Vivado also flagged four negative DDR delay parameters in the supplied
board preset. The corresponding settings are the same in both builds.

The Mac copy in `build/pynq/` includes the matching `.bit` and `.hwh`
files, their SHA-256 manifest, build and implementation reports, and the
two files needed to run the replay test. The timing-failing build is
archived in `build/pynq/100mhz/`. The board previously booted the
PYNQ-Z2 v3.1.1 microSD image and served Jupyter over Ethernet.

On November 20, 2026, the 50 MHz overlay was loaded on the physical
PYNQ-Z2. All four test-file checksums matched the Mac manifest. The
board matched all 17,920 signed scores, including live replacement
and restoration of row 326. The first SSH attempt reported `No Devices
Found` because the privileged Python process lacked the XRT environment.
The root login shell above loaded that environment and the replay
exited successfully.

## Live generation and weight rollback

Run from the project root on the PYNQ-Z2. Copy the `lm_reference/` code,
`fpga/check_classifier.py`, the pinned TinyStories inputs from the
reference instructions, and the matching 50 MHz `.bit` and `.hwh` files.
Adjust the paths below for their location on the board. The script needs
a C compiler and verifies the three upstream input hashes before running.

```sh
sudo bash -lc 'exec /usr/local/share/pynq-venv/bin/python3 -m lm_reference.live_demo \
  --source run.c --model stories260K.bin --tokenizer tok512.bin \
  --bitstream pinkprocess.bit --report live-pynq.json'
```

The ARM processor runs the floating-point transformer and greedy decoder.
For each forward pass, the runtime quantizes 64 activations and sends them
to the Python process over a local socket. Python writes the activations
to the FPGA, starts the classifier, and returns all 512 signed sums.
The runtime applies the input and weight scales before selecting a token.

The demo loads the matrix once, runs the original story, overwrites row
326, generates a fresh story, restores the 64 original bytes, and runs
the original prompt again from a fresh decoder state. It compares every
sum, activation, sampled token, and output byte with the independent C
reference on the same machine. The report records hashes, classifier
round-trip timings including AXI transfers, and generation wall time.
Classifier matrix hashes describe the expected bytes sent to the FPGA.
Readback of the classifier's computed scores verifies the edit and rollback.

On a Mac, use the same module with `python3` and `--software` in place of
`--bitstream pinkprocess.bit` to exercise the live socket and token
generation using Python integer dot products.

### Selective one-row edit

Add `--selective` to either live command. The conventional fit adjusts the
same 64-byte row so Tim wins at the first sampled position. It projects the
row against score constraints from the target, the unedited continuation
after an explicit `Tim`, and two comparison prompts. It rounds the row
back to signed INT8 and checks every constraint on the integer weights.
The live demo checks every hardware score against the C reference and
reconstructs the row with one-hot input vectors in each phase. The edited
run must reproduce the explicit-Tim continuation, and the final run must
reproduce the original story.

| Prompt | Observed change after fitting row 326 |
| --- | --- |
| `Once upon a time, there was a little girl named` | Lily becomes Tim. The next 18 sampled tokens match the unmodified model given Tim. |
| `Once upon a time, there was a little boy named` | Same tokens and text as the original. |
| `Lily and Tim went to the park` | Same tokens and text as the original. |
| `Once upon a time, there was a little girl who` | Same tokens and text as the original. This prompt was excluded from the fit. |
| `One day, a little dog went outside` | Same tokens and text as the original. This prompt was excluded from the fit. |
| `Once upon a time` | Generation reaches the same naming prefix and switches Lily to Tim there, too. |

The original row scored 14,330 for Tim at the target position. The strong
edit scored 177,038 and caused repetition. The selective edit scored
25,921 against Lily's 25,884. The diagnostic prompts cover a fixed test
suite for this checkpoint and greedy decoding. The shared-prefix result
shows that the edit applies whenever generation reaches that token history.

Use `--selective` with `lm_reference/reference.py` to export the fitted row,
its hashes, both sets of token IDs and the comparison outputs with the
reference vectors. The live JSON also records the hardware hashes and
timings by phase.

On November 21, 2026, the physical 50 MHz PYNQ-Z2 matched all 294,912
scores across six prompts and three phases. One-hot probes reconstructed
all 64 row bytes in each phase. The median classifier round-trip was
1.35–1.36 ms, including AXI input writes, polling and score readback.
The row writes took 0.72 ms to apply and 0.73 ms to restore. A 32-step
target generation took 0.93–0.94 seconds on the board. The timings come
from a single run. See `build/pynq/live-pynq-selective.json` for the report
and `build/pynq/live-pynq-verified.json` for the earlier strong edit.

### Apply and roll back a patch on demand

Export a patch bundle from the selective reference vectors generated with
`lm_reference/reference.py --selective`. From the repository root on the Mac,
run the pack command with your vector file.

```sh
python3 -m lm_reference.patch_operator pack \
  --reference build/pynq/selective-reference-vectors.json \
  --out build/pynq/tim.patch.json
```

The bundle identifies the INT8 output row, the base model and quantized
classifier hashes, and the original row hash. It carries the 64 replacement
bytes in hex with their length and SHA-256. Copy the bundle and the current
`lm_reference/` and `fpga/check_classifier.py` to the board alongside the
pinned inputs and matching overlay files.

```sh
sudo bash -lc 'exec /usr/local/share/pynq-venv/bin/python3 \
  -m lm_reference.patch_operator serve \
  --source run.c --model stories260K.bin --tokenizer tok512.bin \
  --bitstream pinkprocess.bit'
```

Wait for the `ready` JSON response, then enter one command per line.

```text
status 326
generate
apply tim.patch.json
verify
generate
rollback
verify
generate
quit
```

`generate` uses the fixed TinyStories prompt and compares the live FPGA
results with the C reference. `status 326` reads back that row, and
`verify` reconstructs and hashes all 512 rows from the board. Commands
return one JSON record each so a client can display the active version
and the result. The process keeps the overlay loaded between commands and
validates each bundle before any patch write. Applying the same bundle
again returns its current status. Roll back before applying a different
bundle.

The operator, live demo, and board replay checker share a lock on the
classifier. Quitting or closing the input restores an active row before
releasing the lock. A new session loads and verifies the pinned base
classifier.

The 50 MHz board rejected a wrong-model bundle and a 63-byte payload.
Board readback preserved the base weights after both rejections and the
edited weights when the short payload was tried again during activation.
The valid bundle produced Lily → Tim → Lily across `generate`, `apply`,
and `rollback`, with all 49,152 scores matching the C reference. Full
matrix readback after rollback matched the original SHA-256. The
session also restored the row when it received `quit` with an active
patch. The command responses are in `build/pynq/live-pynq-operator.jsonl`.

## Full fixed-point TinyStories inference

The fixed-point path runs all five transformer layers on the PYNQ-Z2 ARM
cores with integer arithmetic. It uses per-row signed INT8 projection
weights with Q24 scale factors and Q16 embeddings, normalization gains,
activations, attention and KV caches. RoPE and exponent values are
generated once during export and read from fixed-point lookup tables at
runtime. The output activations are quantized to signed INT8, the FPGA
computes the 512 signed 32-bit scores, and greedy decoding selects the
largest integer score. The BPE tokenizer is shared with the pinned
llama2.c reference.

A Mac check compared seven saved prompts against the floating-point
runtime. Five produced identical sampled token sequences. The explicit
Tim continuation and the little-dog prompt diverged. Bit-exact board
acceptance compares against the integer reference for the same input
and weight version.

Generate the 425,708-byte fixed checkpoint on the Mac from the pinned
floating-point source checkpoint. Export happens offline. The fixed
runtime reads the `.ppq` file directly.

```sh
python3 -m lm_reference.quantize \
  --model build/lm-inputs/stories260K.bin --out build/pynq/stories260K.ppq
python3 -m lm_reference.reproduce \
  --source build/lm-inputs/run.c --model build/pynq/stories260K.ppq \
  --tokenizer build/lm-inputs/tok512.bin --out build/pynq/repro-mac
```

The Mac suite checks the fixed-point reference, replays 17,920 FPGA
scores in Icarus Verilog, and drives the six-prompt live socket test with
Python dot products. For a board run, transfer the source modules
`lm_reference/` and `fpga/check_classifier.py`, the pinned `run.c` and
`tok512.bin`, the `.ppq` model and the matching overlay to the PYNQ-Z2.
Change the paths for the board and run the same suite with the PYNQ
virtual environment.

```sh
sudo bash -lc 'exec /usr/local/share/pynq-venv/bin/python3 \
  -m lm_reference.reproduce --source run.c --model stories260K.ppq \
  --tokenizer tok512.bin --bitstream pinkprocess.bit --out repro-board'
```

The board suite also tests wrong-model and short patches before and
during activation and compares the reconstructed classifier after
rollback. It records the kernel boot ID in `suite.json` and keeps
per-run hashes, raw traces, and the operator transcript. The interactive
`patch_operator serve` and `live_demo` commands take `--fixed` with this
checkpoint. Fixed-mode bundles carry the quantized checkpoint hash.

The suite passed after a verified PYNQ-Z2 reboot. It matched 361,984
scores across board replay, selective live generation, and the patch
operator. It took 119.8 seconds on the board and 129.2 seconds from
upload through report download, measured after boot. The hardware row
write took 0.70 ms to apply and 0.70 ms to restore. See
`build/pynq/repro-board/suite.json` for the verified report and
`build/pynq/cold-repro-timing.json` for transfer timings.

### Historical TinyStories browser operator

To reproduce the early TinyStories weight-edit experiment, use the archived Mac query kit
at `build/pynq/pinkprocess-mac-query.tar.gz` or start the local operator
with `--software`. Enter a story prompt in your browser and generate
with the active weights. See [Mac query instructions](lm_reference/HANDOFF.md#query-tinystories-on-the-mac).

Start the fixed-point browser operator on the PYNQ-Z2. Use the patch
bundle produced by the board suite, or supply multiple `--patch` flags
to make additional saved candidates available.

```sh
sudo bash -lc 'exec /usr/local/share/pynq-venv/bin/python3 \
  -m lm_reference.operator_web --source run.c --model stories260K.ppq \
  --tokenizer tok512.bin --bitstream pinkprocess.bit \
  --patch repro-board/tim.patch.json'
```

The server listens on the board's loopback interface. From your Mac,
forward that port over SSH using the same board user and address you use
for the terminal, then open `http://127.0.0.1:8765`.

```sh
ssh -N -L 8765:127.0.0.1:8765 xilinx@BOARD_ADDRESS
```

**Preview** displays the first divergent token, original and edited
outputs on the six saved prompts, and every row byte. **Apply patch** and
**Roll back** change the live FPGA weights. **Generate story** checks
the live output against the integer C reference, and **Verify weights**
reconstructs the full matrix on the board.

The browser API passed preview, generation, patch application, full
readback, and rollback against the physical board. The transcript is
`build/pynq/fixed-browser-api.json`.

### LM001 four-condition comparison

Prepare the shared authoring packet, conventional patch, and fixed
evaluation protocol. Give the Astra author only `authoring.json`.
Preserve its original JSON response containing a `row` of 64 signed
INT8 integers.

```sh
python3 -m lm_reference.experiment prepare \
  --source build/lm-inputs/run.c --model build/pynq/stories260K.ppq \
  --tokenizer build/lm-inputs/tok512.bin --out build/lm001
python3 -m lm_reference.experiment evaluate \
  --source build/lm-inputs/run.c --model build/pynq/stories260K.ppq \
  --tokenizer build/lm-inputs/tok512.bin --out build/lm001 \
  --astra-response /path/to/raw-astra-response.json
```

The evaluator freezes both candidate hashes and the protocol before
running the original, prompt-only, conventional, and Astra conditions.
It records first-token rank and divergence, all generated tokens and
text, and the target and control counts. On the board, pass
`--bitstream pinkprocess.bit` in the PYNQ virtual environment to check
the same cases through the FPGA. The six target and six control prompts
provide a small evaluation of this checkpoint under greedy decoding.

An evaluator test used two development prompts and a synthetic unchanged
row in the Astra slot. All four paths ran on the PYNQ-Z2 and matched
150,528 FPGA scores against the C reference. The test report is
`build/pynq/lm001-board-fixture.json`.

#### Held-out results

Both weight candidates were frozen before evaluation. Across 48 runs on
the 12 held-out prompts, the PYNQ-Z2 matched 925,696 scores and reproduced
the same token sequences as the software reference. The evaluator
verified the full classifier after applying and rolling back each edit.

| Condition | Tim first on target prompts | Controls with identical sampled tokens |
| --- | ---: | ---: |
| Original model | 0/6 | 6/6 |
| Prompt-only instruction | 0/6 | 0/6 |
| Conventional weight fit | 5/6 | 5/6 |
| GPT-6 Astra weight proposal | 3/6 | 5/6 |

The conventional fit met the fixed criteria of at least four target hits
and five unchanged controls. Astra ranked Tim in the top two on all six
target prompts and selected it first on three. Both edits changed the
same control, a dog named Max in the original continuation, to Tim.
These results cover one checkpoint, one task, and one direct Astra
proposal. The raw results are in `build/lm001/lm001-software.json` and
`build/lm001/lm001-pynq.json`. Candidate provenance is in
`build/lm001/frozen-candidates.json` and
`build/lm001/astra-response.json`.

## Research handoff

The local archive `build/pynq/pinkprocess-pynq-repro.tar.gz` contains
the pinned inputs, 50 MHz overlay, Mac and board runners, browser
operator, and SHA-256 manifest. See [the handoff instructions](lm_reference/HANDOFF.md).

The same archive passed from fresh extraction directories on Mac and
PYNQ-Z2. The Mac and board runs produced identical reference vectors
and patch bundles. Their reports are
`build/pynq/reviewer-kit-mac-suite.json` and
`build/pynq/reviewer-kit-board-suite.json`.

### Hack Western 13 eligibility

The published build window runs from November 20, 2026 at 9:00 p.m. EST
to November 22 at 9:00 a.m. EST. Hack Western allows prior ideas and
public libraries. Significant project development before the opening is
outside the rules. [Official rules](https://hack-western-13.devpost.com/rules) · [Organizer FAQ](https://www.hackwestern.com/)

This checkout includes commits `9532a3b` and `32b40fa` from November
18 and 19. Local evidence also dates the custom INT8 reference and FPGA
classifier RTL to November 20 before the opening. The fixed-point
transformer and browser operator were added during the event window
and depend on that earlier work. The current demo therefore fails
the published build-window requirement as assembled. The kit preserves
this research build for review and reproduction.
