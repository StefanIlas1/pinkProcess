# Measured results and evidence

Every number on this page comes from a recorded run. The raw records are under `kit/evidence/` and `build/`. The broader research log, including the weight-reading studies, is in [research.md](../research/research.md). How to reproduce the hardware work is in [engineering.md](engineering.md).

## Open-ended pool diagnosis

The physical kit received the same question, “What's wrong with my system?”, for three 60-reading uploads. All runs below used the original model weights. Times run from submission to the rendered final answer.

| Recorded queue | Previous decoder | Decoder verifying token proposals | Original model's answer |
| --- | ---: | ---: | --- |
| Cleared after peaking at 12 | 40.39 s | 19.71 s | Correct recovery and peak |
| Increased to 19 waiting | 43.41 s | 23.33 s | Correct active backlog and count, omitted the increase |
| Zero waiting throughout | 42.44 s | 19.65 s | Correct count, incorrectly implied a past queue |

The faster path generated the exact same text in each case. A separate unfamiliar question took 24.70 s versus 42.43 s, and the first-step calibration values also matched. The app shows the measured first, peak and final queue state alongside the generated interpretation. [Board timings](../kit/evidence/open-question-verified-physical.json) · [Fresh wording and saved weights](../kit/evidence/verified-decoder-weight-controls-physical.json)

A separate W8A8 checkpoint, **Queue history v2**, was trained to use the captured queue history. A software evaluation on frozen incidents found it classified 15 of 15 histories correctly, reported the change in all nine active queues correctly, and answered three targeted questions. It also preserved the current state, historical peak and primary forecast on 12 archived cases. Five of those answers described recovery without explicitly giving the current count of zero. [Paired evaluation](../kit/evidence/grounded-history-independent-review.json)

The installed Mac app also submitted the healthy CSV to the physical kit with Queue history v2 selected. Its answer correctly reported a zero peak, described the warning as a forecast and recommended measuring arrivals and connection hold times. First text appeared at 11.76 s and the final answer at 25.42 s with the shared proposal table. All 1,021 projections ran on the FPGA. [Installed app recording](../kit/evidence/installed-queue-history-v2.json)

### Queue history v2 on the paired PYNQ-Z2

With v2 already active and its checkpoint-specific proposal table installed, the app submitted the same open-ended question once for each recording. The board generated the same complete answers as the earlier v2 run. Across these three questions, first text met the ten-second target and the full diagnosis took about eighteen seconds.

| Recorded queue | First generated text | Complete answer | Live FPGA submissions during answer |
| --- | ---: | ---: | ---: |
| Cleared after peaking at 12 | 9.78 s | 17.57 s | 181 |
| Increased from 0 to 19 | 9.34 s | 18.16 s | 186 |
| Zero waiting throughout | 9.38 s | 18.13 s | 186 |

The previous v2 runs used the shared proposal table and selected v2 per dataset while the global selection was original. The comparison includes both configuration changes. Encoder time was 6.29–7.24 s and decoder-stack time was 7.69–8.08 s on the new runs. [Full answers and counters](../kit/evidence/physical-history-verifier-tuned.json)

Astra also authored a named edit to the v2 output row for “Now”. On the same healthy recording and question, selecting the edit produced “Now” as the first word and repeated it later. The edit demonstrated control over output and degraded the wording. Restore original succeeded, and selecting the unmodified v2 checkpoint then reproduced its entire earlier answer byte for byte. All seven library entries remained available. The board stayed powered. [Named edit and rollback](../kit/evidence/physical-history-named-edit-rollback.json)

## Current status

The FPGA scores live telemetry. A provisioned language-model
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
[Physical replay and weight checks](../kit/evidence/physical-systolic-diagnosis.json).
A later board-runtime update used C++ dispatch and reusable buffers on the
same image. Three new synthetic incidents queried through the installed Mac
app's USB backend took 46.0, 48.6 and 59.4 seconds to complete. All three
answers matched their references and the recorded observations.
[Questions, outputs and timings](../kit/evidence/physical-projection-cache.json).
Keeping the vocabulary weights in DMA buffers then gave a 47.345-second warm
saved replay and a 54.433-second answer to a new synthetic question. A real
named edit and rollback each reproduced the corresponding archived answer.
[Resident weights and version changes](../kit/evidence/physical-projection-resident-head.json).
The installed Mac app also answered an unfamiliar question while new readings arrived. After a witnessed power cycle and recovery from a startup
overlay conflict, the kit reproduced the edited answer exactly. Restoring the
original weights reproduced the baseline and preserved the named edit. With the
startup fix installed, the board also reloaded the selected weights on an
unattended software restart. The Mac app reconnected and restored the original
while retaining both named edits.
[Checkpoint](../profiles/flan-telemetry-diagnostic/README.md) ·
[Held-out results](../kit/evidence/blind-diagnosis-replacement-review.md) ·
[Physical incident results](../kit/evidence/three-diagnosis-traces-physical.md) ·
[Native Mac query](../kit/evidence/physical-native-ask.json) ·
[Physical cold-cycle record](../kit/evidence/physical-diagnosis-cold-cycle.json) ·
[Native version switching and restart](../kit/evidence/physical-diagnosis-warm-boot.json)

A later native-window question exposed an important diagnostic error. The kit
counted 27 requests waiting and a recent peak of 92, but the model called
the queue cleared. First text took 10.5 seconds and the complete diagnosis
took 46.2 seconds. The pinned software reference generated the same error.
Inspect the measured readings alongside each answer.
[Native-window quality record](../kit/evidence/native-window-projection-quality-miss.json)

For the queue-onset predictor, the operator-facing quality measure is how often
a warning appears on the Mac at least one second before a real request queues,
alongside misses and false alarms.

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

[Per-run board results](../build/pool-study/observed/report.json) ·
[Native display timings](../build/pool-study/observed/live-display.json) ·
[Importable workload trace](../build/pool-study/observed/7101/telemetry.csv)
(capacity 4).

### Operator-visible warning test

The [acceptance protocol](../kit/forecast_acceptance.md) defines the goal for
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
See the [delivery results](../build/pool-study/rate-holdout/delivery-report.json)
and [rollback evidence](../build/pool-study/rate-holdout/rollback.json).

To repeat the complete measurement, use a fresh capture directory:

```sh
python3 -m kit.capture_pool build/pool-study/new-delivery --seeds 8401 8402 8403
xcrun swiftc -O kit/MeasureWindow.swift -o build/pool-study/measure-window
python3 -m kit.evaluate_delivery build/pool-study/new-delivery --all \
    --window build/pool-study/measure-window --host BOARD_HOST
```

### Queue outlook while requests are waiting

Import a pool telemetry CSV with the header `time_ms,busy,waiting` and ask
whether the current queue will persist or grow over the next ten seconds. With a
board connected the forecast comes from the FPGA. The first answer
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

[Final board results](../build/backlog-study/final-board-report.json) ·
[Fitted model comparison](../build/backlog-study/confirmation-v1.json) ·
[Example input at capacity four](../build/backlog-study/queued-at-25s.csv)

## Physical board check

On this kit the 201-reading sample contained 103 eligible forecast windows.
The base profile warned on 103 and the alternate profile warned on nine.
All 309 scores from the base, alternate and restored runs matched the Mac
fixed-point reference. The restored board weight hash matched the initial hash.
These counts measure how the profiles differ. Assess the forecast quality with
labeled queue episodes. The recorded responses are in
[the board report](../build/pynq/telemetry-kit-physical.json).

## FPGA verification status

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

## LM001 four-condition comparison

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

### Held-out results

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
operator, and SHA-256 manifest. See [the handoff instructions](../lm_reference/HANDOFF.md).

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

