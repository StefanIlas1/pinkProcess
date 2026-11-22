# pinkProcess research notebook and build tracker

Source: [ChatGPT research notebook](https://chatgpt.com/space/page_6af3b6b986c481919b77b397b197bdc2). Snapshot of November 22, 2026.

pinkProcess is a Mac app for asking a paired PYNQ-Z2 about live telemetry. The kit runs the quantized diagnostic model, applies named weight edits, and can restore the original weights. The dated experiments below preserve the progression from the original TinyStories weight-edit demonstration through hardware-accelerated diagnosis.

# Build tracker

- [ ] **Before the event.** Confirm board access, boot image and eligibility of existing assets. Pin the checkpoint, tokenizer and runtime. Choose a name that is one token in context. Record the token ID, tensor shape, quantization scale and whether embeddings share storage with the output row.

Build-window review identified pre-event custom pinkProcess code. See the November 21 eligibility review.

- [x] **Hours 0–4 · Mac feasibility.** Establish deterministic decoding and the base model hash. Make a conventional one-row edit. On development prompts, locate the first divergent token, check its rank and inspect unrelated outputs. If the target changes, log why and freeze a fresh evaluation set.

- [x] **Hours 4–8 · Author comparison.** Completed 48 software and PYNQ-Z2 runs under the frozen four-condition protocol. The raw Astra response and exact quantized patches are saved under `build/lm001/`.

- [x] **Hours 8–14 · FPGA inference.** The PYNQ-Z2 booted its v3.1.1 image and served Jupyter over Ethernet. The AXI classifier RTL and replay test pass simulation. Vivado 2025.2 ML Standard runs in batch mode on AWS EC2 and recognizes `xc7z020clg400-1`. The 50 MHz overlay built and met routed timing. The physical PYNQ-Z2 matched all 17,920 reference scores, including replacement and restoration of one weight row. Live generation on the 50 MHz board matched 294,912 signed scores across six prompts and three phases. The selective edit produced Lily → Tim → Lily and restored all 64 original row bytes. On November 20, 2026, `compute.gaul.csd.uwo.ca` answered SSH. Non-interactive authentication for the Western account was rejected. AWS Academy EC2 became the build host after Western SSH authentication failed. The Waveshare display is deferred. [Western GAUL Mac login guide](https://kb.sci.uwo.ca/doku.php/computerscience:gaul:connectingtogaulmac).

- [x] **Hours 14–21 · Activation and rollback.** The operator validates a versioned JSON bundle before applying a 64-byte change between requests. The physical board rejected the wrong model hash and a truncated payload. Readback preserved the active weights after rejection and restored the full base matrix after rollback. Fresh greedy runs produced Lily → Tim → Lily. Local tests covered retries and interruption during a row write.

- [x] **Hours 21–27 · Operator view.** The local browser operator shows the active version and previews six saved prompts, the first divergent token, both outputs and all 64 row bytes. The HTTP flow exercised preview, apply, generate, verify and rollback on the physical board. The operator now accepts custom story prompts from a Mac browser. Software mode runs the quantized model on the Mac, and board mode uses the PYNQ-Z2 classifier. Both modes returned identical results for the tested custom prompts, including patch application and rollback. The extracted Mac launcher passed an end-to-end API test. The launcher archive is `build/pynq/pinkprocess-mac-query.tar.gz`, SHA-256 `794c0ab85de77aa4313ee8885c1a785b52e54fff0eca304507e401ca6840e62d`. Browser-rendered inspection is pending.

- [ ] **Hours 27–31 · Independent reproduction.** The scripted clean-boot run passed on the PYNQ-Z2. The raw patch is 64 bytes and the versioned JSON is 701 bytes. The recorded apply and restore writes each took about 0.70 ms. The same standalone archive passed from fresh extraction directories on Mac and on the PYNQ-Z2. The two environments generated identical reference vectors and patch bundles. A second developer's run remains to be recorded. The archive is `build/pynq/pinkprocess-pynq-repro.tar.gz`, SHA-256 `2c11756b8e714c64751629c5cfcc5358cce4c71d0362b308b79bd1ab648a8b71`.

- [x] **Hours 31–36 · Integration reserve.** The cold-start suite reproduced the integer demo after a verified board reboot. It ran in 119.8 seconds on the PYNQ-Z2 and took 129.2 seconds from upload through report download.

If Astra scores poorly, report its result and label any conventional patch used in the demo. Record hardware failures and Mac diagnostics in the run log. Mark hardware acceptance complete only after the FPGA run passes.

# Level 1 Mac reference

Level 1 produces a deterministic Mac reference, called **state A**, and integer test vectors for the PYNQ-Z2. FPGA inference will use these vectors to check that its scores match the reference before live patching is enabled.

## 1. Verify the environment

**Completed November 19, 2026.** The Mac reports Apple Silicon (`arm64`), Python `3.15.2`, and Apple Clang `17.0.0`. pinkProcess was at commit `9532a3b` with local telemetry changes. `python3 -m pinkprocess.fixed_point` replayed 14,291 telemetry windows and reproduced the prior 18 decision flips. The language-model smoke test ran in a temporary directory.

```sh
cd ~/pinkProcess
uname -m
python3 --version
clang --version
git status --short
python3 -m pinkprocess.fixed_point
```

## 2. Pin and run the candidate model

**Smoke test passed.** Use the TinyStories `stories260K` checkpoint and its matching `tok512.bin` tokenizer as the first candidate. The checkpoint has hidden size 64 and a vocabulary of 512. Its header indicates that the classifier shares the input embedding weights. For an output-only edit, copy the classifier to its own buffer and verify its baseline against the shared version before quantizing or applying a patch. [Model files and tokenizer](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da/stories260K) · [Runtime source](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb)

```sh
trial=$(mktemp -d /tmp/pinkprocess-lm-reference.XXXXXX)
git clone https://github.com/karpathy/llama2.c.git "$trial/runtime"
git -C "$trial/runtime" checkout 350e04fe35433e6d2941dce5a1f53308f87058eb
curl -fsSL https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da/stories260K/stories260K.bin -o "$trial/model.bin"
curl -fsSL https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da/stories260K/tok512.bin -o "$trial/tok512.bin"
clang -O2 "$trial/runtime/run.c" -o "$trial/run" -lm
"$trial/run" "$trial/model.bin" -z "$trial/tok512.bin" -i 'Once upon a time' -t 0 -n 32 -s 42 > "$trial/a.txt"
"$trial/run" "$trial/model.bin" -z "$trial/tok512.bin" -i 'Once upon a time' -t 0 -n 32 -s 42 > "$trial/b.txt"
cmp "$trial/a.txt" "$trial/b.txt"
shasum -a 256 "$trial/model.bin" "$trial/tok512.bin" "$trial/a.txt"
```

**Observed output.** Both fresh processes generated the same text, beginning “Once upon a time, there was a little girl named Lily.” A diagnostic build logged `next` immediately after `next = sample(sampler, logits);` in `generate()`. Both traces matched across 28 generated token IDs. With this runtime, `-n 32` includes prompt steps and `-t 0` selects greedy decoding.

```text
model SHA-256      b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696
tokenizer SHA-256  037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312
token trace        b913b055681063b2e34a6b2787cb8bdb579c64d38edc5f5748630eee4d2c6d0f
first 8 tokens     432 383 286 261 376 298 315 421
```

## 3. Identify the weight row

- [ ] Encode each proposed name in the exact sentence prefix, with its real whitespace and punctuation. Record prefix IDs and the continuation ID. Choose a name whose continuation adds one token to that prefix.

- [ ] Locate that ID’s classifier row. Export the original 64 values and exact bytes with the dtype, layout and SHA-256. Preserve the tied checkpoint as the source artifact and give the classifier its own runtime buffer. Confirm the unedited copy reproduces the original token trace.

- [ ] Save representative final hidden states and raw scores at the name position. Include prompts where the name belongs and controls where it should be unlikely. Keep the later held-out prompts out of this feasibility check.

## 4. Build and verify the integer reference

- [x] Define the output-layer arithmetic. Use symmetric INT8 from −127 to 127, one scale for the classifier matrix and one per input vector, round ties away from zero, and sum in signed 32-bit integers. Convert sums back to scores on the Mac for token selection.

- [x] Run the Mac INT8 reference twice. The isolated floating-point classifier matched the original text and all 28 sampled tokens. Two fresh INT8 runs matched each other across 28 sampled tokens and 32 forward passes. Independent Python dot products matched all 16,384 C sums.

- [x] Export and simulate the hardware fixtures. Icarus Verilog 13.0 on the Mac matched all 17,920 signed classifier scores. This covers 32 traced TinyStories activations plus three cases for large signed sums, row replacement and restoration. Run `python3 fpga/check_classifier.py /tmp/pinkprocess-lm-int8.json`.

- [x] On development prompts, try a conventional output-row edit. Record whether it changes the intended token, the first place output diverges, and unwanted changes on controls. Restore state A and re-run its check.

**Level 1 status.** The Mac INT8 output-layer reference is reproducible. With the fixed story prompt, editing the 64 signed bytes of output row 326 changed the first sampled token from Lily (317) to Tim (326). Restoring the row returned Lily and reproduced the entire classifier matrix, score trace and token sequence. The intentionally strong edit made Tim repeat, so it establishes reversibility on this prompt. Reproduce using `lm_reference/reference.py --edit-demo` with the pinned model inputs.

# Experiment log

## LM001 Selective output row editing

**Status.** The conventional one-row fit and live board rollback passed the preliminary checks described below. The four-condition evaluation remains open. **Hypothesis:** A quantized one-row edit will increase selection of the target name on intended prompts while keeping unwanted selection on control prompts within a predefined tolerance.

**Before the first run:** Record the target token, prompt IDs, authoring and development split, untouched test split, and acceptable control regression. Define success before inspecting test outputs. Fix decoding, generation length, runtime, model version, and fresh-state procedure. For prompt-only, save the exact extra instruction.

**Four conditions:** Original model, prompt-only, conventional row fit, and one directly proposed Astra row. Give both weight authors the same authoring examples and feature information. Save the allowed tools, original row, scale, hidden-state vectors, instructions and attempt count. Freeze candidates before opening held-out prompts.

**Measurements:** On paired held-out prompts record target selection at the intended position, target rank, first changed token, generated token IDs and unwanted selection on controls. Report numerators and denominators and retain examples where behavior regresses. A selected story demonstrates behavior on one prompt.

## Record for each run

Copy this entry for every attempt, including failed runs. Store raw outputs beside the code in Git or in an artifact directory identified by checksum. Link the evidence here.

- Run ID, date, operator and status

- Hypothesis, protocol version and any deviation with reason

- Git commit and uncommitted diff if applicable, exact command, runtime and hardware

- Model, tokenizer, prompt split and patch identifiers or hashes

- Raw output location, per-example scores and aggregated results

- Observation, interpretation, limitations and next decision

## November 20, 2026 · Vivado build environment

Installed Vivado 2025.2 ML Standard with Zynq-7000 support on an AWS Academy EC2 `m7i.large` running Ubuntu 24.04 x86-64 with a 100 GiB gp3 volume. The first install downloaded 17.08 GB and failed during archive extraction at 75%. An earlier configuration-generator check had looped on empty input and produced a 48 GiB log, leaving little free space. After restarting the expired Learner Lab session, I compressed that log and cleared the partial installation. The retry reused the cached payload and completed.

The first batch launch reported a missing `libpixman-1.so.0`. After installing the required Ubuntu runtime libraries, `~/amd/2025.2/Vivado/bin/vivado -version` returned Vivado 2025.2, SW build 6299465. A batch Tcl check using `get_parts -quiet xc7z020clg400-1` returned the expected device. The installer logs and commands are referenced in the repository README. This checkpoint verifies the build environment. Bitstream generation and on-board score parity are still pending.

## November 20, 2026 · PYNQ-Z2 overlay build

Used Vivado 2025.2 ML Standard (SW build 6299465) on EC2 with the TUL PYNQ-Z2 preset. Vivado rejected the block design name `design` as an HDL keyword, so `fpga/build_pynq.tcl` now names it `pinkprocess_bd`. The 100 MHz route generated a bitstream but missed setup timing by 5.991 ns across 284 endpoints. I set the processor's FCLK0 to 50 MHz and rebuilt. The routed report met all specified constraints with 2.071 ns setup slack and 0.095 ns hold slack. The 50 MHz layout used 1,279 LUTs, 660 slice registers, 8 block RAM tiles and 0 DSPs. Vivado flagged four negative DDR delay parameters supplied by the board preset.

Saved the build log, routed timing report, placed utilization report, overlay and SHA-256 manifest under `build/pynq/` on the Mac. The `.bit` SHA-256 is `ac260b425bbf0e5acc5c5883070f7b03ef6457d8f0d5f716cd041e6e55c1ba93`. The `.hwh` SHA-256 is `6e34e76e9afee7bcc8ab788c863b8984faf84758d857da0915fbe67923297763`. Both hashes matched EC2. The Mac regression still matched all 17,920 scores, including row replacement and restoration. The physical replay result is recorded below.

## November 20, 2026 · Physical classifier replay

The PYNQ-Z2 ran the 50 MHz overlay on the stock PYNQ v3.1.1 microSD image. USB serial confirmed the board's new Ethernet address. I copied the overlay, reference vectors and checker over SSH and verified all four files against the Mac SHA-256 manifest. The first privileged Python invocation reported `No Devices Found` because it lacked the XRT environment. The root login shell loaded the board's profile and XRT setup.

```sh
cd ~/jupyter_notebooks/pinkprocess-50mhz
sha256sum -c SHA256SUMS
sudo bash -lc 'exec /usr/local/share/pynq-venv/bin/python3 check_classifier.py pinkprocess-lm-int8.json --bitstream pinkprocess.bit'
```

The command exited zero and printed `Matched 17,920 scores, including row replacement and restoration.` The checker compared 512 signed integer scores per vector across 32 TinyStories hidden-state vectors and three row-326 cases. Those three cases verify a positive replacement, a negative replacement and restoration of the original row. This establishes classifier parity on the replay vectors.

## November 21, 2026 · Live generation and selective row fit

The physical PYNQ-Z2 ran the pinned TinyStories `stories260K` model with live INT8 output scores from the 50 MHz overlay. The ARM processor ran the floating-point transformer and greedy decoding. The recovered strong-edit run matched 49,152 integer scores against the C reference, then reconstructed and checked all 64 bytes of row 326 in each phase. It changed Lily (317) to Tim (326), repeated Tim, then restored Lily and the original token sequence. Report `build/pynq/live-pynq-verified.json` has SHA-256 `35ded8ae7487198aa876acabf31450e5416eee22d153f105d1dc056e5009f778`.

The conventional fit used 60 score constraints from the target position, the continuation after an explicit Tim, and two development prompts. The strong edit raised Tim's raw score from 14,330 to 177,038. The selective edit raised it to 25,921, just above Lily at 25,884. Tim won and the next 18 tokens matched the unmodified model given an explicit Tim. Both development prompts and both prompts excluded from the fit preserved their original token sequences and text. The shorter `Once upon a time` prompt eventually reached the same naming prefix and changed to Tim there too. This is a small diagnostic suite, with a fresh evaluation split and the Astra comparison still pending.

The board and Mac reports agreed on the exact patch bytes and outputs. The PYNQ-Z2 matched 294,912 signed scores across six prompts and three phases. One-hot probes confirmed the original, edited and restored row bytes. The classifier round-trip medians were 1.363, 1.362 and 1.353 ms. The 64-byte row write took 0.719 ms to apply and 0.728 ms to restore. Each 32-step main story took 0.928–0.941 seconds. These timings came from one run.

Run `python3 -m lm_reference.live_demo` with the pinned `run.c`, `stories260K.bin` and `tok512.bin`, plus `--bitstream pinkprocess.bit --selective --report live-pynq-selective.json`. On the board use the PYNQ virtualenv Python in a root login shell as shown in the repository README. The board report is saved locally as `build/pynq/live-pynq-selective.json`, SHA-256 `5c1308fb7e826952d948c87715e93c742ccbabb0ba39c1f113e8f1731a658adf`. The edited row SHA-256 is `c402ce727e47b4e4c4b254955282f0c9c7136bb91b7a0eaee07bd22f617228f4`. Reports record the pinned input and overlay hashes, row bytes and model outputs. The code was tested from the local checkout at `32b40fa` with the FPGA and language-model work in the working tree.

## November 21, 2026 · On-demand patch activation and rollback

Packed the selective row-326 fit as `build/pynq/tim.patch.json`, SHA-256 `653ad19402e1f3845b4da7380e4fa753efe07c1c76a3f0a4ce00f6d102de897c`. The bundle identifies the INT8 output row and carries the base model, quantized classifier and original row hashes, plus the 64 replacement bytes and their length and hash. The operator verifies these before changing a weight. The session keeps the overlay loaded and accepts `status`, `verify`, `apply`, `generate`, `rollback` and `quit`.

The 50 MHz PYNQ-Z2 rejected a wrong-model bundle and a 63-byte payload. It also rejected the short payload with the valid patch active. Row and full classifier readback confirmed the active weights after rejection. Applying the valid bundle produced Tim and reapplying it returned the same version. Rollback reproduced the original Lily story and the full base classifier SHA-256. Quitting with the patch active also restored the original row. All 49,152 live scores matched the independent C reference. Local tests covered validation, retries, interruption during a write and cleanup on end of input.

The raw operator transcript is `build/pynq/live-pynq-operator.jsonl`, SHA-256 `6a7c941db7225723ac53d0c9f528598035a1c46386f3eaa28da8686d552a1ed7`. It records the board-reconstructed base and edited matrix hashes, command responses, complete generated text and sampled token IDs. The commands to package a candidate and start the operator are in the repository README under **Apply and roll back a patch on demand**.

## November 21, 2026 · Full fixed-point runtime and cold-start verification

The 425,708-byte integer checkpoint has SHA-256 `8b9e81ba42f6906fd21014886757074f93db592eb66ca5e313c75228ed65cd48`. All five transformer layers use integer arithmetic on the PYNQ-Z2 ARM cores, with INT8 projection weights, Q16 activations and KV caches, Q24 projection scales and lookup tables for RoPE and exponentiation. The FPGA consumes INT8 activations and returns 512 integer scores for greedy decoding. The upstream BPE tokenizer remains the shared reference. In seven Mac prompt comparisons, five fixed-point sampled sequences matched the float reference. The explicit Tim and little-dog prompts diverged.

The clean-boot board suite matched 17,920 classifier replay scores, 294,912 live selective scores and 49,152 operator scores. It confirmed Lily → Tim → Lily, rejection of wrong-model and short bundles, and the full original classifier hash after rollback. The new boot ID was `57f80191-68a5-432a-ab04-da22fdd0ea03`. The board archive SHA-256 is `273978d9d976c69f4c680dc56336413baf72640f6e81066145b443d812c9a1d2`. Local evidence is under `build/pynq/repro-board/` and transfer timing is in `build/pynq/cold-repro-timing.json`. The Mac suite took 5.45 seconds. A separate 512-position stress input passed the undefined-behavior sanitizer, and six patch-operator unit tests passed.

The browser operator's physical-board HTTP transcript is `build/pynq/fixed-browser-api.json`, SHA-256 `4f8059b797d1f8653285f434c3bfad6e76a16b14e24a07752b05f64c8cd1bb45`. The CLI and browser both use the same board patch guard and readback.

LM001 v1 now has a shared authoring packet with 60 constraints and a fixed protocol for six target and six control prompts. The protocol SHA-256 is `ce0c6ad2e272296ff1ac904151f6edb57b38b6d5288790d0706823bf0bd0960f`. The evaluator was exercised on disposable prompts. The LM001 prompts were kept separate during authoring.

## November 21, 2026 · Hack Western 13 eligibility review

The event’s build window opens Friday, November 20 at 9:00 p.m. EST and closes Sunday, November 22 at 9:00 a.m. EST. Organizers allow prior ideas and public libraries. They require project development during the event and prohibit continuing previously developed projects. [Official rules](https://hack-western-13.devpost.com/rules) · [Organizer FAQ](https://www.hackwestern.com/)

Git commits `9532a3b` and `32b40fa` are dated November 18 and 19. The local original INT8 reference, custom FPGA classifier RTL and Vivado build script also predate the opening. The newer fixed-point runtime and browser operator rely on that earlier implementation. If this checkout were submitted to Hack Western 13, the earlier custom work would conflict with its published timing requirement. The event organizers decide judging eligibility.

## November 21, 2026 · LM001 held-out comparison

Ran the frozen four-condition protocol across 12 prompts and 48 runs. The PYNQ-Z2 matched 925,696 classifier scores against the integer C reference. Its reports agreed with the Mac on all generated tokens, text, and outcome counts. The evaluator verified the full classifier matrix after applying and rolling back each edit.

| Condition | Tim first on target prompts | Controls with identical sampled tokens |
| --- | ---: | ---: |
| Original model | 0/6 | 6/6 |
| Prompt-only instruction | 0/6 | 0/6 |
| Conventional weight fit | 5/6 | 5/6 |
| GPT-6 Astra weight proposal | 3/6 | 5/6 |

The conventional edit met the fixed criteria of at least four target hits and five unchanged controls. Astra put Tim in the top two on all six target prompts and selected it first on three. Both edits changed the same control, in which a small dog named Max became Tim. The results cover one checkpoint, one task, and one direct Astra proposal.

The raw Mac report is `build/lm001/lm001-software.json`, SHA-256 `7ed276473b4e9a76b0ef5f59a8c4fa73f07126188d3bcf66debd01eebb1e78d4`. The physical board report is `build/lm001/lm001-pynq.json`, SHA-256 `a7c1ee011e9cb9db244926dd319b912c1fe537dacc3f8c634d4e0c790d1088a6`. The author response, both patches, and protocol hashes are under `build/lm001/`.

## November 22 · Live diagnosis on the PYNQ-Z2

The product is the Mac app querying a paired kit while new connection-pool readings arrive. The board snapshots the incident, scores telemetry in programmable logic and generates free-form answers with a resident W8A8 FLAN-T5-small specialist. Its learned matrix multiplications run on the FPGA. The board ARM cores handle tokenization, attention operations and decoding control. The Mac displays readings, generated text and the selected model version. Direct measurement shortcuts and previously saved answers have separate paths. The product latency test times a fresh generation from Ask to the full visible answer.

**Model selection and quality.** TinyStories established that changing physical weights could change an output and that rollback could restore it. Story continuations failed as a diagnostic interface. On twelve synthetic incidents the first diagnostic FLAN candidate met all five answer requirements in 0/12 cases. A replacement evaluated on fresh incidents and question formulations scored 12/12 by direct review and 9/12 by the frozen lexical scorer, which missed three uses of “warns.” Physical replay reproduced three frozen reference answers, including two different hidden causes with identical aggregate readings. The answers identified the ongoing backlog and recommended measuring arrivals and connection hold times. The active cases omitted the predicted end depth. Production accuracy remains unmeasured. [Original blind review](https://github.com/StefanIlas1/pinkProcess/blob/3170880/kit/evidence/blind-diagnosis-review.md) · [Replacement review](https://github.com/StefanIlas1/pinkProcess/blob/3170880/kit/evidence/blind-diagnosis-replacement-review.md) · [Physical traces](https://github.com/StefanIlas1/pinkProcess/blob/3170880/kit/evidence/three-diagnosis-traces-physical.md)

**Weight editing and recovery.** Astra changed 380 bytes in vocabulary row 852 for a saved incident. The first word became “Now,” as requested, and the edit also introduced repetition later. A witnessed power cycle and recovery from a startup overlay conflict reproduced the edited answer. A startup-order fix subsequently passed unattended software restart. Restore original reproduced the original answer while preserving named edits. The installed Mac app also issued a fresh question while the kit ingested telemetry. [Edit and collateral effects](https://github.com/StefanIlas1/pinkProcess/blob/3170880/kit/evidence/physical-diagnosis-edit.json) · [Cold cycle](https://github.com/StefanIlas1/pinkProcess/blob/3170880/kit/evidence/physical-diagnosis-cold-cycle.json) · [Version switching and restart](https://github.com/StefanIlas1/pinkProcess/blob/3170880/kit/evidence/physical-diagnosis-warm-boot.json) · [Native Ask](https://github.com/StefanIlas1/pinkProcess/blob/3170880/kit/evidence/physical-native-ask.json)

**Hardware decision.** The main build incorporates Mohsen’s reference checks before synthesis. The routed 50 MHz overlay retains the telemetry classifier and adds a DDR-fed 8 × 8 signed INT8 array with INT32 accumulation. Isolated encoder and decoder projections improved from 2,100 to 81.5 ms and 43.1 to 12.1 ms respectively (medians of five). Thirteen telemetry submissions started while the array was busy and their forecasts matched. Extending the array to attention, feed-forward layers and the vocabulary head preserved all scores in 842,752 physical comparisons across two shape and weight checks. [Build decision](https://github.com/StefanIlas1/pinkProcess/commit/89be021) · [Array tests](https://github.com/StefanIlas1/pinkProcess/blob/3170880/kit/evidence/physical-systolic-projection.json)

| Physical replay of one saved incident | First observed text | Complete diagnosis |
| --- | ---: | ---: |
| Historical classifier | 334.9 s | 797.5 s |
| Array for two selected query layers | 342.1 s | 830.2 s |
| Array for every learned matrix | 18.1 s | 132.1 s |

Each run used the same question, incident and original weights. All 3,722 projections in the full-array replay used the accelerator and the answer matched byte for byte. Only the two new runs share the same board-local polling protocol. The Mac saw fresh demo telemetry during generation. The prompt contained 186 tokens and generation took 50 steps. At the installed lane count and frequency, the ideal arithmetic-only lower bound is 1.92 seconds for the full answer. [Physical measurements](https://github.com/StefanIlas1/pinkProcess/blob/3170880/kit/evidence/physical-systolic-diagnosis.json) · [Arithmetic floor](https://github.com/StefanIlas1/pinkProcess/blob/3170880/kit/evidence/diagnosis-latency-floor.json)

**Acceptance and next engineering decisions.** The operator target is at most ten seconds to a complete, correct answer on unfamiliar wording while new readings arrive. Keep the checkpoint fixed for latency comparisons. Retain packed weights and reusable buffers, profile the board hot loop, and test native dispatch and queued FPGA work. Measure before changing the lane count or clock. Check each new answer against its own captured readings for present state, cited readings, forecast, uncertainty and next measurement. Track first text, completion, per-case correctness and the slowest case together.

### C++ and packed-weight replay on the installed Mac app

The initial software upgrade kept the 50 MHz bitstream and the diagnostic checkpoint fixed. C++ handled INT8 packing, result unpacking and register dispatch. The board retained packed decoder weights in process memory and reused three physically contiguous work buffers. Model selection and rollback invalidate affected weight layouts. On the paired Mac’s USB backend, the same saved incident completed in 52.784 s after service startup and 51.104 s on a repeat. First text took 15.136 s and 15.024 s. Both answers matched exactly. The warm run made zero new buffer allocations or weight packs, yet it still uploaded about 1.95 GB of weights and spent 22.52 s waiting for FPGA commands. This motivated a separate test of retaining the vocabulary weights in FPGA-readable memory. The 132.089 s comparison used board-local polling.

Three new synthetic incidents with fresh, unscripted-by-the-runtime questions ran through the Mac backend. First text / completion times were 12.145 / 45.953 s (cleared), 13.617 / 48.625 s (active) and 12.081 / 59.393 s (uncertain cause). Each matched its reference answer and passed direct review against its question and captured measurements. The uncertain case explicitly said “The cause is unknown” and recommended checking arrivals and hold times. The original lexical matcher missed that phrasing. Forecast counters advanced during generation. The median completion was 48.625 s and the slowest was 59.393 s; zero of three met ten seconds. These are individual measurements on one board, using the native app’s backend and generated traffic. [Inputs, answers, phase counters and timings](https://github.com/StefanIlas1/pinkProcess/blob/fe750bc/kit/evidence/physical-projection-cache.json)

The new RTL stages up to eight independent projections and reuses activation tiles of depth up to 1,024 in block RAM. The captured encoder projection took 1,044,306 simulated AXI clocks versus 1,935,342 previously. The captured decoder projection took 37,194 versus 40,950. Every signed score, including padding, matched. Simulation covered full queues, invalid writes, backpressure, bus errors, retry and the deeper streaming fallback. A separate vocabulary reducer preserves the model’s two FP32 rounding steps and the earliest index on ties; the first output step still exposes the full data used to calibrate weight edits. New MAC and DDR counters will help choose between further memory changes and more arithmetic lanes. These simulated cycle counts exclude host work.

### Resident vocabulary weights and rollback

The first cache removed repeated packing. The next revision also allowed a bounded 16 MiB of physically contiguous board memory for the vocabulary weights and reused the shared input to grouped projections. All eight output-head segments fit on the test kit (16,449,536 bytes). The warm saved replay reused those segments 400 times with zero uploads of that head. All learned projections still ran on the FPGA. Total weight uploads fell from about 1.95 GB to 1.14 GB for the same 50-step diagnosis.

| Physical Mac USB run | First observed text | Complete answer |
| --- | ---: | ---: |
| Saved incident after startup | 16.003 s | 48.624 s |
| Same saved incident, warm | 14.448 s | 47.345 s |
| Previously saved Astra edit on its own captured incident | 12.753 s | 55.394 s |
| Restore original on that same edit capture | 14.624 s | 49.376 s |
| New synthetic incident with uncertain recovery | 13.249 s | 54.433 s |

All five answers reproduced their references. Switching versions forced the selected head back into FPGA-readable memory. The historical edit also reproduced its known repetitive wording. The unfamiliar recovery case reported five requests waiting, a peak of twelve, the persistence forecast and uncertainty about cause. The original model and both named versions remained available. These measurements use the installed Mac app’s USB backend. [Physical report](https://github.com/StefanIlas1/pinkProcess/blob/ba2cac2/kit/evidence/physical-projection-resident-head.json)

**Native-window trial and memory-traffic limit.** A new on-call question entered through the installed Mac window showed first text in 13.4 s and the full answer in 48.6 s. The answer cited 71 waiting, a peak of 74, a forecast of continued backlog and an undetermined cause. The capture was taken at stream time 328.4 s; the app showed a newer reading at 374.3 s when the answer completed. The release, wording, answer and capture fingerprint are recorded. [Rendered Mac query](https://github.com/StefanIlas1/pinkProcess/blob/f35c56d/kit/evidence/native-window-projection-ask.json)

For the 50-step saved replay, the projection engine reads 2.452 GB of weights. Block-RAM activation reuse reduces calculated activation traffic from 769 MB to 24.9 MB. At one 64-bit read per clock and 50 MHz, weight and activation reads alone require at least 6.19 s under perfect utilization. This excludes score writes, any secondary vocabulary scan, processor work, and transport. The 1.92 s arithmetic-only and 6.19 s read-only bounds can overlap. Reaching two seconds on this workload would require changing the memory path or the work itself. [Reproducible traffic calculation](https://github.com/StefanIlas1/pinkProcess/blob/2a06251/kit/evidence/diagnosis-latency-floor.json)

**Frontend integration history.** Commit `ddb1969` imported the expert-cartridge study and briefly shipped its interface in the Mac bundle. The product uses the dark native kit workspace recovered from main at `13bc182`. The cartridge study retains its own card-based evaluation and kernel vectors. [Dark workspace source](https://github.com/StefanIlas1/pinkProcess/blob/13bc182/pinkprocess/kit.html)

**Native-window diagnosis quality.** An on-call question produced first text in 10.5 seconds and a complete answer in 46.2 seconds. The model reported 27 waiting and a peak of 92 but incorrectly said the queue had cleared. The software reference reproduced the same answer. Compare the model's interpretation with the measured rows. [Captured evidence](https://github.com/StefanIlas1/pinkProcess/blob/c221af3/kit/evidence/native-window-projection-quality-miss.json)

### Timing closure and physical comparison of the command queue

The first vocabulary reducer produced correct outputs in simulation but missed the 50 MHz setup deadline by 9.748 ns after routing. Splitting the exact FP32 arithmetic into pipeline stages preserved first-index ties and passed another 10,000 randomized tests. The combined image met timing with +2.329 ns setup and +0.028 ns hold slack. It uses 15,959 LUTs, 21 block RAM tiles and four DSPs. The existing Academy instance compiled the image and was stopped after the verified artifacts were downloaded. Inference ran on the PYNQ-Z2. [Bitstream, handoff, source hashes and timing](https://github.com/StefanIlas1/pinkProcess/tree/13bc182/build/projection-queue-qualified)

On the physical board the queue matched 1,160 signed scores. Native single-row quantization matched the reference for 4,854 values and their row scales. The FPGA vocabulary reducer selected the exact winner and returned row 4,095 on a tie across segments. The legacy telemetry status and reserved weight row matched. In one captured encoder projection, the array fed the MAC units for 26.7% of elapsed cycles. In the decoder projection it did so for 3.7%. DDR delivered a read beat in 27.3% and 29.9% of elapsed cycles respectively. The next hardware investigation is DMA burst scheduling and idle cycles ahead of any increase in arithmetic lanes.

On the same routed image, the queue with vocabulary reduction completed the saved diagnosis in 49.872 s cold and 46.688 s warm. The queue with software token selection completed it in 48.289 s cold and 46.944 s warm. Every answer matched. The warm difference is smaller than the one-second polling interval. Software selection had a shorter measured vocabulary phase in both runs and retains the simpler path, so the kit enables the queue and keeps FPGA vocabulary reduction opt-in. All learned matrix multiplications still run on the FPGA.

On the selected configuration the archived edit completed in 55.792 s, rollback in 46.768 s, and a fresh synthetic diagnosis in 55.825 s. All three answers matched their references byte for byte. The original weights were selected at the end, both saved versions remained available and the telemetry forecast was fresh. [Physical counter, feature comparison and version evidence](https://github.com/StefanIlas1/pinkProcess/blob/4d0fe17/kit/evidence/physical-projection-queue-comparison.json)

# Evidence for the paper

**Preliminary telemetry experiment · completed on Mac.** On four held-out simulated runs, the Astra-authored logistic profile warned about 451 of 772 queue episodes with 52 false alerts. The conventional fit warned about 454 with 53 false alerts. Fixed-point replay of the Astra profile changed 18 of 14,291 decisions. Reproduction entry point in the pinkProcess repository: `python3 -m pinkprocess.fixed_point`. Scope: synthetic telemetry and a three-feature logistic model.

**Language-model experiment · fixed-point hardware result.** The November 21 entries give model and row hashes, tokens and outputs, rejected-patch results, and full cold-start measurements. The four-condition evaluator ran on the physical PYNQ-Z2 with two development prompts and a synthetic unchanged row. All 150,528 FPGA scores matched the software reference. A separate GPT-6 Astra task proposed one valid 64-value row from the authoring packet. It met 56 of 60 authoring constraints and missed the sole target example. The raw response, patch, and candidate hashes are frozen under `build/lm001/`. The 12-prompt held-out comparison is complete. See the November 21 results for the outcome and report hashes. The fixture report is `build/pynq/lm001-board-fixture.json`, SHA-256 `4b82937a618ee977dcbbcb6822e9ba7924c28d2d30b64db16f62d4aab711fac2`.

**Industrial evaluation · future study.** For a visual inspection pilot, shadow the incumbent model on one line. Compare patch deployment with full-model replacement on good products rejected per 10,000, defects missed, update size, activation time and rollback time.

# Decision history

**Scope clarification, November 19.** The hackathon acceptance test requires the FPGA to use the patched weights in live inference and reproduce the original output after rollback. Log the bitstream, model and patch hashes alongside the raw outputs for all three stages. Keep the four-condition research comparison and record subsequent protocol changes with their rationale.

### Dark Mac workspace and single-pass CSV analysis

The app uploads a connection-pool CSV to the paired PYNQ-Z2. The operator can select a saved language model or synthesize a named INT8 output-row edit, with automatic calibration on the uploaded snapshot. The workspace shows an editable transcript, measured dataset scope, the first sentence as it arrives, excerpts and the verbatim final answer. Those answer views share one generation. Follow-up chat keeps the same evidence and the previous exchange. [Operator walkthrough and sample](https://github.com/StefanIlas1/pinkProcess/blob/528608c/README.md#query-the-pinkprocess-kit)

On the physical kit, calibration and Astra authoring took 54.495 s from job receipt. The edited model took 53.392 s for the first question and 56.929 s for its follow-up. Each answer ran all 4,744 learned projections on the FPGA. The edit caused repetition of “Now,” which is preserved in the raw output. Live telemetry progressed and the global original weights remained active. [Board evidence](https://github.com/StefanIlas1/pinkProcess/blob/a499e01/kit/evidence/physical-csv-workflow.json)

All 125 software tests passed with the local checkpoint fixtures enabled. Separate browser integration tested real HTTP handlers and temporary persisted weights with the numerical hardware and Astra response substituted. The test checked that a failed synthesis saves no version, one question triggers one generation, and browsing the table during streaming keeps the user's place. The packaged arm64 Mac app passed signature and archive checks. [Release](https://github.com/StefanIlas1/pinkProcess/commit/c29f81d)

The installed app's served frontend completed a physical query on the 60-row recovery CSV in 16.92 s. Its first partial arrived at 8.52 s. The original model correctly reported the peak of 12 and its timing but omitted whether the queue cleared. The measured view showed zero waiting at the final reading. A separate browser drove the installed app's own loopback backend, and the FPGA accelerated 1,021 of 1,021 projections. [Rendered views and timing method](https://github.com/StefanIlas1/pinkProcess/blob/dba2dab/kit/evidence/installed-mac-csv-recovery.json)

### November 22 · Open-ended diagnosis and decoder optimization

The operator asked “What’s wrong with my system?” on three 60-reading traces using original weights. The physical kit identified recovery and an ongoing backlog of 19, but called a queue-free window “cleared.” First text took 6.99–9.10 s and completion took 40.39–43.41 s. Generation continued alongside live telemetry in every case. [Inputs, exact answers, and board profile](https://github.com/StefanIlas1/pinkProcess/blob/b6dc9d8/kit/evidence/open-question-three-physical.json)

Combining aligned decoder projections into one hardware launch preserved all three final answers. The recovery case took 40.47 s versus 40.39 s previously, the growing case 41.55 s versus 43.41 s, and the queue-free case 38.86 s versus 42.44 s. The recovery comparison included 121 cold cache misses versus zero in the earlier run. The ten-second complete-answer goal is still open.

A candidate 512-byte output-row edit generated a correct queue-free diagnosis on the development case but failed additional paraphrases or recovered-pool controls. It is experimental. A separately trained W8A8 checkpoint also corrected some development cases but contradicted measurements on independent traces. Neither candidate has been promoted. An experimental decoder now checks groups of proposed tokens with the actual quantized model. On the physical PYNQ-Z2 it reproduced the original answers exactly in 19.71 s (recovery), 23.33 s (growing), and 19.65 s (queue-free). Original completion times were 40.39 s, 43.41 s, and 42.44 s. It also reproduced the original healthy-pool mistake. A separate unseen question and named-weight replay are being checked before deployment. [First hardware optimization](https://github.com/StefanIlas1/pinkProcess/blob/2867928/kit/evidence/open-question-fused-physical.json) · [Verified-block implementation](https://github.com/StefanIlas1/pinkProcess/commit/b5989ff) The app separately displays the measured queue history while the model generates.

A new wording on the physical kit took 24.70 seconds with verified decoding versus 42.43 seconds with scalar decoding. The final text, all 512 calibration activation values, raw score maximum and output-row scale matched. A pre-existing named weight edit reproduced its archived answer. [Three original-weight replays](https://github.com/StefanIlas1/pinkProcess/blob/b56c8d4/kit/evidence/open-question-verified-physical.json) · [Runtime and deployment procedure](https://github.com/StefanIlas1/pinkProcess/blob/79f9c8b/docs/engineering.md)

For diagnosis quality, a separate 360-step W8A8 checkpoint was trained on generated telemetry, with a short measured-history cue derived from the first, peak and latest waiting counts. In the quantized software reference it correctly distinguished all 15 independently frozen queue histories, got all nine active-window endpoint changes correct, and answered three targeted questions. On 12 older incidents it preserved the measured state, peak and primary forecast. Five recovery answers said the queue had cleared without explicitly giving the current count of zero. The alternate is being integrated as a separately selectable checkpoint so the original and its saved versions remain usable. [Independent paired evaluation](https://github.com/StefanIlas1/pinkProcess/blob/416ccd1/kit/evidence/grounded-history-independent-review.json)

Queue history v2 then ran on the physical kit. With original weights selected for the live stream and v2 selected for each uploaded dataset, it identified recovery and the peak of 12 in 30.40 s, reported that waiting rose from 0 to 19 in 33.42 s, and reported no observed backlog with current and peak both zero in 25.36 s. These timings include dataset-specific checkpoint switching and restoration; they are separate from the 19.65–23.33 s original-model timings. The known false recovery on the queue-free upload is corrected on the board. [Exact released model delta](https://github.com/StefanIlas1/pinkProcess/releases/tag/queue-history-v2) · [Reconstruction and training procedure](https://github.com/StefanIlas1/pinkProcess/tree/cc304da/profiles/flan-queue-history-v2)

### Installed Mac app qualification

The production interface connected to the paired PYNQ-Z2, uploaded the 60-reading healthy recording and submitted “What's wrong with my system?” once with Queue history v2. The app showed first text at 11.76 s and completion at 25.42 s. The answer reported zero waiting throughout, identified the next-ten-second warning as a forecast, and recommended measuring arrivals and connection hold times. All 1,021 learned projections ran on the FPGA. The measured panel independently reported that no requests waited in the recorded readings. This run used the original proposal table.

### Queue history v2 on the physical kit with verified token proposals

With v2 already active, the paired Mac submitted “What's wrong with my system?” on three recordings. Recovery after a peak of 12 took 9.78 s to first text and 17.57 s to finish. The growing queue (0 to 19) took 9.34 s and 18.16 s. The healthy queue took 9.38 s and 18.13 s. All three diagnoses matched the prior v2 text byte for byte. During each answer the FPGA processed another 181–186 live windows. The ten-second target has been reached for first text on these recordings. Full answers still take roughly eighteen seconds.

The earlier v2 run used a shared proposal table and switched from the globally selected original checkpoint per dataset. These figures cover both keeping v2 selected and using a table trained on its development examples. Encoder work took 6.29–7.24 s, and decoder stacks took 7.69–8.08 s. All 188 regression tests passed with both real quantized checkpoints available. [Physical measurements](https://github.com/StefanIlas1/pinkProcess/blob/main/kit/evidence/physical-history-verifier-tuned.json) · [Released checkpoint and proposal table](https://github.com/StefanIlas1/pinkProcess/releases/tag/queue-history-v2)

### Saved weight edit on Queue history v2

Astra authored `History v2 Now calibration` against the active v2 checkpoint. The kit verified distinct checkpoint and classifier hashes. Replaying the healthy recording and the exact same question with this version made the answer begin with “Now” and repeat the word later. This demonstrates a real change in generation and also shows how a single-row edit can degrade the wording. Restore original succeeded. Reselecting unmodified v2 reproduced its entire answer byte for byte. All seven catalog entries remained available and v2 is active on the kit. The v2 selection survived an earlier service restart. A physical power-cycle test of v2 remains open.

The source, exact answers, and timing counters are on GitHub main at `5cccb7a`. [Physical edit and rollback evidence](https://github.com/StefanIlas1/pinkProcess/blob/5cccb7a/kit/evidence/physical-history-named-edit-rollback.json) · [Research log](https://github.com/StefanIlas1/pinkProcess/blob/5cccb7a/research.md)

### Mac release and manual test handoff

Pulled the updated frontend and sample library, corrected the Swift shell’s deployment target after the upstream binary required macOS 27.2, and installed the rebuilt signed release on macOS 26.6.2. Built-in queue samples and file uploads now both use the board diagnosis workflow. All 201 backend and regression tests passed, along with JavaScript syntax, native Swift compilation and navigation-origin checks.

In a browser driving the installed app’s production interface and the physical PYNQ-Z2, the bundled “Pool, recovery” recording at capacity 12 produced a correct peak of 87 and recovery to zero (10.21 s first text, 24.65 s complete, 1,094 FPGA projections). Uploading the healthy CSV through the file picker produced zero throughout (10.92 s first text, 19.07 s complete, 656 FPGA projections). Both used Queue history v2. [Testable release and manual instructions](https://github.com/StefanIlas1/pinkProcess/tree/82e73d6)

### Ten-second full diagnosis experiment

The pre-change board completed recovery, growing and healthy uploads in 17.31, 19.22 and 18.29 s. A separate unfamiliar question took 21.71 s. Success means a complete board-generated diagnosis within 10 s, with the same question, data, weights, forecast and generated answer, while live submissions advance. Runs are timed from the Mac app backend and include polling; the user-visible workflow is timed separately.

The candidate packs eight-row tiles with integer NEON instructions, combines aligned FPGA commands and tests prompt-derived guesses verified by the unchanged model. In the quantized software reference, the new decoder used 82 verification passes across 18 independent questions, versus 138 before. Every output token and first-step calibration remained identical. Physical timings and weight version controls are under test. [Reference experiment](https://github.com/StefanIlas1/pinkProcess/blob/db21bb2/kit/evidence/history-prompt-copy-parity.json)

The first combined board configuration (command fusion, prompt proposals, revised host weight cache and NEON tile packing) finished the same four full answers in 15.20, 15.47, 17.28 and 16.50 s. Text, forecast and selected weights matched the pre-change runs, with 157–177 additional live submissions per answer. The packed native library also matched the reference on fourteen signed, padded or strided array layouts and three quantizer checks on the physical ARM processor. The next trial adds the same quantization kernel for multiple rows and batches independent cross-attention queries while retaining each position’s original dot-product shape.
