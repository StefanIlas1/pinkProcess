# Reading model weights: research log, findings and open problems

Shared research file for pinkProcess. Written 2026-11-21 by Mohsen Sarrafan Chaharsoughi from the committed records of the black-box research program (local repo `blackbox-research/`, not public). Every number comes from a decision record or an independent verifier report named in this file. Where a result is not verified, it says so.

The file has five parts:
1. the claim and why it matters for pinkProcess;
2. how the experiments are kept honest;
3. every result, in order, including what failed;
4. what is open;
5. a plan for turning a large model's expertise into a tiny fixed-point model on a PYNQ-Z2 FPGA, using reading and writing weights together.

---

## 1. The claim

**Main theorem.** An LLM can read a model's weights as text or numbers, without running the model, and use that reading to tell what the model does, what it was trained on, and how to change it. The model runs only afterwards, to check predictions that were sealed beforehand. Activation-based methods appear only as baselines.

This is the reading half of what pinkProcess already does from the writing side. In `pinkprocess/author.py`, Astra writes a profile's weights directly from grouped training data. `fixed_point.py` then replays those weights bit-exactly in the integer arithmetic an FPGA would use. Our research asks the inverse question: given weights someone else wrote or trained, what can be read out of them?

**Product thesis.** A weights scanner. Given an open-weight fine-tune and its public base, it reports from the weight files alone:
- what the model was trained on;
- whether benchmark test data leaked into training;
- whether it holds a backdoor.

It needs no GPU and runs no forward passes, so it scales to scanning a whole model hub. Closed models are out of scope, because there are no weights to read and no base to compare against.

---

## 2. How the experiments are kept honest

These rules are why the numbers below can be trusted. Reuse them for any pinkProcess experiment where an LLM makes predictions.

1. **The answer key exists before any reader starts, but no reader sees it.** We build the ground truth ourselves, for example by fine-tuning a model on a secret mix. Its sha256 is committed to git before any reader launches.
2. **Readers work blind.** Each reader is a fresh LLM session in its own sandboxed room, holding only the disguised weights and a README.
   - **Disguise:** tensors are renamed and permuted so a reader can't recognise a public model by its file.
   - **Leak checks:** five automatic checks (L1–L5 plus a full scan) run before a reader may act. The room has no network, no memory files and no project names, and the README has no hints.
3. **Readers may not run the model.** They may only do arithmetic on weights. A checker blocks forward-pass steps, such as softmax and simulating a layer. A separate blind auditor then reads every script, and a reader caught simulating the model is ruled ineligible.
4. **Readers must beat real competition.** Every arm has these baselines frozen in git before readers start:
   - weight formulas;
   - a "running the model" reference built from activations;
   - a cheap recipe distilled from earlier readers' methods.
   "Promise" means beating the best of them by a margin set in advance.
5. **Sealed before scored.** Reader answers are committed first. The truth is measured only after every reader is sealed and audited.
6. **An independent verifier** rebuilds models from their recorded draws, re-scores everything and checks each claim. Only then does a label count.
7. **Every judgement call is logged** in `PLAN_ISSUES.md`. Any change made after seeing results counts as a failure.

**Labels:**
- **PROMISE:** beats every baseline by the pre-set margin.
- **AMBIGUOUS:** neither PROMISE nor STOP.
- **SOLVED BY RECIPE:** a cheap formula already does the job.
- **STOP:** below baseline.
- **NOT MEASURED:** too few eligible readers.

---

## 3. Results, in order

### 3.1 Phase 1: interpretability lab (Nov 4–5)

LLM-run pipelines mapped circuits and made weight edits on GPT-2 small and a 1-layer modular-addition transformer.

| stream | result | why |
|---|---|---|
| IOI circuit (GPT-2 small) | **Proven** (narrower thesis) | Faithfulness 0.815–0.844; 23 heads; edits to the negative movers worked as predicted. Editing the name movers failed because backup heads took over. 10 of 23 heads have no explained role. |
| Fact localisation and rank-one edit ("capital of France" → Rome) | Not proven | Localised (0.787 recovery, 0.0964% of params changed). Unfiltered paraphrase efficacy was only 0.48–0.68, and one seed leaked into other capitals. |
| Loop fix (greedy repetition) | Partial | Loops cut by 91.7% with induction kept at 0.972 in v3. The earlier version damaged in-context copying. |
| Tiny modular-addition model | Not proven | The Fourier mechanism was found and phase-rotation edits worked (0.9997). Magnitudes failed: raw-logit R² was 0.05–0.84 on fresh seeds. |

**Lesson:** reading recovers structure (which parts, what algorithm) well, but recovers magnitudes (how big an effect will be) poorly.

### 3.2 Phase 2: three theses (Nov 6–7, verified)

| thesis | result |
|---|---|
| T1: seeing the weights fixes a model with fewer test runs | **Not decidable.** No arm met the held-out goal in 20 passes (0 of 16 readers, 0 of 3 random-search seeds). |
| T2: weights tell you what to change, running tells you how much | **Not supported.** Only part c holds: magnitude error falls with passes, from about 2.1–2.6× to 1.2×. |
| T4: weight reading breaks down above some model size | **No breakdown up to Pythia-410M.** Readers beat the fixed recipe only at 410M. |

**Lesson:** at every size tried, readers find mechanisms such as induction heads. But a simple formula often finds the same thing, so the question became which job reading weights does better than both a formula and running the model.

### 3.3 Phase 3: pilot screen (Nov 7–13, each verified)

| lead | label | key numbers | how it's used now |
|---|---|---|---|
| A. Training archaeology (order checkpoints, date when induction heads formed) | **SOLVED BY RECIPE** | Readers 0.993 and 0.959; formula 0.989. Weight norm alone orders checkpoints perfectly. | Pivot statement only |
| B. Training-data provenance (Pythia-70m, 6 copies, 10 topics) | **PROMISE** | Topic AUC: readers 1.000, best formula 0.821, running the model 0.961. Share error 0.17 against the formula's 0.67. | **Main line** |
| E. Blind decompilation (write a program that matches a tiny network) | PROMISE per auditor, NOT MEASURED per verifier | Readers decompiled 4–5 of 5 networks; zero-query baselines 0. A query-based search caught up within 2–8 queries. | Dropped (prior work exists) |
| C. Unlearning audit (was this fact really removed?) | **AMBIGUOUS** | One reader 0.786 against c-loc 0.646, and that depends on one decoy. Running the model reached 0.882. | Pivot statement only |
| D. Backdoor screening | **AMBIGUOUS** | Detection 0.906 / 0.781 / 0.781 against the best weight baseline's 0.5625. Trigger recovery 0, 1 and 0 against a bar of 3. | Building block (detects, can't name the trigger) |
| F. Fact location and weight edits with no compute | **PROMISE** | Location 0.641 and 0.797 against a formula's 0.234 (running the model 0.781). Edits worked 3 of 4 times against 0, all in input-embedding rows. | Building block |
| G. Changelog (what changed between two versions) | PROMISE by rule | AUC 1.0, localisation 4/4. But a two-line recipe also gets 1.0. | Building block only |

### 3.4 Phase 4: full provenance study P8 (verified: PARTLY CONFIRMED)

27 fine-tuned copies, harder mixtures, near-duplicate topics, decoys, and dataset size varied separately from topic.

| entry | topic AUC | share error (L1) |
|---|---|---|
| LLM readers | 0.944 | 0.572 |
| W* (best frozen weight formula) | 0.969 | 0.452 |
| R (cheap recipe from readers' methods, frozen in advance) | 0.972 | 0.538 |
| B-ushift (output shift) | – | 0.954 |
| A* (running the model) | 0.911 | – |
| O-train (oracle) | 0.993 | – |

- **H1a holds:** readers beat B-ushift on 22 of 27 copies (p 0.0008).
- **H1b fails:** readers beat W* on only 9 of 26.

**Meaning:** on lab-made copies, a cheap formula matches the LLM readers. That counts as a product win (the scanner can be cheap), not a loss for the theorem.

### 3.5 Phase 5: real-world models, Arm W (verified: PROMISE)

12 public GPT-2 fine-tunes from the Hugging Face hub, chosen by a written rule (most-downloaded models with documented training data) and made by strangers. Their data covered:
- dialogue and tweets;
- movie reviews;
- Shakespeare;
- COVID preprints, math and arXiv text;
- image prompts and red-team text;
- WikiText, legal text and lyrics.

Readers saw disguised weight differences plus samples for 19 candidate topics, 10 of them decoys.

| entry | right topic first | AUC |
|---|---|---|
| **LLM readers** | **10 of 12** | **0.990** |
| Recipe R | 7 of 12 | 0.923 |
| Best weight formula (unclipped) | 6 of 12 | 0.888 |
| Running the model (unclipped) | – | 0.778 |

- **Readers vs recipe:** READERS AHEAD OF R (delta 0.067, p 0.031). This reverses P8. On stranger-made models, formulas and the recipe lose ground while readers stay near perfect, so the LLM is where the value is.
- **Metadata control:** a session that saw only the model files' metadata named the right topic 0 of 4 times, so readers weren't recognising models from file details.
- **Verifier caveats:**
  - some readers wrote dataset names such as "IMDB-style" or "WikiText" (flag N);
  - one model card also lists Shakespeare's sonnets, which the plan missed;
  - the record originally reported clipped baselines (fixed above).
- **W-D, source discrimination (key failure):** readers got two samples on the same topic, one from the model's actual training source and one from elsewhere. They picked the true source only 1 of 4 times.

**Meaning:** readers identify the topic of the training data, but not yet which specific dataset on that topic.

### 3.6 Phase 6: benchmark contamination, Arm C9 (verified: AMBIGUOUS)

12 GPT-2 fine-tunes on an instruction carrier:
- 6 contaminated: benchmark test splits planted at 1%, 2% or 5%;
- 3 LEGIT: trained on the same benchmarks' training split, which is allowed;
- 3 CLEAN.

| entry | named the leaked set | told test from train | AUC | false alarms |
|---|---|---|---|---|
| **LLM readers** | **7 of 7** | **7 of 7** | **1.000** | 1 reader on 1 clean copy |
| B-frozen (weight formula) | 5 | 5 | 0.841 | 0 |
| Recipe R | 4 | 6 | 0.905 | 0 |
| A-frozen (running the model) | 4 | 7 | 0.889 | not comparable |

- **By plant size:**
  - 1%: readers caught 1 of 2, every baseline 0 of 2;
  - 2%: readers 2 of 2, formula 0 of 2;
  - 5%: all caught.
- **No LEGIT copy was flagged.** Readers told the test split from the training split of the same benchmark every time. That is the specific-data question that failed in W-D, answered correctly.
- **Why not PROMISE:** the rule required exactly zero false alarms. One reader rated one set 0.65 on a clean copy.
- **Significance:** too few copies. Readers beat the formula 2–0 with 5 ties (p 0.25).
- **Verifier:** rebuilt a copy bit for bit and re-scored with no difference.

### 3.7 Phase 7: scaled contamination, Arm C10 (IN PROGRESS, no results yet)

Pre-registered in `18-amendment-10-contamination-scale.md` and reviewed in `19-review-amendment-10.md`.
- **Design:** 30 copies in 6 folders, with at least 8 CLEAN copies and plants at 0.25%, 0.5%, 1%, 2% and 5%.
- **Rules:** the false-alarm rule is now a rate with a pre-set bar, and sign tests against each baseline run in a fixed sequence.
- **Run mode:** one job at a time, to keep the laptop's CPU free.
- **Status on 2026-11-21:** folders F1 and F2 are built and read (4 readers sealed). F3–F6, scoring and the verifier are still to run. The run is paused by usage limits.
- **Blinding:** truth stays sealed until every reader is in, so no partial numbers exist.

### 3.8 Summary: what works and what doesn't

**Works (verified):**
- Naming the topic of a fine-tune's training data from weight differences alone: 10 of 12 real hub models, beating formulas and running the model.
- Telling a benchmark's leaked test split from its legitimate training split: 7 of 7, and 1-of-2 detection at 1% plants where every baseline gets zero.
- Finding where a fact lives and editing it with no compute (embedding rows).
- Detecting that a backdoor exists: detection AUC up to 0.91.
- Finding structure and mechanisms at every size up to Pythia-410M.

**Doesn't work (yet):**
- Naming the specific source among same-topic datasets (1 of 4).
- Recovering a backdoor's trigger.
- Predicting the magnitude of an edit's effect without running the model.
- Fixing a model with fewer test runs (T1 undecidable).
- Proving that unlearning removed a fact (running the model wins).
- Archaeology and changelog: a formula already solves them.

**Not tested:** models above 410M, open-set provenance with no candidate list, and leak rates below 1% (C10 covers that).

---

## 4. Prior work and who wants this (web scan, 2026-11-16)

**Closest prior work:**
- **Sparks of Weight-Space Programming** (Takehi, Teoh, Zhang, Khattab; blog post, Nov 2026). A frontier LLM reads raw weights of tiny RL policies and names the game; it also edits weights, raising Qwen3.5-0.8B's MATH-500 from 7.9% to 48.1%. This is the same core move as our theorem, and pinkProcess's Astra authoring is the writing half. Nothing in it covers training data or contamination.
- **Diff Interpretation Tuning** (Goel et al., ICLR 2026). Trains an adapter so a model describes its own weight diff. It needs training for each behaviour type and generalises poorly across types.
- **Activation Difference Lens** (Minder, Nanda et al., ICLR 2026). An LLM agent reads activation differences and names the fine-tuning goal at 1B–32B. It runs the model and is the strongest competitor.
- **SELECT** (Morris et al. 2025). Gradient matching on the weight diff picks public documents that resemble the training data. It needs a GPU and a corpus.
- **Contamination detectors** (Min-K%, CoDeC, ConStat, dataset inference, Excess Separability) are all behavioural or activation-based, and rewording leaked questions defeats the behavioural ones. **No weights-only contamination check was found.**
- **Cisco Model Provenance Kit** (2026) identifies a model's base model, not its training data.

**Who's looking for this:**
- **Leaderboards:** the Hugging Face leaderboard flags contamination through user reports, and the Llama 4 test-set accusations show the problem.
- **Model insurers and warranty firms** (Armilla, Munich Re aiSure): they evaluate a model before covering it.
- **EU AI Act Art. 53 compliance:** substantial fine-tuners must publish a data summary, and no verification method exists.
- **Investor and procurement due diligence.**
- **Dataset owners:** watermarks only work if data was marked before release.

---

## 5. Open problems

1. **Scale.** Everything is on 70M–410M models. Does reading weight differences hold at 1B–8B? This needs cloud hardware, or chunked reading of LoRA-sized diffs.
2. **Specific-source attribution.** Separate "our movie reviews" from "someone else's movie reviews". C9 suggests test-vs-train separation works, but W-D failed.
3. **Open-set provenance.** Name the training topic with no candidate list.
4. **Realistic contamination.** Leak rates of 0.25% and below, reworded leaks, and leaks mixed into big multi-task fine-tunes.
5. **Cheap scanner versus LLM reader.** On lab copies a recipe ties the readers; on wild copies it doesn't. What exactly does the LLM add that the recipe misses? Distil it into a cheaper scanner.
6. **Trigger recovery for backdoors**, and magnitude prediction for edits.
7. **Reading written weights.** Can a reader audit weights that another LLM wrote, such as Astra's profiles in pinkProcess, and find planted bias or a hidden rule before deployment? Section 6 relies on this.

---

## 6. Plan: turning a large model's expertise into a tiny FPGA model

**Goal:** a $150 PYNQ-Z2 that acts as a narrow field expert for someone without the training, such as:
- a technician diagnosing equipment;
- a warehouse worker reading sensor patterns;
- a nurse's aide triaging a checklist.

Running a huge model on the board is impossible. The plan is to have a huge model **pour its expertise into a tiny model's weights**, quantise those weights to fixed point, verify every bit, and run it on the FPGA.

### 6.1 Hard limits of the PYNQ-Z2

| resource | Zynq XC7Z020 | consequence |
|---|---|---|
| CPU | 2× Cortex-A9, 650 MHz | Runs Linux, PYNQ and the control loop |
| RAM | 512 MB DDR3, shared | About 300 MB usable for a model |
| Bandwidth | about 1–2 GB/s in practice | The speed limit for any language model |
| On-chip RAM | about 630 KB (BRAM) | Only a tiny model fits fully on chip |
| Compute | 220 DSP48, 53k LUTs | Enough for one INT8 matrix-vector unit with 32–64 lanes |

So "expert" must mean a narrow expert. One device covers one domain.

### 6.2 Three tiers, from what fits best

**Tier 1: authored decision profiles.** This is what pinkProcess already does.
- **What:** a big model reads domain data and writes a small, readable weight profile, as in `author.py`. Examples: a logistic rule, a small decision list or a tiny MLP. It is replayed bit-exactly in fixed point, as in `fixed_point.py`: scale 256 for inputs and weights, scale 65,536 for sums, signed widths checked.
- **Cost:** a handful of multiplies per decision. It runs in nanoseconds on DSPs, uses almost no power, and every weight can be read and audited by a person.
- **Good for:** alarms, forecasts, triage scores and go/no-go calls. These are the decisions an expert makes from sensor or checklist inputs.

**Tier 2: distilled small networks.**
- **What:** small MLP or CNN classifiers (10k–1M parameters) trained on labels the big model generated or vetted. They are quantised to INT8 or INT4 and fully on chip with FINN or hls4ml.
- **Good for:** recognising patterns no hand rule captures, such as vibration signatures, images of a faulty part or ECG shapes.

**Tier 3: a tiny language model for explanation.**
- **What:** a 15M–135M parameter model (llama2.c stories15M to start, then SmolLM2-135M), distilled on the domain's Q&A and quantised to INT4 or ternary. It streams from DDR at roughly 10–40 tokens/s.
- **Qwen2.5-0.5B also fits at INT4** if the vocabulary is trimmed from 152k to about 30k tokens, at about 1–3 tokens/s. Use it only as a demo.
- **Good for:** turning the Tier 1 and 2 decisions into instructions a worker can follow ("check valve B, the pressure trend says the seal is going"). The language model explains; it doesn't decide.

**Design rule:** decisions come from Tier 1 and Tier 2, which are verifiable, bit-exact and auditable. Language comes from Tier 3. Never let the tiny LM make the safety-relevant call.

### 6.3 Pipeline

1. **Pick one domain and one decision.** Define inputs, labels and a scoring rule the way pinkProcess does: event-level metrics, a false-alert budget, and disjoint seeds for fit, validation and test.
2. **Build a curriculum with the big model.** It generates and labels cases, explains edge cases, and writes test cases a human expert signs off. Freeze the test set and commit its hash before any training (section 2, rule 1).
3. **Write or train the weights.**
   - **Tier 1:** the big model authors the profile directly (Astra-style), and it's compared against a conventional fit on the same validation false-alert budget, exactly as `author.py` already does.
   - **Tier 2 and 3:** distil with quantisation-aware training, using Brevitas (exports QONNX) or GPTQ/AWQ for the LM.
4. **Quantise and replay bit-exactly.** Encode with round-half-to-even and widen signed products before adding. Generate reference vectors, as in `profiles/service-b-fixed-point-vectors.json`, including the min and max sum cases. Gate: the fixed-point replay must reproduce every held-out decision of the float model. This is the same "function gate" idea used to check every model copy in the research.
5. **Read the weights before flashing.** This is where our research comes in. Run an LLM reader and the weight formulas over the authored or distilled weights to check:
   - that the profile uses the features it claims, with no hidden dependency on an input that shouldn't matter;
   - that no test case leaked into training (the contamination check from section 3.6);
   - that the weights carry no backdoor-like outlier;
   - for a later version, what changed, using the changelog building block.
6. **Build the hardware.**
   - Vitis HLS kernel: an `m_axi` port for weights, AXI-Lite for control, and one INT8 MAC array reused for every layer.
   - Vivado block design: Zynq PS plus the kernel on an HP port, giving `.bit` and `.hwh` files.
   - Driver: `pynq.Overlay` plus `pynq.allocate`.
   - Toolchain: Vivado/Vitis 2020.2–2022.2 on an x86 Linux machine (it won't run on macOS); the board itself is driven over SSH or Jupyter.
7. **Verify on the board.** Replay the reference vectors through the FPGA and require bit-exact sums and alert bits. Compare layer by layer against the float model for Tier 2 and 3.
8. **Field test with real experts.** Measure agreement between the device and an expert, and between a novice with the device and the expert, on held-out real cases. **This is the number that matters:** does a minimum-wage worker with the device decide as well as the expert?

### 6.4 Why the pairing (writing plus reading) is the edge

- **Writing weights directly** (Sparks; pinkProcess's Astra authoring) makes small expert models cheap to produce. There is no training loop for Tier 1, and the result is readable by construction.
- **Reading weights** (this research) is how you trust what was written or distilled, before it ships to hardware that can't be patched in the field.
- **Fixed-point replay** makes the deployed artifact exactly the artifact that was read and tested, with nothing lost in quantisation.

Together they form a loop: author, quantise, read, verify, flash. Every step is checked, and the expert knowledge ends up in a few kilobytes of integers that anyone can audit.

### 6.5 First concrete milestones

1. Port pinkProcess's three-weight profile to a Vitis HLS kernel on the PYNQ-Z2 and pass the existing reference vectors bit-exactly. That proves the toolchain end to end, in days.
2. Add a reader check (section 6.3 step 5) for authored profiles: plant a hidden rule in one profile, and test whether a blind reader finds it, with a sealed prediction against a formula baseline.
3. Run llama2.c stories15M on the ARM cores, then move its `matmul` to the same HLS MAC array, verified layer by layer.
4. Pick the first real domain and run the full pipeline with one human expert's test set.

---

## Where the evidence lives

All of this is in the local research repo (ask Mohsen for access). The full results history is not in this repo.
- Plans and reviews: `program/14-amendment-8-provenance.md` through `program/19-review-amendment-10.md`.
- Decision records: `program/protocol/P7/decisions.md`, `P8/decisions.md`, `P9/decisionW.md`, `P9/decisionC9.md`.
- Verifier reports: `program/VERIFY_P7*.md`, `VERIFY_P8.md`, `VERIFY_P9.md`, `VERIFY_P9C9.md`.
- Rulings: `program/PLAN_ISSUES.md`.
- Earlier phases: `CONCLUSION.md`, `paper/paper.md`, `paper/addendum-studies.md`.

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
