# pinkProcess spec

This is the design record for Mohsen’s expert-cartridge prototype and its
historical results. The current pinkProcess Mac app and PYNQ-Z2 service use
the live diagnostic backend described in the [project README](README.md).
The board status and next steps below refer to the cartridge kernel.
To reproduce the cartridge study from this checkout, run
`python3 -m cartridge.check`; `python3 -m cartridge.server` serves its
original interface on localhost.


This document records the prototype design, file map, measurements and open research questions. Track and demo notes for Hack Western are in [docs/hack-western.md](docs/hack-western.md).

## Run it

```sh
python3 -m cartridge.build     # train and quantise the cartridges from decks/
python3 -m cartridge.check     # held-out score, writes hw/vectors.txt
python3 -m cartridge.server    # http://127.0.0.1:5320
```

Python 3.10 or later, standard library only. `scoping/` needs torch and transformers and is not needed to run the app.

## The problem

A new hire on a warehouse floor, a first-week care aide or a 20-year-old looking at their first paycheque all need the same thing: the right five steps, now, from something they can trust. A chatbot in the cloud fails each part of that:

- it needs a network and an account;
- it answers outside its competence instead of saying it doesn't know;
- nobody can show what it will say before it says it;
- the same question can get a different answer tomorrow.

The answer has to be offline, bounded to one field, checked before it ships and identical every time.

## Why that leads to an FPGA

Those four requirements describe integer arithmetic on a fixed circuit.

- **Identical every time.** The kernel is 16 accumulators and int8 multiplies. There is no floating point, scheduler or sampling. The C++ kernel reproduces the Python reference on every one of 3,264 scores (`hw/README.md`).
- **Checked before it ships.** A cartridge is 64 KB of integers. `research.md` shows what weight readers can and can't verify from weights alone; a cartridge is small enough to audit in full.
- **Bounded.** Each cartridge has a refuse class trained on the other fields and on off-topic questions. A question outside the field loses to it.
- **Switchable.** The circuit never changes. The expertise is the contents of one block RAM. Writing 64 KB into that RAM turns a first-aid expert into a money expert. A CPU can do the arithmetic too, but only the FPGA makes the claim physical: the logic you audited is the logic that runs, and the only thing that changes between experts is a memory image you can hash.

A GPU box is the wrong shape for this. It is power-hungry, non-deterministic across drivers, and far more than the job needs. The PYNQ-Z2 is $150, runs from a USB cable and holds a cartridge in a tenth of its block RAM.

## What a cartridge is

`cartridges/<name>.json`:

| field | meaning |
|---|---|
| `weights` | 4096 × 16 int8. Row = hashed word, column = card |
| `bias` | 16 integers. Unused slots hold -2^30 |
| `labels` | card ids in column order, ending with `refuse` |
| `cards` | title, steps and the urgent line for each card |
| `scale`, `accumulator_signed_bits` | quantisation scale and the proven accumulator width |

A question is lower-cased, stripped of stop words, stemmed and hashed (FNV-1a mod 4096) into at most 64 `(row, count)` pairs. The kernel adds `count × row` into the accumulators. The best card wins if it beats `refuse`.

In `Auto` mode the device loads each cartridge in turn, runs the same question and keeps the one with the largest margin. That is the domain switch: three memory writes and three runs.

## Deleting weights and getting them back

The first idea was to delete the weights a field never uses. Two measurements on GPT-2 (in `scoping/`, written up in `research.md` section 7) say:

1. There is almost nothing unused to delete: 0.21% of MLP neurons are dead on finance but used by code.
2. Removing half the neurons and retraining on the field does give a smaller, field-only model, at a cost in quality.

So nothing is deleted from the device. The base stays whole in storage, and a field is a small pack laid over it:

- **Cartridges (shipping now):** the whole expert is the pack. Switching rewrites 64 KB.
- **Scope packs (frontier, `scoping/scope_packs.py`):** for a language model, a pack is one bit per neuron saying keep or skip, plus a small low-rank correction trained on the field. The base weights are never written, so unloading a pack restores the base bit for bit (checked by sha256). On GPT-2 a pack is 2.95 MB, 0.6% of the base, and takes finance perplexity from 212.3 to 163.6 while skipping half the MLP neurons. A dense model fine-tuned the same way reaches 65.7, so the pack is not free. Full table in `research.md` section 7.6.

Skipped neurons are the part that matters on the board: decoding is limited by memory bandwidth, and a skipped neuron is a row of weights never read.

## Files added

| path | role |
|---|---|
| `decks/*.json` | the three fields: cards, example questions, keywords |
| `cartridge/features.py` | question to sparse integer features |
| `cartridge/build.py` | fit, quantise with one shared scale, write cartridges |
| `cartridge/runtime.py` | integer kernel reference and the device that swaps cartridges |
| `cartridge/check.py` | held-out score and reference vectors for the kernel |
| `cartridge/server.py` | stdlib web server: the app and `/api/ask` |
| `cartridge/board.py` | PYNQ overlay driver (not yet run on hardware) |
| `app/index.html` | the interface |
| `hw/` | HLS kernel, testbench, reference vectors, build notes |
| `scoping/` | the pruning screen and the scope-pack experiment |

## How it joins the existing code

- `pinkprocess/profile.py` is the smallest cartridge: three weights, one threshold. `fixed_point.py` and `cartridge/check.py` follow the same discipline: quantise, export vectors, require exact replay.
- `pinkprocess/author.py` has a large model write weights directly and compares them with a conventional fit on the same budget. The same comparison applies here: have the model author a cartridge's 64 KB from the deck and score it against `cartridge/build.py` on the held-out questions.
- The pool forecaster can ship as a fourth cartridge type (sensor features instead of words) on the same kernel.

## Measured, 2026-11-21

| check | result |
|---|---|
| held-out questions answered with the right card | 52 of 60 |
| routed to the right field | 56 of 60 |
| off-topic questions refused | 6 of 8 |
| C++ kernel against Python reference | 3,264 of 3,264 scores equal |
| cartridge switch, software replay | about 1 µs for 64 KB |

The held-out questions were also used to pick training settings, so treat the first three rows as optimistic.

## Board status

The FPGA only runs a design after it is compiled into a **bitstream**, the file that configures the chip. Compiling needs Xilinx Vivado and Vitis HLS, which run on x86 Linux or Windows and not on a Mac. That step has not been done.

Two ways to demo on the board:

| mode | needs | what runs where |
|---|---|---|
| Board CPU | nothing new: copy the repo to the board, run `python3 -m cartridge.server --host 0.0.0.0` | same integer arithmetic on the board's ARM processor; the app shows "software replay" |
| FPGA kernel | the bitstream built from `hw/` (steps in `hw/README.md`) | scores computed in the FPGA logic; the app shows "PYNQ-Z2 overlay" and stops if they differ from the reference |

Both give the same answers by construction. Say which one is running.

## Not done

- No bitstream yet. The kernel has only been compiled and replayed as C++.
- `cartridge/board.py` has never touched a board. Its register offsets come from the HLS output.
- The decks were written quickly and have not been reviewed by a clinician, a financial planner or a guide.
- Eight of 60 held-out questions get the wrong card or a refusal. More example questions per card is the fix.
- The app answers from 30 written cards. Free-form answers need a language model with scope packs, which are measured on GPT-2 only, one seed.

## Next steps, in order

1. Serve the app from the board's CPU (`--host 0.0.0.0`) so a laptop needs only a cable and a browser.
2. Build the bitstream on a machine with Vivado and pass `vectors.txt` on the board.
3. Add 10 more questions per card and re-score on fresh held-out questions.
4. Author one cartridge with the `author.py` flow and compare.
5. Run a blind weight reader over a cartridge with one planted bad card, under the rules in `research.md` section 2.
