# PYNQ Process research

[Read the full research log](research.md) for the experiments, attribution, failed hypotheses and unresolved questions. It includes the weight-reading studies, scoped pruning, the move to a PYNQ-Z2 and physical tests of the queue-diagnosis workflow. The source is Mohsen Sarrafan Chaharsoughi’s shared research document and the team’s subsequent engineering entries. The private black-box program’s decision records are described in the log.

## What the experiments established

| Question | Result | Limit |
| --- | --- | --- |
| Can readers infer training data from weight differences? | Blind readers identified the training topic for 10 of 12 real fine-tunes. | They identified the specific source in one of four same-topic cases. The study supplied candidate answers. |
| Can readers distinguish a benchmark’s test split from its training split? | Readers identified the split in all seven scored cases. At a 1% planting rate they found one of two leaks. | One reader flagged a clean copy, so the preregistered verdict was ambiguous. The scaled contamination arm remained open. |
| Can unused weights be removed to fit a small board? | The pruning screen found little consistently unused capacity. A reversible finance scope pack improved finance perplexity from 212.3 to 163.6. | A densely fine-tuned control reached 65.7. The pack experiment used one seed. [Scoping experiments](../scoping/) |
| Does FPGA execution preserve the reference answer? | On three frozen traces, the original checkpoint’s verified decoder reproduced the baseline answer byte for byte in 19.65–23.33 seconds. | The baseline called a queue-free trace “cleared.” Numerical parity also reproduced that error. [Physical replays](../kit/evidence/open-question-verified-physical.json) |
| Does the queue-history checkpoint improve diagnosis? | Its software reference identified 15 of 15 independent queue states and nine of nine active endpoint trends. On three board traces, the globally selected checkpoint and tuned proposal table completed in 17.57–18.16 seconds. | Broad answers sometimes omitted secondary forecasts or an explicit zero. The physical comparison covers three recorded histories and a combined workflow change. [Independent review](../kit/evidence/grounded-history-independent-review.json) · [Physical timings](../kit/evidence/physical-history-verifier-tuned.json) |
| Can a named weight edit be reversed? | The physical kit saved a one-row edit. Reselecting the unmodified checkpoint restored its earlier answer byte for byte. | The edited answer repeated its target word. The v2 run covered restart while powered. [Edit and rollback](../kit/evidence/physical-history-named-edit-rollback.json) |

On the 18 frozen queue-history inputs, a later software-reference trial added prompt-derived token proposals and verified up to 16 positions per pass. Verifier blocks fell from 138 to 82. All 18 answers, emitted token IDs and first calibration readings matched the baseline. This trial measured software parity and pass counts. Physical latency for the new mode requires a board run. [Prompt-copy parity record](../kit/evidence/history-prompt-copy-parity.json)

## Find the supporting records

- [Measured results](../docs/results.md) and [engineering reference](../docs/engineering.md) cover the workload, hardware configuration and reproduction procedures.
- [Board evidence](../kit/evidence/) contains the reviewed captures, model outputs, quality misses and comparisons. [Training recipe](../profiles/flan-queue-history-v2/) and [reference implementation](../lm_reference/) cover the released checkpoint.
- [Supplemental experiment records](experiments/) preserve the intermediate protocols, CSV traces, captures, model evaluations and synthesis reports recovered from project work directories. The [experiment guide](experiments/README.md) maps those files to the studies.
- [Historical cartridge study](../docs/earlier-prototype.md) covers the earlier card-based prototype. Its [original specification](https://github.com/StefanIlas1/pinkProcess/blob/b75186a6457ca29208714bda422539796302a9f2/SPEC.md) and [Hack Western plan](https://github.com/StefanIlas1/pinkProcess/blob/b75186a6457ca29208714bda422539796302a9f2/docs/hack-western.md) remain available at the cartridge branch revision.
- [Original design materials](design-history/README.md) include the team plan, simulator, enclosure designs and brand concepts.
