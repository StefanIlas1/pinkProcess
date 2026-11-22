# Supplemental experiment records

These files preserve development and qualification runs from November 2026. Directory names retain the original study or run identifiers. JSON and CSV captures include intermediate model outputs and superseded configurations. For the accepted comparisons and their measurement boundaries, see the [reviewed evidence](../../kit/evidence/) and [measured results](../../docs/results.md).

| Records | What to look for |
| --- | --- |
| [Backlog study](backlog-study/) and [pool study](pool-study/) | Workload protocols, seed-level traces, arrival clocks, predictions and candidate comparisons. [Released backlog reports](../../build/backlog-study/) and [pool reports](../../build/pool-study/) hold the companion files already tracked in the repository. |
| [FLAN](flan/), [balanced history](balanced-history.eZp0acjS/) and [history cues](history-cue/) | Domain and arithmetic studies, the balanced checkpoint evaluation and an exploratory synthetic training-data script. |
| [Diagnosis iterations](diagnosis-iteration.u3CiFY1I/) | Captures by source revision covering recovered, growing and healthy queues, verification, checkpoint selection and row editing. |
| [Physical latency](latency-physical.E8uf9oCI/) and [live kit](live-kit/) | Decoder baselines, service and board snapshots, timings, counter readings, verification and rendered-app captures. |
| [PYNQ](pynq/), [projection synthesis](projection-dsp-ooc/), [argmax vectors](argmax-pipe-random.OWxfaX7t/) and [queue synthesis](projection-queue-xc7.rpt) | Early fixed-point qualification, FPGA utilization, timing reports and randomized arithmetic cases. [Released board artifacts](../../build/pynq/) include the companion fixtures. |
| [Quantization replay](native-quantization-replay.json), [replay script](verify-native-quantization.py) and [acceptance trace](kit-acceptance/) | Integer parity and operator acceptance inputs. Historical scripts assume the repository root as their working directory and reference original build paths. |
| [Workbench captures](workbench/) | Local model comparison, Mac UI results and board catalog snapshots from the project’s temporary work area. Local network identifiers in these snapshots are generalized. |

The supplemental archive omits byte-identical copies of files already tracked on the remote. A home-directory prefix in a Mac suite record is shortened to a tilde.

[Replay the archived research checkpoints](checkpoint-replay.md) to reconstruct the quantized models used for the early telemetry domain study and the rejected 160-step balanced-history candidate.
