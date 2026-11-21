# Three-trace diagnosis replay

The frozen diagnostic W8A8 model correctly distinguished a cleared queue from an ongoing backlog in the software reference. All three FIFO simulations had a peak of 19 waiting requests at 3 seconds. The same unfamiliar question asked whether the backlog was active, what the latest and worst counts showed, what the ten-second prediction said, and what to measure next.

| Trace | Observed at 12 s | Ten-second model forecast | Diagnosis answer |
| --- | --- | --- | --- |
| Cleared | 0 waiting, 4 of 8 connections busy | New queue-onset warning | Queue cleared, peak 19, a new queue may form |
| Ongoing | 11 waiting, 8 of 8 busy | Queue stays nonempty and ends with no more than 11 waiting | Backlog ongoing, peak 19, queue predicted to remain nonempty |
| Ambiguous cause | 11 waiting, 8 of 8 busy | Same as ongoing | Identical to ongoing |

The ongoing and ambiguous-cause traces have exactly the same timestamped queue depth and utilization samples. Their private request histories differ. One received 979 requests that held connections for 100 ms each. The other received 211 requests that held connections for 500 ms each. Those histories were excluded from the model input. Every answer said the cause was unknown and proposed measuring request arrivals and connection hold times. The pair demonstrates why this aggregate telemetry cannot separate high arrival volume from long connection holds.

The two active-backlog answers omit the forecast's additional end-of-horizon depth estimate. The simulations stop at the question timestamp, so this run checks how the model reports the forecast. It does not check whether the forecast came true. The synchronized arrivals and holds are controlled synthetic workloads. No physical FPGA inference or production incidents were exercised. [Exact workloads, samples, prompts and answers](three-diagnosis-traces-w8a8.json)

Run from the repository root with the frozen exported checkpoint and a fresh result path.

```sh
build/flan/venv/bin/python -m kit.verify_diagnosis_traces \
  build/flan/diagnostic build/flan/three-diagnosis-traces-w8a8.json \
  --manifest-sha256 14d7912c66a78edf407bbb272f7aa04b953af920827a9a035275194ec4f2f2f2
```
