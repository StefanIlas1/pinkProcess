# Physical replay of three diagnosis traces

The paired PYNQ-Z2 reproduced all three frozen software-reference answers exactly. The runner verified authenticated telemetry ingress, captured observations and integer forecasts, each capture's SHA-256, the original checkpoint, and programmable-logic projection counts. It released each labeled synthetic producer immediately after capture to let the SQLite demo publish during generation.

| Trace | Requests waiting at capture | PL projections | First observed partial answer | Question to completed answer | Ingress to completed answer |
| --- | ---: | ---: | ---: | ---: | ---: |
| cleared | 0 | 3,576 | 316.9 s | 773.2 s | 774.6 s |
| ongoing | 11 | 3,722 | Unobserved | Cross-clock handoff | Cross-clock handoff |
| ambiguous-cause | 11 | 3,722 | 334.9 s | 797.5 s | 797.8 s |

The cleared trace reported an empty current queue separately from the risk of a new queue. The two active traces reported 11 waiting requests and a peak of 19. Their aggregate samples were identical even though their hidden arrival rates and connection hold times differed. All three answers left the cause open and proposed measuring arrivals and hold times. The active-trace answers omitted the forecast's end-of-horizon depth estimate.

Elapsed times include polling and transport on the connection used for each trace. The time to a partial answer is when the client first saw generated text. These were scripted synthetic incidents on one board. Forecast correctness at the horizon and diagnosis accuracy on production incidents were outside the run. [Raw physical report](three-diagnosis-traces-physical.json) · [Frozen software traces](three-diagnosis-traces-w8a8.json)

The Mac USB relay completed the cleared case and captured the ongoing case before handoff. The board-local process adopted the same in-flight job and completed the ambiguous-cause case over local HTTPS. The last Mac observation saw ongoing still running at 134.1 s. The board observed completion 616.0 s after adoption. The clients used separate monotonic clocks, so the full ongoing latency is unavailable.
