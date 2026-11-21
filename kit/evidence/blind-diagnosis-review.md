# Blind diagnosis review

The frozen FLAN-T5-small candidate failed the blind diagnosis gate. None of 12 cases met all five requirements. The most consistent gap was uncertainty about the cause. Nine answers named a concrete diagnostic comparison, but zero explicitly stated that the recorded queue and occupancy readings could not establish the cause.

The evaluation used the W8A8 software reference. It covered six cleared queues and six still-active backlogs. Every case had its own independently generated readings and an unfamiliar question. Forecast inputs included a cleared queue with a new onset warning and an active queue predicted to drain. The seed was drawn after the candidate manifest and the evaluation protocol had been frozen. The evidence records the exact model inputs, answers and revision fingerprints. [Raw evidence](blind-diagnosis-w8a8.json)

| Criterion | Frozen lexical scorer | Direct answer review |
| --- | ---: | ---: |
| Identify whether the queue has cleared or remains active | 2/12 | 10/12 |
| Cite both a current measurement and the historical peak | 5/12 | 10/12 |
| Label the ten-second forecast and state its direction | 8/12 | 11/12 |
| Explain that the cause is undetermined | 0/12 | 0/12 |
| Name a concrete next measurement | 0/12 | 9/12 |
| Meet every requirement in the same answer | 0/12 | 0/12 |

The direct review credited “the queue cleared” for current state. It credited either a current queue count or a current occupancy count when paired with the historical peak. There were five answers in each group. The lexical scorer required an explicit current queue count. It also missed “warns” as a forecast label and “compare” as a diagnostic action. All nine credited next steps proposed comparing arrival rates with connection hold times. Three answers omitted major parts of the question. Two gave only a forecast, and one gave only readings.

These results cover synthetic evidence summaries and software inference. They do not establish diagnosis accuracy on real incidents or parity with FPGA inference. The recorded incidents should remain evaluation evidence. Any revised candidate needs a new seed and an independently reviewed holdout.
