# Replacement model blind diagnosis review

Direct review found that the frozen replacement passed all five requirements on **12 of 12** new synthetic incidents in the W8A8 software reference. The 12 incident snapshots and six question formulations were disjoint from the previous evaluation. The seed was drawn after the candidate and the question set were frozen.

| Criterion | Frozen lexical scorer | Direct answer review |
| --- | ---: | ---: |
| Identify whether the queue has cleared or remains active | 12/12 | 12/12 |
| Cite the current queue count and historical peak | 12/12 | 12/12 |
| Label the ten-second forecast and state its direction | 9/12 | 12/12 |
| Explain that the cause is undetermined | 12/12 | 12/12 |
| Name a concrete next measurement | 12/12 | 12/12 |
| Meet every requirement in the same answer | 9/12 | 12/12 |

Three lexical failures came from the same scorer limitation. Its forecast label pattern missed “warns.” In each case, the model correctly reported that the current queue had cleared and that the FPGA warned requests might start waiting within ten seconds. The exact case IDs are `4972f038c496`, `4fbed0a3beaf`, and `ec87586efe9d`. The scorer and raw results were preserved as run. [Raw inputs, answers, and scores](blind-diagnosis-replacement-w8a8.json)

For both incidents with an active queue that the FPGA predicted would drain, the model kept the present backlog separate from the predicted recovery. Every answer explicitly said the cause was unknown and proposed measuring request arrivals and connection hold times. These results establish the model’s behavior on this synthetic holdout in the software reference. Physical FPGA parity and accuracy on real incidents require separate evidence.
