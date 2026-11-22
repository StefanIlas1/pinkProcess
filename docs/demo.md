# Judge demo

Three files, one question each, about ten seconds per file on stage. Every file has 10,000 rows and one planted problem written down in [datasets/README.md](../datasets/README.md#planted-causes), so anyone can check the answer.

## Run it

1. Open the app (or `PYNQ_SIM_BOARD=1 PYTHONPATH=$PWD python3 mac/kit_backend.py` and open the printed address).
2. **Import data**, then pick one of the samples below.
3. Ask **What's wrong with my system?** and open **Table**.

Timing on the command line:

```sh
PYTHONPATH=$PWD python3 kit/demo_benchmark.py datasets/checkout-api-10k.csv datasets/cold-chain-10k.csv datasets/store-sales-10k.csv
```

## The three files

| File | What happened | What PYNQ Process says | Time |
| --- | --- | --- | ---: |
| `checkout-api-10k.csv` | `/checkout` in `eu-west` slows and fails from 14:12 to 14:32 because the database pool backs up | Latency went from 143 ms to 1,928 ms on /checkout in eu-west between 14:12 and 14:31. Likely cause: DB pool wait, which went from 6.4 ms to 1,474 ms at the same moment. Next step: check the database connection pool behind /checkout in eu-west. | 71 ms |
| `cold-chain-10k.csv` | Truck T07's compressor dies at 14:00 and its cargo warms to 14 C. Door openings on T03 are a decoy | Compressor went from 6.57 A to 0.45 A at Nov 19 14:00, then temperature rose to 14 C. T03 is called out as normal handling. Next step: check truck T07's compressor and inspect cargo loaded during the warm window. | 89 ms |
| `store-sales-10k.csv` | Card payments in the north region go down from 12:00 to 17:00 while cash keeps working | Revenue went from 280 per hour to 20.4 per hour on card in north (all 4 stores) between Oct 30 12:00 and Oct 30 17:00. Revenue stayed normal for cash in north. Next step: check the card payment provider for the north stores. | 73 ms |

Times are parse, profile and all four answer styles on an M-series MacBook, measured with the command above.

## Against a cloud model

We gave Claude Haiku 5.5 each raw file and the same question. It read all 10,000 rows in chunks with no code tools, the way someone would paste a file into a chatbot. A separate model (Claude Sonnet 5.5) then graded both answers blind against the planted truth, with the labels shuffled per file, on cause, location, time window, numbers, false claims and usefulness.

| | PYNQ Process | Cloud model reading the file |
| --- | ---: | ---: |
| Time per file | 71 to 89 ms | 37 to 57 s |
| Tokens per file | 0 | 355,000 to 394,000 |
| Data leaves the machine | No | Yes |
| Blind grade, three files | 140 / 180 | 154 / 180 |

Both found the right cause, place and window on all three files, and neither fell for the decoy truck. The cloud model writes longer answers with more explanation, which is where it gains its points. We ran the grade three times while improving the analyst, and the gap closed from 53 points to 14. The headline: nearly the same diagnosis, more than 400 times faster, offline and free.

## What to say

- "Every number you see was computed by code from the file. The model only explains measured facts, so it cannot make up a number."
- "This is 10,000 rows. It answered before I let go of the key."
- "A chatbot needed up to a minute and almost 400,000 tokens to read the same file, and the file left the building."
- "The language model runs on a $150 FPGA. On the board, first words show up in about 9 seconds and the full diagnosis in 14 to 18 seconds, with all 1,021 matrix projections on the fabric."
- "Models are weight files, not bitstreams. Here are three quantized checkpoints, and switching takes a second."

## Board answers without the board

The board twin runs the same board service and INT8 checkpoints on the Mac. With a queue file such as `pool-recovery.csv`, upload to the board and pick a checkpoint under **Skills**. See [developer guide](developer-guide.md#board-twin-on-the-mac).
