# Judge demo

Three synthetic files, one broad question each. Every file has 10,000 rows and one planted problem written down in [datasets/README.md](../datasets/README.md#planted-causes), so anyone can check the answer.

## Run it

1. Open the app (or `PYNQ_SIM_BOARD=1 PYTHONPATH=$PWD python3 mac/kit_backend.py` and open the printed address).
2. **Import data**, then pick one of the samples below.
3. Ask **What's wrong with my system?** and open **Table**. A provisioned board generates an answer above the separately labeled **Mac analysis** findings.

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

## Mac analysis and board answers

Both paths used identical files and question text, with no prior question. Times are from submission to the final result after import. Board timings include one-second polling. These five checks test the facts listed below and are not a general language-model quality score.

| Recording and question | Verified result | Mac analysis | PYNQ-Z2 |
| --- | --- | ---: | ---: |
| Checkout, what's wrong? | Latency spike on `/checkout` in `eu-west` | 27 ms | 33,671 ms |
| Checkout, which region? | `eu-west` | 29 ms | 14,245 ms |
| Checkout, what should I check next? | Database connection pool | 28 ms | 12,711 ms |
| Cold chain, which truck? | T07 | 53 ms | 11,739 ms |
| Store sales, which payment method? | Card | 28 ms | 9,357 ms |

For the broad checkout question, the Mac gave the timing, likely cause and suggested check. The board gave the latency change, endpoint and window, and omitted the cause and next step. For the four focused questions the board gave the requested item directly. The Mac returned a fuller incident report containing the requested item. The app shows both contributions with their source labels.

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

- "Mac analysis computes the numbers from the file. Use those findings and the table to check the generated board answer."
- "This is 10,000 rows. It answered before I let go of the key."
- "A chatbot needed up to a minute and almost 400,000 tokens to read the same file, and the file left the building."
- "The board generates the answer, and the FPGA runs the INT8 projections. For this checkout demo the full board answer took about 34 seconds. The Mac analysis appears while it runs."
- "Models are weight files, not bitstreams. Here are three quantized checkpoints, and switching takes a second."

## Board answers without the board

The board twin runs the same board service and INT8 checkpoints on the Mac. With a queue file such as `pool-recovery.csv`, upload to the board and pick a checkpoint under **Skills**. See [developer guide](developer-guide.md#board-twin-on-the-mac).
