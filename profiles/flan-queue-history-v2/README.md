# Queue-history diagnostic checkpoint

This optional W8A8 FLAN-T5-small checkpoint interprets connection-pool measurements with a ten-second FPGA forecast. Its encoder receives the same question, readings, and forecast as the original diagnostic model, plus a measured description derived from the first, final, and peak waiting counts. The decoder generates the answer. When the first and last counts differ, the description reports the net change across the captured window. Intermediate fluctuations can occur.

## Reconstruct the released checkpoint

Run these commands at the repository root. First follow the [pinned FLAN-T5-small export instructions](../flan-telemetry/README.md#reconstruct-the-exact-checkpoint) through the creation of `build/flan/int8`. Download the release asset with a GitHub account that can read this repository. The archive contains `model.json` and `weights.delta.z`.

```sh
py=build/flan/venv/bin/python
stage=$(mktemp -d "$PWD/build/queue-history-release.XXXXXXXX")
asset=pynq-process-queue-history-v2-delta.tar.gz
gh release download queue-history-v2 --repo StefanIlas1/pinkProcess --pattern "$asset" --dir "$stage"
printf '%s  %s\n' c9ec81ac1b81db1aa84a6ac774073cddd7e405b07089dba53e8492514962ec61 "$stage/$asset" | shasum -a 256 -c -
tar -xzf "$stage/$asset" -C "$stage"
"$py" -m lm_reference.flan_delta apply build/flan/int8 "$stage" build/flan/queue-history-v2
printf '%s  %s\n' d8d97a3327aae96859b3ea5c165ccc860f06d55fe38a79d03c50da9203b22aeb build/flan/queue-history-v2/model.json | shasum -a 256 -c -
```

The output directory must be new. Reconstruction verifies the input base, tokenizer and output weights. The released weights have SHA-256 `4ab0bbf9e8c249bf20d08fbe1a1d03321d0819164c5571c13f6ac42df2289f9d`. Use the history-aware prompt formatter for this checkpoint. The original diagnostic checkpoint retains its original prompt.

## Train from the floating-point parent

The adaptation started from the merged 840-step **floating-point diagnostic parent**. Its safetensors SHA-256 was `91a1498dc3dc4fe4414d1b7ef684170fe8fb7bb3091782feb2738260a4a68c4a`. For a fresh training workspace, follow the upstream checkpoint setup linked above. In that environment use NumPy 2.5.5, SentencePiece 0.2.3, safetensors 0.8.0, PyTorch 2.15.0 and Transformers 4.49.0. The tested training backend was Apple MPS.

If you need to recreate the parent, place the pinned upstream config and floating-point weights in a Hugging Face model directory and train the diagnostic adapter with its original settings:

```sh
py=build/flan/venv/bin/python
uv pip install -p "$py" torch==2.15.0 transformers==4.49.0
src=https://huggingface.co/google/flan-t5-small/resolve/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab
upstream=$(mktemp -d "$PWD/build/queue-history-upstream.XXXXXXXX")
curl -fL "$src/config.json" -o "$upstream/config.json"
cp build/flan/source.safetensors "$upstream/model.safetensors"
"$py" -m lm_reference.adapt_flan "$upstream" build/flan/spiece.model build/flan/diagnostic.safetensors \
  --steps 840 --diagnostic --seed 83126
```

To train the queue-history version using the parent:

```sh
py=build/flan/venv/bin/python
trial=$(mktemp -d "$PWD/build/queue-history-train.XXXXXXXX")
mkdir "$trial/source"
src=https://huggingface.co/google/flan-t5-small/resolve/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab
curl -fL "$src/config.json" -o "$trial/source/config.json"
cp build/flan/diagnostic.safetensors "$trial/source/model.safetensors"
"$py" -m lm_reference.train_queue_history "$trial/source" build/flan/spiece.model "$trial/merged.safetensors"
"$py" -m lm_reference.export_flan "$trial/merged.safetensors" build/flan/spiece.model "$trial/int8" \
  --adaptation 'experimental measured window cue, 360 steps, seed 4227'
"$py" -m unittest lm_reference.test_queue_history_recipe
```

Seed 4227 selects 360 optimizer steps, eight examples per step, rank-16 additive adapters on query, value and feed-forward output projections, AdamW with learning rate 0.002, and a gradient-norm limit of 1. The task sampler gives diagnosis 16 of 26 slots. Each of the ten targeted question types gets one slot. The generator selects balanced legacy examples with probability 0.18. Otherwise it generates queue-free, pulsed, rising, steady or falling synthetic traces and scores their final windows with the kit's fixed-point reference forecaster. The trainer reads only the model inputs and two forecast profiles. The recipe test checks the exact prompt and target bytes across all 2,880 examples and compares their prompt hashes against the independent evaluation corpus. A full repeat of the 360-step training and export from the pinned floating-point parent on the same Mac reproduced all three release files byte for byte. Training on a different platform can change floating-point results. Use the release delta for the qualified model bytes.

## Optional verified token proposals

The [release](https://github.com/StefanIlas1/pinkProcess/releases/tag/queue-history-v2) includes `queue-history-v2-token-draft.tsv.gz` (SHA-256 `e84e9244b04623ea7924bf1cc027738f5a6b48d5a305a1a4e8f0e49a3da948d2`). To reproduce it, run the following command from the checkout using the training environment.

```sh
python -m lm_reference.flan_draft build/flan/int8/tokenizer.json queue-history-v2-token-draft.tsv.gz --queue-history
```

Install the table in a root-owned location readable by the kit service. Add `--history-token-draft /absolute/path/to/queue-history-v2-token-draft.tsv.gz` alongside `--verified-draft` and `--history-model` in the service command. The table proposes tokens from synthetic development examples. The active quantized model verifies each emitted token. The runtime checks the table digest before loading it.

## Evaluation and limits

On the independent W8A8 software evaluation, the model distinguished all 15 queue histories, cited the correct peak and primary forecast in all 15, and described net change in all nine active histories. It also answered the three targeted occupancy, count and root-cause questions appropriately. Four of the 15 general answers conveyed a zero current queue without explicitly citing the numeric current count. On the twelve archived incidents it correctly identified state, peak and primary forecast throughout, but omitted an explicit current count on five recoveries. Its broad diagnoses generally omit the separate forecast of end-of-horizon queue magnitude. These synthetic tests do not establish accuracy on production incidents. [Frozen inputs](../../kit/evidence/frozen-queue-history-18.json.gz) · [Paired original and candidate answers](../../kit/evidence/grounded-history-independent-review.json)

This checkpoint derives from Apache-2.0-licensed FLAN-T5-small. [License](../flan-telemetry/LICENSE)
