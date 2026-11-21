# Resident telemetry language model

This directory contains the compressed parameter difference and manifest for the telemetry adaptation of Google's FLAN-T5-small. The encoder reads the operator question and one board telemetry snapshot. The decoder generates one token at a time over 32,128 output rows. All encoder, decoder and vocabulary matrix projections use the PYNQ tiled INT8 fabric backend. The board processor performs tokenization, embedding lookup, scaling, normalization, attention score and value mixing, activation functions and greedy token selection. The checkpoint contains 76,961,152 parameters. Dense weights have signed INT8 values and one FP32 scale per output row. Input activations have one dynamic INT8 scale per vector. [Upstream checkpoint](https://huggingface.co/google/flan-t5-small/tree/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab)

## Reconstruct the exact checkpoint

Run these commands from the repository root on the development machine. The export verifies the original weights and tokenizer digests. Applying the delta verifies the base binary, tokenizer and reconstructed binary before publishing the directory.

```sh
mkdir -p build/flan
uv venv --python 3.12 build/flan/venv
uv pip install -p build/flan/venv/bin/python numpy==2.5.5 sentencepiece==0.2.3 safetensors==0.8.0
src=https://huggingface.co/google/flan-t5-small/resolve/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab
curl -fL "$src/model.safetensors" -o build/flan/source.safetensors
curl -fL "$src/spiece.model" -o build/flan/spiece.model
py=build/flan/venv/bin/python
"$py" -m lm_reference.export_flan build/flan/source.safetensors build/flan/spiece.model build/flan/int8
"$py" -m lm_reference.flan_delta apply build/flan/int8 profiles/flan-telemetry build/flan/resident
shasum -a 256 build/flan/resident/model.json
```

The release manifest SHA-256 is `8ef4344f91f034573a6c39d2d6607a1177ba7deb2404e3b09523ef01388f3807`. The full weights SHA-256 is `3a2792730769bfd3f802597ecf2153cd574a8d1757cc0d89dff57e7bb06fc8f0`. All three reconstructed files belong together. Inference needs NumPy and the board fabric driver. SentencePiece, PyTorch, safetensors, Transformers and the model-download client run on the development machine only.

## Training and evaluation

Training used rank-16 adapters on query, value and feed-forward output matrices, an Apple M4, seed 3717, batch size eight and 620 optimizer steps. The training generator samples synthetic connection-pool readings and pairs questions with factual targets. Inference uses the merged, quantized weights. The following commands reproduce the training procedure. Use the release delta for byte-exact weights across machines.

```sh
uv pip install -p "$py" torch==2.15.0 transformers==4.49.0
mkdir -p build/flan/hf
ln -s ../source.safetensors build/flan/hf/model.safetensors
curl -fL "$src/config.json" -o build/flan/hf/config.json
"$py" -m lm_reference.adapt_flan build/flan/hf build/flan/spiece.model build/flan/trained.safetensors --steps 620 --detailed --seed 3717
"$py" -m lm_reference.export_flan build/flan/trained.safetensors build/flan/spiece.model build/flan/new-candidate --adaptation telemetry-lora-detailed-620-3717
"$py" -m lm_reference.evaluate_flan build/flan/resident build/flan/held-out.json
```

On the fixed ten-question development evaluation, the release met all expected phrases for six questions. It distinguished the recovered queue, quoted the peak and occupancy, identified the demo, described the FPGA estimate and suggested measurements. It confused two questions about historical context with current queue depth. It omitted the starting queue value on a comparison and omitted the forecast's end-of-horizon growth clause. These measurements use the NumPy reference. [Inputs and generated outputs](reference-evaluation.json)

The upstream FLAN-T5 model uses the Apache License, Version 2.0. This derivative adds telemetry-specific adapter training and signed INT8 quantization. [License](LICENSE)
