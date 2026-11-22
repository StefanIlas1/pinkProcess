# Replay the historical INT8 checkpoints

Two research snapshots use the same pinned Google FLAN-T5-small base as the deployed models. The first is the early telemetry model used in the [domain study](flan/domain-study.json). The second is the 160-step balanced-history candidate used in the [candidate evaluation](balanced-history.eZp0acjS/evaluation.json) and the [independent comparison](../../kit/evidence/grounded-history-independent-review.json). It produced contradictory diagnoses on some traces. The released Queue history v2 checkpoint has its own [reconstruction and training instructions](../../profiles/flan-queue-history-v2/).

Download the archives from the [historical research checkpoints release](https://github.com/StefanIlas1/pinkProcess/releases/tag/research-checkpoints-2026-11-22). Each archive includes the checkpoint manifest, an exact quantized-weight delta and the upstream Apache License 2.0. The upstream tokenizer and base are shared. The recorded comparison concerns inference with quantized weights. Training optimizer state and floating-point candidates are outside these replay bundles.

| Asset | Archive SHA-256 | Reconstructed weights SHA-256 |
| --- | --- | --- |
| `telemetry-domain-int8-delta.tar.gz` | `7fc0e2b13a1bc922513697977c0b33f5bcd6cf4521645e8f7c29814c0a1ebacb` | `2fe952d7130f7bf61496cbe3919f6dadc3591900b97df90fc70342b9fe5e2d5f` |
| `balanced-history-160-int8-delta.tar.gz` | `7e39e99413884ef105fbbae1dbdaef44091e3d8eed47fad6bff365011f4a8419` | `cdec0fd46d76ed4827e53cc0b06e80385449e895d8f377d784285870ac1f83e7` |

Follow the [pinned base export instructions](../../profiles/flan-telemetry/README.md#reconstruct-the-exact-checkpoint) through creation of `build/flan/int8`. From the repository root, run the following in Bash. Reconstruction checks the base, tokenizer and reconstructed weight digests.

```bash
set -e
stage=$(mktemp -d "$PWD/build/historical-models.XXXXXXXX")
gh release download research-checkpoints-2026-11-22 --repo StefanIlas1/pinkProcess \
  --pattern '*.tar.gz' --dir "$stage"
for name in telemetry-domain balanced-history-160; do
  mkdir "$stage/$name-delta"
  tar -xzf "$stage/$name-int8-delta.tar.gz" -C "$stage/$name-delta"
  build/flan/venv/bin/python -m lm_reference.flan_delta apply \
    build/flan/int8 "$stage/$name-delta" "$stage/$name"
  shasum -a 256 "$stage/$name/weights.bin"
done
```
