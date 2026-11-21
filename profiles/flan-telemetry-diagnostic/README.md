# Quantized FLAN live-diagnosis checkpoint

This adaptation uses the original FLAN-T5-small INT8 base from the
[model reconstruction instructions](../flan-telemetry/README.md). Its
training examples cover observed queue pressure, recovery, a ten-second
hardware forecast, uncertainty about causes and useful follow-up measurements.
Training used rank-16 adapters, 840 steps, seed 83126 and the diagnostic task mix.

From the repository root, after exporting the base:

```sh
py=build/flan/venv/bin/python
"$py" -m lm_reference.flan_delta apply build/flan/int8 profiles/flan-telemetry-diagnostic build/flan/diagnostic-install
shasum -a 256 build/flan/diagnostic-install/model.json
```

The pinned manifest SHA-256 is
`14d7912c66a78edf407bbb272f7aa04b953af920827a9a035275194ec4f2f2f2`.
The independent software evaluation includes the questions, observations,
forecasts, generated text, frozen scoring rules and manual adjudication.
[Held-out diagnosis results](../../kit/evidence/blind-diagnosis-replacement-review.md)
