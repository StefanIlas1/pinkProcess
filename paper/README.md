# Paper

`main.pdf` is the 4-page report. Rebuild it with:

```sh
curl -fsSL https://raw.githubusercontent.com/karpathy/llama2.c/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c -o build/lm-inputs/run.c
curl -fsSL https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da/stories260K/tok512.bin -o build/lm-inputs/tok512.bin
inputs="--source build/lm-inputs/run.c --model build/pynq/stories260K.ppq --tokenizer build/lm-inputs/tok512.bin"
python3 -m paper.experiments.lm001_extended $inputs --lm001 build/lm001 --out paper/experiments/lm001-extended.json
python3 -m paper.experiments.contrastive_fit $inputs --out paper/experiments/contrastive-fit.json
python3 -m paper.experiments.contrastive_sweep $inputs paper/experiments/contrastive-sweep.json
python3 paper/analysis.py
cd paper && tectonic main.tex
```

The extended prompts were committed in d9b261d and the contrastive success criterion in 0891cf9, both before their runs.
