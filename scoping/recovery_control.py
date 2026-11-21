import glob, io, json, math, os, random, re, sysconfig, urllib.request
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_num_threads(2)
here = os.path.dirname(os.path.abspath(__file__))
resultsPath = os.path.join(here, "recovery_control.json")
seqLen, calTokens, testTokens, steps = 256, 60000, 30000, 122
random.seed(0)
torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True).to(device).eval()
results = {"steps": steps, "lr": 1e-4, "batch": 8, "runs": {}}


def toBlocks(texts, tokenBudget):
    ids = []
    for text in texts:
        ids += tokenizer(text)["input_ids"] + [tokenizer.eos_token_id]
        if len(ids) >= tokenBudget:
            break
    count = min(len(ids), tokenBudget) // seqLen
    return torch.tensor(ids[: count * seqLen]).view(count, seqLen)


def wikiTexts(split):
    pattern = os.path.expanduser(f"~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/wikitext-2-raw-v1/{split}-*.parquet")
    frame = pd.read_parquet(sorted(glob.glob(pattern))[0])
    return [t.strip() for t in frame["text"] if len(t.strip()) > 200]


def financeTexts():
    url = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/sent_train.csv"
    frame = pd.read_csv(io.BytesIO(urllib.request.urlopen(url, timeout=30).read()))
    texts = [re.sub(r"https?://\S+", "", t).strip() for t in frame["text"]]
    return [t for t in texts if len(t) > 30]


def codeTexts():
    paths = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    random.Random(0).shuffle(paths)
    return [open(p, encoding="utf-8", errors="ignore").read()[:20000] for p in paths[:120]]


finance = financeTexts()
random.Random(0).shuffle(finance)
code = codeTexts()
calIds = toBlocks(finance[: len(finance) // 2], calTokens)
tests = {
    "domain": toBlocks(finance[len(finance) // 2 :], testTokens),
    "offDomain": toBlocks(code[60:], testTokens),
    "general": toBlocks(wikiTexts("test"), testTokens),
}
blocks = model.transformer.h
denseState = {k: v.clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def perplexity(batchIds):
    lossSum, count = 0.0, 0
    for i in range(0, len(batchIds), 8):
        ids = batchIds[i : i + 8].to(device)
        lossSum += model(ids, labels=ids).loss.item() * ids.shape[0]
        count += ids.shape[0]
    return round(math.exp(lossSum / count), 2)


@torch.no_grad()
def applyMasks(masks):
    for block, drop in zip(blocks, masks):
        block.mlp.c_fc.weight[:, drop] = 0
        block.mlp.c_fc.bias[drop] = 0
        block.mlp.c_proj.weight[drop, :] = 0


def randomMasks(fraction):
    generator = torch.Generator().manual_seed(0)
    return [torch.randperm(3072, generator=generator)[: int(3072 * fraction)].to(device) for _ in blocks]


def run(label, fraction):
    model.load_state_dict(denseState)
    masks = randomMasks(fraction) if fraction else []
    applyMasks(masks)
    model.eval()
    before = {name: perplexity(ids) for name, ids in tests.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    generator = torch.Generator().manual_seed(0)
    model.train()
    for step in range(steps):
        ids = calIds[torch.randint(0, len(calIds), (8,), generator=generator)].to(device)
        loss = model(ids, labels=ids).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        applyMasks(masks)
    model.eval()
    results["runs"][label] = {"neuronsRemoved": fraction, "before": before, "after": {name: perplexity(ids) for name, ids in tests.items()}}
    print(label, results["runs"][label], flush=True)
    json.dump(results, open(resultsPath, "w"), indent=1)


run("denseFineTuned", 0.0)
run("random0.5FineTuned", 0.5)
run("random0.7FineTuned", 0.7)
