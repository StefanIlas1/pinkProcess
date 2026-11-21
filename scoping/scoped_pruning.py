import glob, io, json, math, os, random, re, sysconfig, time, urllib.request
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

startTime = time.time()
budgetSeconds = float(os.environ.get("BUDGET_SECONDS", "480"))
here = os.path.dirname(os.path.abspath(__file__))
resultsPath = os.path.join(here, "results.json")
seqLen, calTokens, testTokens = 256, 60000, 30000
sparsities = [0.3, 0.5, 0.7, 0.9]
random.seed(0)
torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True).to(device).eval()
results = {"model": "gpt2", "device": device, "seqLen": seqLen, "perplexity": {}, "notes": []}


def save():
    results["elapsedSeconds"] = round(time.time() - startTime, 1)
    json.dump(results, open(resultsPath, "w"), indent=1)


def toBlocks(texts, tokenBudget):
    ids = []
    for text in texts:
        ids += tokenizer(text)["input_ids"] + [tokenizer.eos_token_id]
        if len(ids) >= tokenBudget:
            break
    count = min(len(ids), tokenBudget) // seqLen
    return torch.tensor(ids[: count * seqLen]).view(count, seqLen)


def wikiTexts(config, split):
    pattern = os.path.expanduser(f"~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/{config}/{split}-*.parquet")
    frame = pd.read_parquet(sorted(glob.glob(pattern))[0])
    return [t.strip() for t in frame["text"] if len(t.strip()) > 200]


def financeTexts():
    try:
        url = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/sent_train.csv"
        raw = urllib.request.urlopen(url, timeout=30).read()
        frame = pd.read_csv(io.BytesIO(raw))
        texts = [re.sub(r"https?://\S+", "", t).strip() for t in frame["text"]]
        texts = [t for t in texts if len(t) > 30]
        results["domainSource"] = "zeroshot/twitter-financial-news-sentiment sent_train.csv"
        return texts
    except Exception as error:
        results["notes"].append(f"finance download failed: {error!r}; wikitext-103 keyword fallback")
        words = re.compile(r"\b(bank|stock|investor|shares|economy|economic|financial|inflation|profit|revenue|billion|market|currency|debt)\b", re.I)
        texts = [t for t in wikiTexts("wikitext-103-raw-v1", "train")[:200000] if len(words.findall(t)) >= 4]
        results["domainSource"] = "wikitext-103 train filtered by finance keywords"
        return texts


def codeTexts():
    paths = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    random.Random(0).shuffle(paths)
    return [open(p, encoding="utf-8", errors="ignore").read()[:20000] for p in paths[:120]]


finance = financeTexts()
random.Random(0).shuffle(finance)
code = codeTexts()
pools = {
    "domain": (toBlocks(finance[: len(finance) // 2], calTokens), toBlocks(finance[len(finance) // 2 :], testTokens)),
    "offDomain": (toBlocks(code[:60], calTokens), toBlocks(code[60:], testTokens)),
    "general": (toBlocks(wikiTexts("wikitext-2-raw-v1", "train"), calTokens), toBlocks(wikiTexts("wikitext-2-raw-v1", "test"), testTokens)),
}
results["tokens"] = {name: {"calibration": cal.numel(), "test": test.numel()} for name, (cal, test) in pools.items()}
blocks = model.transformer.h
linearNames = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
linears = {f"{i}.{n}": block.get_submodule(n) for i, block in enumerate(blocks) for n in linearNames}
denseState = {k: v.clone() for k, v in model.state_dict().items()}


def restore():
    model.load_state_dict(denseState)


@torch.no_grad()
def perplexity(batchIds):
    lossSum, count = 0.0, 0
    for i in range(0, len(batchIds), 16):
        ids = batchIds[i : i + 16].to(device)
        lossSum += model(ids, labels=ids).loss.item() * ids.shape[0]
        count += ids.shape[0]
    return round(math.exp(lossSum / count), 2)


def evaluate(label):
    results["perplexity"][label] = {name: perplexity(test) for name, (cal, test) in pools.items()}
    print(label, results["perplexity"][label], round(time.time() - startTime), flush=True)
    save()


@torch.no_grad()
def calibrate(batchIds):
    inputSquares = {k: 0 for k in linears}
    neuronAbs, neuronFire, tokenCount, hooks = {}, {}, 0, []
    for key, layer in linears.items():
        hooks.append(layer.register_forward_hook(lambda m, inp, out, key=key: inputSquares.__setitem__(key, inputSquares[key] + inp[0].float().pow(2).sum((0, 1)))))
    for i, block in enumerate(blocks):
        def actHook(m, inp, out, i=i):
            neuronAbs[i] = neuronAbs.get(i, 0) + out.abs().sum((0, 1))
            neuronFire[i] = neuronFire.get(i, 0) + (out > 0.05).float().sum((0, 1))
        hooks.append(block.mlp.act.register_forward_hook(actHook))
    for i in range(0, len(batchIds), 16):
        ids = batchIds[i : i + 16].to(device)
        model(ids)
        tokenCount += ids.numel()
    for hook in hooks:
        hook.remove()
    return {
        "inputNorm": {k: (v / tokenCount).sqrt() for k, v in inputSquares.items()},
        "meanAbs": torch.stack([neuronAbs[i] / tokenCount for i in range(len(blocks))]),
        "fireRate": torch.stack([neuronFire[i] / tokenCount for i in range(len(blocks))]),
    }


@torch.no_grad()
def pruneUnstructured(sparsity, inputNorm):
    zeros, total = 0, 0
    for key, layer in linears.items():
        weight = layer.weight
        if inputNorm is None:
            score = weight.abs()
            threshold = score.flatten().cpu().kthvalue(int(score.numel() * sparsity)).values.item()
            mask = score > threshold
        else:
            score = weight.abs() * inputNorm[key][:, None]
            keep = weight.shape[0] - int(weight.shape[0] * sparsity)
            mask = torch.zeros_like(weight, dtype=torch.bool)
            mask.scatter_(0, score.topk(keep, dim=0).indices, True)
        weight.mul_(mask)
        zeros += (weight == 0).sum().item()
        total += weight.numel()
    assert abs(zeros / total - sparsity) < 0.01, (zeros / total, sparsity)
    return zeros / total


@torch.no_grad()
def pruneNeurons(fraction, meanAbs, generator=None):
    removed, masks = 0, []
    for i, block in enumerate(blocks):
        width = block.mlp.c_fc.weight.shape[1]
        dropCount = int(width * fraction)
        if generator is None:
            score = meanAbs[i] * block.mlp.c_proj.weight.norm(dim=1)
            drop = score.argsort()[:dropCount]
        else:
            drop = torch.randperm(width, generator=generator)[:dropCount].to(device)
        block.mlp.c_fc.weight[:, drop] = 0
        block.mlp.c_fc.bias[drop] = 0
        block.mlp.c_proj.weight[drop, :] = 0
        deadRows = (block.mlp.c_proj.weight.abs().sum(1) == 0) & (block.mlp.c_fc.weight.abs().sum(0) == 0) & (block.mlp.c_fc.bias == 0)
        assert deadRows.sum().item() == dropCount, (deadRows.sum().item(), dropCount)
        removed += dropCount
        masks.append(drop)
    return removed, masks


evaluate("dense")
stats = {name: calibrate(cal) for name, (cal, test) in pools.items()}
domainRate, codeRate, generalRate = (stats[n]["fireRate"] for n in ["domain", "offDomain", "general"])
deadThreshold, liveThreshold = 0.001, 0.01
totalNeurons = domainRate.numel()
results["deadNeurons"] = {
    "definition": "post-GELU activation > 0.05 on fewer than 0.1% of calibration tokens",
    "totalMlpNeurons": totalNeurons,
    "deadOnDomain": int((domainRate < deadThreshold).sum()),
    "deadOnCode": int((codeRate < deadThreshold).sum()),
    "deadOnGeneral": int((generalRate < deadThreshold).sum()),
    "deadOnDomainNotDeadOnCode": int(((domainRate < deadThreshold) & (codeRate >= deadThreshold)).sum()),
    "deadOnDomainLiveOnCodeAbove1pct": int(((domainRate < deadThreshold) & (codeRate >= liveThreshold)).sum()),
    "deadOnCodeNotDeadOnDomain": int(((codeRate < deadThreshold) & (domainRate >= deadThreshold)).sum()),
    "deadOnDomainPerLayer": [int(x) for x in (domainRate < deadThreshold).sum(1)],
    "fireRateCorrelationDomainVsCode": round(torch.corrcoef(torch.stack([domainRate.flatten(), codeRate.flatten()]))[0, 1].item(), 4),
    "fireRateCorrelationDomainVsGeneral": round(torch.corrcoef(torch.stack([domainRate.flatten(), generalRate.flatten()]))[0, 1].item(), 4),
    "medianFireRateDomain": round(domainRate.median().item(), 4),
    "shareFiringUnder1pctDomain": round((domainRate < 0.01).float().mean().item(), 4),
    "shareFiringUnder1pctDomainButOver1pctCode": round(((domainRate < 0.01) & (codeRate >= 0.01)).float().mean().item(), 4),
}
results["deadNeurons"]["shareDeadOnDomain"] = round(results["deadNeurons"]["deadOnDomain"] / totalNeurons, 5)
results["deadNeurons"]["shareDeadOnCode"] = round(results["deadNeurons"]["deadOnCode"] / totalNeurons, 5)
print(results["deadNeurons"], flush=True)
save()

mlpParamsPerNeuron = 768 + 1 + 768
results["paramCount"] = {"total": sum(p.numel() for p in model.parameters()), "mlpParamsPerNeuron": mlpParamsPerNeuron, "structuredRemovedParams": {}}
for fraction in sparsities:
    restore()
    removed, _ = pruneNeurons(fraction, stats["domain"]["meanAbs"])
    results["paramCount"]["structuredRemovedParams"][str(fraction)] = removed * mlpParamsPerNeuron
    evaluate(f"structuredDomain@{fraction}")
    restore()
    pruneNeurons(fraction, None, torch.Generator().manual_seed(0))
    evaluate(f"structuredRandom@{fraction}")
for sparsity in sparsities:
    for label, norms in [("magnitude", None), ("wandaGeneral", stats["general"]["inputNorm"]), ("wandaDomain", stats["domain"]["inputNorm"])]:
        restore()
        pruneUnstructured(sparsity, norms)
        evaluate(f"{label}@{sparsity}")

remaining = budgetSeconds - (time.time() - startTime)
if remaining < 150:
    results["recovery"] = "not run"
else:
    restore()
    _, masks = pruneNeurons(0.5, stats["domain"]["meanAbs"])
    before = {name: perplexity(test) for name, (cal, test) in pools.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    calIds = pools["domain"][0]
    generator = torch.Generator().manual_seed(0)
    model.train()
    stepsDone = 0
    for step in range(150):
        if time.time() - startTime > budgetSeconds - 25:
            break
        ids = calIds[torch.randint(0, len(calIds), (8,), generator=generator)].to(device)
        loss = model(ids, labels=ids).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for block, drop in zip(blocks, masks):
                block.mlp.c_fc.weight[:, drop] = 0
                block.mlp.c_fc.bias[drop] = 0
                block.mlp.c_proj.weight[drop, :] = 0
        stepsDone += 1
    model.eval()
    for block, drop in zip(blocks, masks):
        assert block.mlp.c_proj.weight[drop, :].abs().sum().item() == 0
    results["recovery"] = {"steps": stepsDone, "lr": 1e-4, "batch": 8, "before": before, "after": {name: perplexity(test) for name, (cal, test) in pools.items()}}
print("recovery", results["recovery"], flush=True)
save()
