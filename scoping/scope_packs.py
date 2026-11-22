import glob, hashlib, io, json, math, os, random, re, sysconfig, time, urllib.request
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_num_threads(2)
here = os.path.dirname(os.path.abspath(__file__))
resultsPath = os.path.join(here, "scope_packs.json")
seqLen, calTokens, testTokens, steps, rank, removedFraction = 256, 60000, 30000, 120, 8, 0.5
random.seed(0)
torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True).to(device).eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
blocks = model.transformer.h
hidden, inner = 768, 3072
results = {"model": "gpt2", "steps": steps, "rank": rank, "removedFraction": removedFraction, "packs": {}}


def toBlocks(texts, tokenBudget):
    ids = []
    for text in texts:
        ids += tokenizer(text)["input_ids"] + [tokenizer.eos_token_id]
        if len(ids) >= tokenBudget:
            break
    count = min(len(ids), tokenBudget) // seqLen
    return torch.tensor(ids[: count * seqLen]).view(count, seqLen)


def financeTexts():
    url = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/sent_train.csv"
    frame = pd.read_csv(io.BytesIO(urllib.request.urlopen(url, timeout=30).read()))
    texts = [re.sub(r"https?://\S+", "", t).strip() for t in frame["text"]]
    texts = [t for t in texts if len(t) > 30]
    random.Random(0).shuffle(texts)
    return texts


def codeTexts():
    paths = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    random.Random(0).shuffle(paths)
    return [open(p, encoding="utf-8", errors="ignore").read()[:20000] for p in paths[:120]]


finance, code = financeTexts(), codeTexts()
pools = {
    "finance": (toBlocks(finance[: len(finance) // 2], calTokens), toBlocks(finance[len(finance) // 2 :], testTokens)),
    "code": (toBlocks(code[:60], calTokens), toBlocks(code[60:], testTokens)),
}
activePack = {"pack": None}


def install():
    for layer, block in enumerate(blocks):
        def expandHook(module, inputs, output, layer=layer):
            pack = activePack["pack"]
            return output if pack is None else output + inputs[0] @ pack["expandA"][layer] @ pack["expandB"][layer]

        def maskHook(module, inputs, output, layer=layer):
            pack = activePack["pack"]
            return output if pack is None else output * pack["keep"][layer]

        def projectHook(module, inputs, output, layer=layer):
            pack = activePack["pack"]
            return output if pack is None else output + inputs[0] @ pack["projectA"][layer] @ pack["projectB"][layer]

        block.mlp.c_fc.register_forward_hook(expandHook)
        block.mlp.act.register_forward_hook(maskHook)
        block.mlp.c_proj.register_forward_hook(projectHook)


@torch.no_grad()
def perplexity(batchIds):
    lossSum, count = 0.0, 0
    for i in range(0, len(batchIds), 8):
        ids = batchIds[i : i + 8].to(device)
        lossSum += model(ids, labels=ids).loss.item() * ids.shape[0]
        count += ids.shape[0]
    return round(math.exp(lossSum / count), 2)


@torch.no_grad()
def logitsDigest():
    ids = pools["finance"][1][:2].to(device)
    return hashlib.sha256(model(ids).logits.float().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def neuronScores(batchIds):
    sums, hooks = {}, []
    for layer, block in enumerate(blocks):
        hooks.append(block.mlp.act.register_forward_hook(
            lambda m, i, out, layer=layer: sums.__setitem__(layer, sums.get(layer, 0) + out.abs().sum((0, 1)))))
    for i in range(0, len(batchIds), 8):
        model(batchIds[i : i + 8].to(device))
    for hook in hooks:
        hook.remove()
    return [sums[layer] * blocks[layer].mlp.c_proj.weight.norm(dim=1) for layer in range(len(blocks))]


def newPack(scores):
    keep = []
    for score in scores:
        mask = torch.ones(inner, device=device)
        mask[score.argsort()[: int(inner * removedFraction)]] = 0
        keep.append(mask)
    scaled = lambda rows, cols: (torch.randn(rows, cols, device=device) * 0.01).requires_grad_(True)
    zeros = lambda rows, cols: torch.zeros(rows, cols, device=device, requires_grad=True)
    return {
        "keep": keep,
        "expandA": [scaled(hidden, rank) for _ in blocks], "expandB": [zeros(rank, inner) for _ in blocks],
        "projectA": [scaled(inner, rank) for _ in blocks], "projectB": [zeros(rank, hidden) for _ in blocks],
    }


def train(pack, calIds):
    trainable = [t for key in ("expandA", "expandB", "projectA", "projectB") for t in pack[key]]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(0)
    activePack["pack"] = pack
    for step in range(steps):
        ids = calIds[torch.randint(0, len(calIds), (8,), generator=generator)].to(device)
        loss = model(ids, labels=ids).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for tensor in trainable:
        tensor.requires_grad_(False)


def packBytes(pack):
    maskBytes = len(blocks) * inner // 8
    deltaValues = sum(t.numel() for key in ("expandA", "expandB", "projectA", "projectB") for t in pack[key])
    return {"maskBytes": maskBytes, "deltaValues": deltaValues, "float32Bytes": maskBytes + 4 * deltaValues,
            "int8Bytes": maskBytes + deltaValues}


def save():
    json.dump(results, open(resultsPath, "w"), indent=1)


baseDigestBeforeHooks = logitsDigest()
install()
results["base"] = {name: perplexity(test) for name, (cal, test) in pools.items()}
results["baseDigest"] = baseDigestBeforeHooks
print("base", results["base"], flush=True)
save()
scoresByDomain = {name: neuronScores(cal) for name, (cal, test) in pools.items()}
packs = {}
for name, (calIds, testIds) in pools.items():
    pack = newPack(scoresByDomain[name])
    activePack["pack"] = pack
    maskOnly = {other: perplexity(test) for other, (cal, test) in pools.items()}
    train(pack, calIds)
    packs[name] = pack
    results["packs"][name] = {"maskOnly": maskOnly, "withPack": {other: perplexity(test) for other, (cal, test) in pools.items()},
                              "size": packBytes(pack)}
    print(name, results["packs"][name], flush=True)
    save()

switchTimes = []
for repeat in range(200):
    started = time.perf_counter()
    activePack["pack"] = packs["finance" if repeat % 2 else "code"]
    switchTimes.append((time.perf_counter() - started) * 1e6)
activePack["pack"] = None
results["switch"] = {
    "medianMicroseconds": round(sorted(switchTimes)[len(switchTimes) // 2], 2),
    "note": "pointer swap between packs already in memory; the base weights are never written",
    "digestAfterUnload": logitsDigest(),
    "baseRestoredBitExact": logitsDigest() == baseDigestBeforeHooks,
    "sharedNeuronsKeptByBoth": round(sum(float((packs["finance"]["keep"][l] * packs["code"]["keep"][l]).sum()) for l in range(len(blocks))) / (len(blocks) * inner), 4),
}
results["baseParams"] = sum(p.numel() for p in model.parameters())
print("switch", results["switch"], flush=True)
save()
