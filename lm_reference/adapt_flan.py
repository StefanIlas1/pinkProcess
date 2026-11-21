"""Train the telemetry FLAN candidate on synthetic, numerically grounded QA."""
import argparse
from pathlib import Path
import random

from lm_reference.flan_prompt import telemetry_prompt


def example(rng, task, *, detailed=False):
    capacity = rng.choice([4, 6, 8, 12, 16, 24, 32, 64])
    waiting = rng.choice([0, 0, rng.randint(1, 90)])
    busy = capacity if waiting else rng.randint(0, capacity)
    peak = max(waiting, rng.randint(0, 190))
    mean = round(rng.uniform(min(waiting, peak), peak), 1)
    age = rng.randint(1, 59)
    obs = {"peak_waiting": {"waiting": peak, "time_ms": (60 - age) * 1000},
           "first": {"waiting": rng.randint(0, peak)}, "covered_ms": 60000,
           "waiting": {"mean": mean}, "complete": True}
    forecast = rng.choice(["The FPGA warns that requests may start waiting within the next ten seconds.",
                           "The FPGA expects the queue to persist over the next ten seconds.", None])
    e = {"origin": rng.choice(["demo", "live"]), "state": "ready", "capacity": capacity,
         "latest": {"time_ms": 60000, "busy": busy, "waiting": waiting},
         "observations": obs, "forecast_interpretation": forecast}
    questions = [
        ["What seems wrong and what should I investigate next?", "Diagnose the situation.",
         "How is the system doing?", "Explain what happened with the queue.",
         "Any signs of pool pressure? What should we check?", "Give me a summary and next steps."],
        ["How long is the queue right now?", "How many requests are still waiting?", "What is the latest queue depth?"],
        ["When was the queue largest and how large was it?", "What was the maximum backlog?", "Describe peak pressure."],
        ["What is the average waiting queue?", "What was mean queue depth over the recorded window?"],
        ["How many connections are busy?", "What is the current occupancy?", "How much of the pool is in use?"],
        ["Which endpoint was slow?", "Did a slow database cause this?", "What was the P95 latency?",
         "Was there a connection leak?", "Is a particular query responsible?", "What caused this?"],
        ["What does the FPGA predict?", "Is a forecast available?", "What does the model expect next?"]]
    summary = (f"{waiting} requests are waiting now and {busy} of {capacity} connections are busy. "
               f"The queue peaked at {peak}. Check request arrival rates and connection hold times.")
    answers = [summary, f"{waiting} requests are waiting.", f"The queue peaked at {peak} requests {age} seconds ago.",
               f"The mean queue was {mean:g} requests.", f"{busy} of {capacity} connections are busy.",
               "The measurements do not identify endpoints, latency or root cause.",
               forecast or "The forecast is unavailable."]
    if detailed:
        first = rng.randint(0, peak)
        obs["first"]["waiting"] = first
        obs["covered_ms"] = rng.choice([5000, 10000, 30000, 60000])
        obs["complete"] = rng.choice([True, True, False])
        age = 0 if peak == waiting else round(rng.uniform(0.1, obs["covered_ms"] / 1000 - 0.1), 1)
        obs["peak_waiting"]["time_ms"] = int((60 - age) * 1000)
        answers[2] = f"The queue peaked at {peak} requests {age:g} seconds ago."
        busy = rng.randint(0, capacity) if rng.random() < 0.4 else busy
        e["latest"]["busy"] = busy
        answers[4] = f"{busy} of {capacity} connections are busy."
        forecast = rng.choice([
            "The FPGA predicts the queue will remain nonempty over the next ten seconds.",
            "The FPGA predicts the queue will drain at least once over the next ten seconds.",
            "The FPGA warns that requests may start waiting within the next ten seconds.",
            "The FPGA issued no queue-onset warning for the next ten seconds.",
            "Forecast unavailable."])
        e["forecast_interpretation"] = forecast
        if forecast.startswith("The FPGA predicts"):
            e["forecast_interpretation"] += (f" It expects {rng.choice(['at most', 'more than'])}"
                                             f" the current {waiting} waiting at the end of that period.")
        opening = "In this demo, " if e["origin"] == "demo" else ""
        if waiting == 0 and peak:
            status = f"the queue cleared after peaking at {peak}."
        elif waiting == 0:
            status = "no backlog was observed."
        else:
            status = f"{waiting} requests are waiting, with a peak of {peak}."
        status = opening + status if opening else status.capitalize()
        answers[0] = (f"{status} {busy} of {capacity} connections are busy. "
                      f"{forecast} Compare arrival rates with connection hold times.")
        answers[6] = e["forecast_interpretation"]
        questions.extend([
            ["How large is the connection pool?", "What is the capacity?", "How many connections can the pool serve?"],
            ["How has the number waiting changed?", "Compare the queue at the beginning and end.", "Where did the backlog start and end?"],
            ["How much history is recorded?", "Do we have the full observation window?", "What time span is covered?"],
            ["Which service is broken?", "Is CPU utilization high?", "What was throughput?", "What was the memory footprint?", "How large is the database cache?"]])
        questions[0].extend(["What happened?", "What's the state of things?", "What trouble do you see, and why?",
                             "Is this healthy? Summarize the evidence.", "What should the on-call engineer investigate?"])
        questions[3].extend(["What was the window average?", "How many requests typically waited?", "Average backlog?"])
        questions[2].extend(["How far back was the high-water mark?", "What was the worst queue measurement?"])
        complete = "complete" if obs["complete"] else "partial"
        answers.extend([f"The pool capacity is {capacity} connections.",
                        f"The queue started at {first} and ended at {waiting} waiting requests.",
                        f"The recorded window is {obs['covered_ms'] / 1000:g} seconds and is {complete}.",
                        "The measurements only describe connection occupancy and queue depth."])
    return telemetry_prompt(rng.choice(questions[task]), e), answers[task]


def train(source, tokenizer, out, steps, seed, detailed):
    import sentencepiece as sp
    import torch
    from transformers import T5ForConditionalGeneration
    from safetensors.torch import save_file

    rng = random.Random(seed)
    torch.manual_seed(seed)
    proc = sp.SentencePieceProcessor(model_file=str(tokenizer))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = T5ForConditionalGeneration.from_pretrained(str(source)).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    class Adapter(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.base = layer
            self.a = torch.nn.Linear(layer.in_features, 16, bias=False, device=dev)
            self.b = torch.nn.Linear(16, layer.out_features, bias=False, device=dev)
            torch.nn.init.zeros_(self.b.weight)

        @property
        def weight(self):
            return self.base.weight

        def forward(self, x):
            return self.base(x) + self.b(self.a(x))

    adapters = []
    for module in list(model.modules()):
        for name in ("q", "v", "wo"):
            layer = getattr(module, name, None)
            if isinstance(layer, torch.nn.Linear):
                adapter = Adapter(layer)
                setattr(module, name, adapter)
                adapters.append((module, name, adapter))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.002)
    def batch(texts, pad):
        encoded = [proc.encode(t) + [1] for t in texts]
        size = max(map(len, encoded))
        return torch.tensor([x + [pad] * (size - len(x)) for x in encoded], device=dev)

    model.train()
    for i in range(steps):
        tasks = [0, 0, 0] + list(range(1, 11 if detailed else 7))
        qa = [example(rng, rng.choice(tasks), detailed=detailed) for _ in range(8)]
        inp, labels = zip(*qa)
        x = batch(inp, 0)
        loss = model(input_ids=x, attention_mask=x.ne(0), labels=batch(labels, -100), use_cache=False).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        opt.step()
        if i % 10 == 0:
            print(i, loss.item(), flush=True)
    with torch.no_grad():
        for module, name, adapter in adapters:
            adapter.base.weight.add_(adapter.b.weight @ adapter.a.weight)
            setattr(module, name, adapter.base)
    state = {k: v.cpu().contiguous() for k, v in model.state_dict().items()
             if k not in ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight")}
    save_file(state, str(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "tokenizer", "out"):
        p.add_argument(k, type=Path)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=3717)
    p.add_argument("--detailed", action="store_true", help="Train with forecast, provenance, trend and coverage examples.")
    args = p.parse_args()
    train(args.source, args.tokenizer, args.out, args.steps, args.seed, args.detailed)
