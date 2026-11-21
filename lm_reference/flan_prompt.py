"""Compact one telemetry snapshot into input for the resident encoder."""


def telemetry_prompt(question, evidence):
    if not isinstance(question, str) or not 0 < len(question.strip()) <= 500:
        raise ValueError("Enter a question of up to 500 characters.")
    latest = evidence["latest"]
    obs = evidence.get("observations") or {}
    origin = "demo" if evidence.get("origin") == "demo" else "live"
    sentences = [f"Connection pool {origin} measurements ({evidence.get('state', 'unknown')}).",
                 f"Current queue: {latest['waiting']} waiting requests.",
                 f"Current occupancy: {latest['busy']} of {evidence['capacity']} connections busy."]
    peak = obs.get("peak_waiting")
    if peak:
        age = (latest["time_ms"] - peak["time_ms"]) / 1000
        sentences.append(f"Peak queue: {peak['waiting']} requests, {age:g} seconds ago.")
    if obs.get("first"):
        sentences.append(f"Queue at start: {obs['first']['waiting']} requests.")
    if "covered_ms" in obs:
        sentences.append(f"Recorded duration: {obs['covered_ms'] / 1000:g} seconds.")
    if obs.get("complete") is False:
        sentences.append("Only part of the requested window is available.")
    waiting = obs.get("waiting") or {}
    if "mean" in waiting:
        sentences.append(f"Mean queue: {waiting['mean']:.4g} requests.")
    busy = obs.get("busy") or {}
    if "mean" in busy:
        sentences.append(f"Mean occupancy: {busy['mean']:.4g} connections.")
    sentences.extend([evidence.get("forecast_interpretation") or "Forecast unavailable.",
                      "Endpoint, root cause and request latency: unknown."])
    return "Answer the question using the measurements. " + " ".join(sentences) + "\nQuestion: " + question.strip() + "\nAnswer:"
