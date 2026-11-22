"""Compact one telemetry snapshot into input for the resident encoder."""


def measured_history(evidence):
    """Describe the observed endpoints and peak for the history-trained checkpoint."""
    observations = evidence.get("observations") or {}
    first = (observations.get("first") or {}).get("waiting")
    peak = (observations.get("peak_waiting") or {}).get("waiting")
    last = (evidence.get("latest") or {}).get("waiting")
    if any(type(n) is not int or n < 0 for n in (first, peak, last)) or peak < max(first, last):
        return ""
    if peak == 0:
        return "No backlog was observed in this window."
    if last == 0:
        return "The queue cleared by the final sample."
    if first == last:
        return f"Active queue. First and final waiting counts were both {last}."
    direction = "rose" if last > first else "fell"
    return f"Active queue. Waiting {direction} from {first} to {last}."


def telemetry_prompt(question, evidence, *, history=False):
    if not isinstance(question, str) or not 0 < len(question.strip()) <= 500:
        raise ValueError("Enter a question of up to 500 characters.")
    latest = evidence["latest"]
    obs = evidence.get("observations") or {}
    origin = {"demo": "demo", "dataset": "CSV"}.get(evidence.get("origin"), "live")
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
    if history:
        measured = measured_history(evidence)
        if measured:
            sentences.append("Measured queue history: " + measured)
    return "Answer the question using the measurements. " + " ".join(sentences) + "\nQuestion: " + question.strip() + "\nAnswer:"
