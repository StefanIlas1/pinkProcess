"""Timestamped measurements and forecast interpretations shared by both runtimes."""

def live_answer(snapshot, intent, metrics):
    state = snapshot["state"]
    latest = snapshot.get("latest")
    if latest is None:
        return "The board has no telemetry readings for this stream."
    if not snapshot["fresh"]:
        return "The board's telemetry is stale. It last saw {} busy and {} waiting.".format(
            latest["busy"], latest["waiting"])
    if intent == "stream_status":
        if metrics == ["busy"]:
            return f'{latest["busy"]} of {snapshot["capacity"]} connections are busy.'
        if metrics == ["waiting"]:
            return f'Waiting requests: {latest["waiting"]}.'
        return f'{latest["busy"]} of {snapshot["capacity"]} connections busy, with {latest["waiting"]} requests waiting.'
    if intent in ("peak", "trend"):
        observations = snapshot["observations"]
        if intent == "trend" and observations["sample_count"] < 2:
            return "The board needs at least two readings to report a trend."
        labels = {"busy": "Busy connections", "waiting": "Waiting requests"}
        statements = []
        for metric in metrics:
            if intent == "peak":
                peak = observations[f"peak_{metric}"]
                statements.append(f'{labels[metric]} peaked at {peak[metric]} at {peak["time_ms"] / 1000:g} s.')
            else:
                first, last = observations["first"][metric], observations["latest"][metric]
                if first == last:
                    statements.append(f"{labels[metric]} started and ended at {first}.")
                else:
                    direction = "rose" if last > first else "fell"
                    statements.append(f"{labels[metric]} {direction} from {first} to {last}.")
        span = (observations["end_ms"] - observations["start_ms"]) / 1000
        statements.append(f'Based on {observations["sample_count"]} readings spanning {span:g} seconds.')
        return " ".join(statements)
    forecast = snapshot.get("forecast")
    if not forecast:
        if state == "warming":
            return "The board is collecting five seconds of continuous readings for a forecast."
        return "The FPGA forecast is unavailable for this stream."
    if forecast["model"] == "queue_outlook":
        persists, grows = [score >= 0 for score in forecast["scores"]]
        n = latest["waiting"]
        persistence = "remain nonempty" if persists else "drain at least once"
        growth = "more than" if grows else "at most"
        return (f"The FPGA predicts the queue will {persistence} over the next ten seconds. "
                f"It expects {growth} the current {n} waiting at the end of that period.")
    if forecast["model"] == "pool_wait_logistic":
        if forecast["alert"]:
            return "The FPGA warns that requests may start waiting within the next ten seconds."
        return "The FPGA issued no queue-onset warning for the next ten seconds."
    return "The board returned an unsupported forecast model."


def analyst_evidence(snapshot):
    """Give the analyst a readable view of the timestamped data used for the answer."""
    fields = ("origin", "source", "capacity", "state", "latest", "observations", "limitations",
              "forecast_reading")
    evidence = {key: snapshot[key] for key in fields if key in snapshot}
    latest = snapshot["latest"]
    capacity = snapshot.get("capacity")
    if capacity:
        evidence["observed_now"] = (
            f'{latest["busy"]} of {capacity} connections occupied '
            f'({100 * latest["busy"] / capacity:.1f}%); {latest["waiting"]} requests waiting.'
        )
    observations = snapshot.get("observations") or {}
    peak = observations.get("peak_waiting")
    if peak:
        evidence["observed_peak"] = (
            f'{peak["waiting"]} requests were waiting at the window maximum, '
            f'{(latest["time_ms"] - peak["time_ms"]) / 1000:g} seconds before the latest reading. '
            f'{peak["busy"]} connections were occupied at that instant.'
        )
    forecast = snapshot.get("forecast")
    if forecast:
        scored = {**snapshot, "latest": snapshot.get("forecast_reading") or latest}
        evidence["forecast"] = {key: forecast[key] for key in ("version", "time_ms", "horizon_ms")
                                if key in forecast}
        evidence["forecast_interpretation"] = live_answer(scored, "stream_forecast", [])
    return evidence
