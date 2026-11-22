"""Score development telemetry windows with the two fixed-point kit profiles."""

import json

from pinkprocess.backlog import VERSION as OUTLOOK_VERSION, features as outlook_features
from pinkprocess.dataset import Example, FORECAST_MS
from pinkprocess.fixed_point import FixedProfile, encode_features
from pinkprocess.profile import Profile
from pinkprocess.telemetry_device import BASE as ONSET_VERSION


def make_forecaster(profiles):
    onset = FixedProfile.quantize(Profile.load(profiles / f"{ONSET_VERSION}.json"))
    outlook = json.loads((profiles / f"{OUTLOOK_VERSION}.json").read_bytes())["weights"]

    def forecast(window, capacity):
        common = {"time_ms": window[-1].time_ms, "horizon_ms": FORECAST_MS}
        if window[-1].waiting:
            features = outlook_features(window, capacity)
            scores = [sum(weight * value for weight, value in zip(row, features)) for row in outlook]
            return {"model": "queue_outlook", "version": OUTLOOK_VERSION, "scores": scores, **common}
        score = onset.score(encode_features(Example(window[-1].time_ms, window, None), capacity))
        return {"model": "pool_wait_logistic", "version": ONSET_VERSION, "alert": score >= onset.threshold,
                "score": score, "threshold": onset.threshold, **common}

    return forecast
