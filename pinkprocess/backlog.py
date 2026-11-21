"""Integer inputs for forecasts about an already-occupied waiting queue."""

from .fixed_point import encode_ratio

FIT_VERSION = "queue-outlook-v1"
VERSION = "queue-outlook-v2"
TARGETS = ("persists", "grows")
FIT_INPUTS = ("depth", "change_1s", "change_4s", "occupied_fraction", "full_fraction", "bias")


def scaled(numerator, denominator):
    # encode_ratio rounds ties to even at scale 256. Scale to 127 instead.
    return max(-127, min(127, encode_ratio(numerator * 127, denominator * 256)))


def fitted_features(window, capacity):
    current = window[-1].waiting
    return [scaled(current, 16 * capacity),
            scaled(current - window[-11].waiting, 4 * capacity),
            scaled(current - window[-41].waiting, 12 * capacity),
            scaled(sum(r.waiting > 0 for r in window), len(window)),
            scaled(sum(r.busy == capacity for r in window), len(window)), 127]

INPUTS = ("depth_scaled", "change_4s_scaled", "change_4s_clipped", "bias")


def features(window, capacity):
    """Give depth and its change one common scale for the persistence projection."""
    depth = window[-1].waiting
    change = depth - window[-41].waiting
    maximum = max(127, depth, abs(change))
    return [scaled(depth, maximum), scaled(change, maximum), max(-127,min(127,change)), 1]
