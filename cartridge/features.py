"""Turn a question into the sparse integer feature list the kernel consumes."""

import re

FEATURE_COUNT = 4096
MAX_ACTIVE = 64
MAX_COUNT = 15
STOPWORDS = frozenset(
    "a an the is are was were be been am i me my we our you your he she it they them his her its "
    "this that these those of to in on at for with and or but if so do does did has have had "
    "what how when where which who why can could should would will there here from as by not no".split()
)
SUFFIXES = ("ing", "ed", "es", "s")


def stem(word: str) -> str:
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(word) for word in words if word not in STOPWORDS]


def fnv1a(text: str) -> int:
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def encode(text: str) -> list[tuple[int, int]]:
    """Return up to MAX_ACTIVE (feature index, count) pairs, sorted by index."""
    counts: dict[int, int] = {}
    for word in tokens(text):
        index = fnv1a(word) % FEATURE_COUNT
        counts[index] = min(MAX_COUNT, counts.get(index, 0) + 1)
    return sorted(counts.items())[:MAX_ACTIVE]
