"""A small development-trained token proposal table for verified FLAN decoding."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from random import Random


class TokenDraft:
    def __init__(self, transitions):
        self.transitions = transitions

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = Path(__file__).with_name("telemetry-token-draft.tsv.gz")
        transitions = {}
        with gzip.open(path, "rt", encoding="ascii") as source:
            for line in source:
                *prefix, token = map(int, line.split())
                transitions[tuple(prefix)] = token
        return cls(transitions)

    @classmethod
    def load_history(cls, path):
        """Load the independently distributed proposal table matching this recipe."""
        compressed = Path(path).read_bytes()
        digest = sha256(compressed).hexdigest()
        if digest != "e84e9244b04623ea7924bf1cc027738f5a6b48d5a305a1a4e8f0e49a3da948d2":
            raise ValueError("Queue history proposal table SHA-256 mismatch.")
        return cls.load(BytesIO(compressed))

    def propose(self, tokens, limit):
        # The initial decoder symbol is zero. Only five preceding IDs matter.
        history = (tokens[-5:] if len(tokens) >= 5 else [0, *tokens])
        guesses = []
        for _ in range(limit):
            for size in range(min(5, len(history)), 0, -1):
                token = self.transitions.get(tuple(history[-size:]))
                if token is not None:
                    break
            else:
                break
            if token == 1:
                break
            guesses.append(token)
            history.append(token)
        return guesses


def main():
    """Reproduce the shipped draft from the existing training generator."""
    from .adapt_flan import example
    from .flan import UnigramTokenizer

    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("tokenizer", type=Path)
    args.add_argument("output", type=Path)
    args.add_argument("--queue-history", action="store_true",
                      help="Train proposals on development examples from the queue-history recipe")
    options = args.parse_args()
    tokenizer = UnigramTokenizer(options.tokenizer)
    counts = defaultdict(Counter)
    random = Random(147)
    if options.queue_history:
        from functools import partial
        from kit.reference_forecast import make_forecaster
        from .train_queue_history import example as history_example

        history = partial(history_example, forecast=make_forecaster(Path(__file__).resolve().parents[1] / "profiles"),
                          detailed=True, diagnostic=True, balanced_history=True)
    for index in range(800 if options.queue_history else 500):
        if options.queue_history:
            _, answer = history(random, random.choice([0] * 12 + list(range(1, 11))))
        else:
            _, answer = example(random, index % 7, detailed=True, diagnostic=True)
        sequence = [0, *tokenizer.encode(answer)]
        for pos in range(1, len(sequence)):
            for size in range(1, min(5, pos) + 1):
                counts[tuple(sequence[pos-size:pos])][sequence[pos]] += 1
    with options.output.open("wb") as dest:
        with gzip.GzipFile(fileobj=dest, filename="", mode="wb", mtime=0) as table:
            for prefix, votes in sorted(counts.items()):
                row = (*prefix, votes.most_common(1)[0][0])
                table.write((" ".join(map(str, row)) + "\n").encode("ascii"))


if __name__ == "__main__":
    main()
