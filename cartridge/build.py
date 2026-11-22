"""Train, quantise and verify one cartridge per deck."""

import json
import random
from argparse import ArgumentParser
from math import exp
from pathlib import Path

from .features import FEATURE_COUNT, MAX_ACTIVE, MAX_COUNT, encode
from .resources import DECKS as DECK_DIR, output_path

CLASS_SLOTS = 16
HELD_OUT_PER_CARD = 2
HELD_OUT_OFFTOPIC = 8
REFUSE = "refuse"
EPOCHS = 300
LEARNING_RATE = 0.3
WEIGHT_DECAY = 1e-4
REFUSE_WEIGHT = 0.5
INT8_MAX = 127
EMPTY_SLOT_BIAS = -(2 ** 30)
DECKS = ("health", "money", "wild")


def load_deck(name: str) -> dict:
    return json.loads((DECK_DIR / f"{name}.json").read_text(encoding="utf-8"))


def load_offtopic() -> tuple[list[str], list[str]]:
    questions = json.loads((DECK_DIR / "offtopic.json").read_text(encoding="utf-8"))
    return questions[:-HELD_OUT_OFFTOPIC], questions[-HELD_OUT_OFFTOPIC:]


def split_questions(deck: dict) -> tuple[list, list]:
    train, held_out = [], []
    for card in deck["cards"]:
        questions = card["questions"]
        train += [(q, card["id"]) for q in questions[:-HELD_OUT_PER_CARD]]
        train.append((card["title"] + " " + " ".join(card["steps"]), card["id"]))
        train += [(keyword, card["id"]) for keyword in card["keywords"]]
        held_out += [(q, card["id"]) for q in questions[-HELD_OUT_PER_CARD:]]
    return train, held_out


def fit(examples: list, labels: list[str], seed: int) -> tuple[list, list]:
    """Multinomial logistic regression on sparse integer features, standard library only."""
    order = random.Random(seed)
    width = len(labels)
    weights = [[0.0] * width for _ in range(FEATURE_COUNT)]
    bias = [0.0] * width
    rows = [(encode(text), labels.index(label)) for text, label in examples]
    for _ in range(EPOCHS):
        order.shuffle(rows)
        for features, target in rows:
            scores = list(bias)
            for index, count in features:
                row = weights[index]
                for k in range(width):
                    scores[k] += count * row[k]
            peak = max(scores)
            exps = [exp(score - peak) for score in scores]
            total = sum(exps)
            emphasis = REFUSE_WEIGHT if labels[target] == REFUSE else 1.0
            for k in range(width):
                gradient = emphasis * (exps[k] / total - (k == target))
                bias[k] -= LEARNING_RATE * gradient
                for index, count in features:
                    row = weights[index]
                    row[k] -= LEARNING_RATE * (gradient * count + WEIGHT_DECAY * row[k])
    return weights, bias


def quantise(weights: list, bias: list, scale: float) -> tuple[list, list]:
    padding = CLASS_SLOTS - len(bias)
    int_weights = [[round(value * scale) for value in row] + [0] * padding for row in weights]
    return int_weights, [round(value * scale) for value in bias] + [EMPTY_SLOT_BIAS] * padding


def build_all(output: Path) -> list[dict]:
    decks = {name: load_deck(name) for name in DECKS}
    off_train, _ = load_offtopic()
    splits = {name: split_questions(deck) for name, deck in decks.items()}
    fitted = {}
    for name, deck in decks.items():
        foreign = [(q, REFUSE) for other in DECKS if other != name for q, _ in splits[other][0]]
        labels = [card["id"] for card in deck["cards"]] + [REFUSE]
        fitted[name] = (labels, *fit(splits[name][0] + foreign + [(q, REFUSE) for q in off_train],
                                     labels, seed=sum(map(ord, name))))
    largest = max(abs(value) for _, weights, _ in fitted.values() for row in weights for value in row)
    scale = INT8_MAX / largest
    built = []
    for name, (labels, weights, bias) in fitted.items():
        deck = decks[name]
        int_weights, int_bias = quantise(weights, bias, scale)
        worst = MAX_ACTIVE * MAX_COUNT * INT8_MAX + max(abs(value) for value in int_bias[: len(labels)])
        built.append({
            "schema_version": 1,
            "name": name,
            "title": deck["title"],
            "tagline": deck["tagline"],
            "notice": deck["notice"],
            "labels": labels,
            "feature_count": FEATURE_COUNT,
            "class_slots": CLASS_SLOTS,
            "scale": scale,
            "weight_bytes": FEATURE_COUNT * CLASS_SLOTS,
            "accumulator_signed_bits": 1 + worst.bit_length(),
            "cards": {card["id"]: {k: card[k] for k in ("title", "steps", "urgent")} for card in deck["cards"]},
            "examples": [card["questions"][0] for card in deck["cards"][:4]],
            "bias": int_bias,
            "weights": int_weights,
        })
    output.mkdir(parents=True, exist_ok=True)
    for cartridge in built:
        text = json.dumps(cartridge, separators=(",", ":")) + "\n"
        (output / f"{cartridge['name']}.json").write_text(text, encoding="utf-8")
    return built


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=output_path("cartridges", "cartridges"))
    args = parser.parse_args()
    for cartridge in build_all(args.output):
        print(f"{cartridge['name']:<7} {len(cartridge['cards'])} cards  "
              f"{cartridge['weight_bytes']} weight bytes  "
              f"accumulator {cartridge['accumulator_signed_bits']} bits")


if __name__ == "__main__":
    main()
