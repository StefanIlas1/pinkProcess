"""Score the shipped cartridges on held-out questions and export kernel test vectors."""

import json
from argparse import ArgumentParser
from pathlib import Path

from .build import DECKS, load_deck, load_offtopic, split_questions
from .features import encode
from .runtime import ROOT, Device


def held_out_report(device: Device) -> dict:
    right_card = right_domain = total = 0
    for name in DECKS:
        for question, card in split_questions(load_deck(name))[1]:
            reply = device.ask(question)
            total += 1
            right_domain += reply["answered"] and reply["cartridge"] == name
            right_card += reply["answered"] and reply["cartridge"] == name and reply["card"]["id"] == card
    offtopic = load_offtopic()[1]
    refused = sum(not device.ask(question)["answered"] for question in offtopic)
    return {"in_domain_questions": total, "right_domain": right_domain, "right_card": right_card,
            "offtopic_questions": len(offtopic), "offtopic_refused": refused}


def export_vectors(device: Device, path: Path) -> int:
    """Write weights and expected scores in the plain text format hw/testbench.cpp reads."""
    lines = []
    questions = [q for name in DECKS for q, _ in split_questions(load_deck(name))[1]] + load_offtopic()[1]
    for cartridge in device.library.values():
        lines.append(f"cartridge {cartridge.name} {len(cartridge.weights)} {len(cartridge.bias)}")
        lines.append(" ".join(map(str, cartridge.bias)))
        lines += [" ".join(map(str, row)) for row in cartridge.weights]
        lines.append(f"vectors {len(questions)}")
        for question in questions:
            features = encode(question)
            flat = " ".join(f"{index} {count}" for index, count in features)
            lines.append(f"{len(features)} {flat}".strip())
            lines.append(" ".join(map(str, cartridge.scores(features))))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(questions) * len(device.library)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=ROOT / "hw" / "vectors.txt")
    args = parser.parse_args()
    device = Device()
    report = held_out_report(device)
    print(json.dumps(report, indent=2))
    print(f"wrote {export_vectors(device, args.vectors)} reference vectors to {args.vectors}")


if __name__ == "__main__":
    main()
