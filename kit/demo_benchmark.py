"""Time the Mac analyst on demo files and write the answers judges see."""

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

from pinkprocess.kit_data import STYLES, answer_question, parse_table, profile_table

QUESTION = "What's wrong with my system?"


def run(path, question):
    text = Path(path).read_text(encoding="utf-8")
    started = perf_counter()
    table = parse_table(text, Path(path).name)
    parsed = perf_counter()
    profile = profile_table(table, Path(path).name)
    profiled = perf_counter()
    answers = {style: answer_question(table, question, style) for style in STYLES}
    answered = perf_counter()
    return dict(file=Path(path).name, rows=table["rows"], columns=len(table["columns"]), bytes=len(text.encode()),
                question=question, parse_ms=round((parsed - started) * 1000, 1),
                profile_ms=round((profiled - parsed) * 1000, 1), answer_ms=round((answered - profiled) * 1000, 1),
                total_ms=round((answered - started) * 1000, 1), recommendation=profile["recommendation"]["model_id"],
                short=answers["short"]["answer"], summary=answers["summary"]["answer"],
                table=answers["table"]["table"], readings=answers["short"]["readings"])


def markdown(results):
    lines = [f"Machine: {platform.machine()} {platform.mac_ver()[0] or platform.system()}, Python {platform.python_version()}", ""]
    for result in results:
        lines += [f"## {result['file']}", "",
                  f"{result['rows']:,} rows, {result['columns']} columns, {result['bytes'] / 1024:.0f} KB. "
                  f"Parse {result['parse_ms']} ms, profile {result['profile_ms']} ms, all four answers {result['answer_ms']} ms, "
                  f"total {result['total_ms']} ms.", "", f"**Q:** {result['question']}", "", "**Short:** " + result["short"], "",
                  "**Summary:**", "", *[f"> {line}" for line in result["summary"].splitlines()], ""]
        table = result["table"]
        if table:
            lines += ["| " + " | ".join(table["columns"]) + " |", "|" + " --- |" * len(table["columns"])]
            lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table["rows"][:12]]
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = [run(path, args.question) for path in args.files]
    print(json.dumps(results, indent=2) if args.json else markdown(results))


if __name__ == "__main__":
    main()
