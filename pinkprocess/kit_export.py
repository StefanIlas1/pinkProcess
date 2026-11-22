"""Turn a saved PYNQ Process session into a Markdown report, a CSV file or JSON."""

import csv
from datetime import datetime
import io
import json
import re

from .kit_accounts import clean_chat

FORMATS = {"md": "text/markdown", "csv": "text/csv", "json": "application/json"}
HOT_PATH = (("data_ms", "Read the data"), ("profile_ms", "Profile the columns"), ("model_ms", "Pick the model"),
            ("fabric_ms", "FPGA forecast"), ("answer_ms", "Write the answer"))
CSV_COLUMNS = ("number", "question", "style", "answer", "readings", "dataset", "model", "asked")


def safe_filename(text, extension):
    stem = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()[:60] or "session"
    return f"{stem}.{extension}"


def answer_style(message):
    if message.get("style"):
        return message["style"]
    if message.get("table"):
        return "table"
    return "transcript" if message.get("mono") else "text"


def exchanges(session):
    rows, question = [], None
    for message in session["messages"]:
        if message["role"] == "user":
            question = message
            continue
        rows.append(dict(number=len(rows) + 1, question=question["text"] if question else "", answer=message["text"],
                         style=answer_style(message), readings=message.get("readings", ""), table=message.get("table"),
                         mono=bool(message.get("mono")), asked_ms=(question or message)["time_ms"]))
        question = None
    return rows


def stamp(milliseconds):
    if not milliseconds:
        return ""
    try:
        return datetime.fromtimestamp(milliseconds / 1000).strftime("%Y-%m-%d %H:%M")
    except (ValueError, OverflowError, OSError):
        return ""


def cell_text(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table_lines(table):
    columns, rows = table.get("columns"), table.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list) or not columns:
        return []
    lines = ["| " + " | ".join(cell_text(column) for column in columns) + " |", "|" + " --- |" * len(columns)]
    for row in rows[:200]:
        if isinstance(row, list):
            cells = [cell_text(row[index]) if index < len(row) else "" for index in range(len(columns))]
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def to_markdown(session):
    lines = [f"# {session['title'] or session['dataset'] or 'PYNQ Process session'}", ""]
    for label, value in (("Dataset", session["dataset"]), ("Model", session["model"]), ("Updated", stamp(session["updated_ms"]))):
        if value:
            lines.append(f"- {label}: {value}")
    timings = session["timings"]
    shown = [(label, timings[key]) for key, label in HOT_PATH if key in timings]
    if shown:
        lines += ["", "## Hot path", "", "| Step | Time |", "| --- | --- |"]
        lines += [f"| {label} | {value:g} ms |" for label, value in shown]
    items = exchanges(session)
    lines += ["", "## Questions and answers"]
    if not items:
        lines += ["", "No questions were asked in this session."]
    for item in items:
        lines += ["", f"### {item['number']}. {item['question'] or 'Answer'}", "", f"Answer style: {item['style']}", ""]
        if item["mono"]:
            lines += ["```", item["answer"], "```"]
        else:
            lines.append(item["answer"])
        if item["table"]:
            lines += [""] + table_lines(item["table"])
        if item["readings"]:
            lines += ["", f"Readings behind this answer: {item['readings']}"]
    lines += ["", "## Notes", "", session["notes"] or "No notes."]
    return "\n".join(lines) + "\n"


def spreadsheet_safe(value):
    value = str(value)
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def to_csv(session):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for item in exchanges(session):
        writer.writerow([item["number"], *(spreadsheet_safe(value) for value in (
            item["question"], item["style"], item["answer"], item["readings"], session["dataset"], session["model"],
            stamp(item["asked_ms"])))])
    return output.getvalue()


def export_session(session, fmt):
    if fmt not in FORMATS:
        raise ValueError("Choose Markdown (md), CSV (csv) or JSON (json) for the export.")
    cleaned = clean_chat(session) if isinstance(session, dict) else None
    if cleaned is None:
        cleaned = clean_chat({**session, "id": "session"}) if isinstance(session, dict) else None
    if cleaned is None:
        raise ValueError("There is no session to export.")
    if fmt == "json":
        content = json.dumps(session, indent=2, ensure_ascii=False, default=str)
    elif fmt == "csv":
        content = to_csv(cleaned)
    else:
        content = to_markdown(cleaned)
    return dict(filename=safe_filename(cleaned["title"] or cleaned["dataset"], fmt), mime=FORMATS[fmt], content=content)


def export_action(data, accounts):
    if set(data) - {"action", "session_id", "session", "format"}:
        raise ValueError("Exporting takes a session, a session id and a format.")
    session_id = data.get("session_id")
    if session_id is not None:
        if not isinstance(session_id, str):
            raise ValueError("That session is no longer available.")
        session = accounts.get_chat(session_id)
    else:
        session = data.get("session")
        if not isinstance(session, dict):
            raise ValueError("Choose a saved session to export, or send the session itself.")
    return export_session(session, data.get("format"))
