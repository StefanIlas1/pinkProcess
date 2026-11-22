"""Build every figure and number in the paper from the committed evidence files."""

import json
import re
import subprocess
import sys
from math import comb, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
TARGET = 326
COLUMN = (3.5, 2.4)
FULL = (7.16, 2.6)

TERMES_STYLES = ("Regular", "Bold", "Italic", "BoldItalic")
FONT_CACHE = ROOT / "build/paper-fonts"


def truetype_copy(source):
    target = FONT_CACHE / (source.stem + ".ttf")
    if target.exists():
        return target
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable
    font = TTFont(source)
    order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    glyf = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = {}
    for name in order:
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf
    del font["CFF "]
    for name in order:
        glyph = glyf.glyphs[name]
        glyph.recalcBounds(glyf)
        advance, _ = font["hmtx"][name]
        font["hmtx"][name] = (advance, getattr(glyph, "xMin", 0))
    maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for field in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
                  "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, field, 1 if field == "maxZones" else 0)
    font["maxp"] = maxp
    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = order
    font.sfntVersion = "\000\001\000\000"
    FONT_CACHE.mkdir(parents=True, exist_ok=True)
    font.save(target)
    return target


TERMES_SOURCES = [path for style in TERMES_STYLES for path in
                  (Path.home() / "Library/Caches/TectonicProject.Tectonic/bundles/data").glob(f"*/TeXGyreTermesX-{style}.otf")]
TERMES_FILES = [truetype_copy(path) for path in TERMES_SOURCES]
for termes_file in TERMES_FILES:
    font_manager.fontManager.addfont(str(termes_file))
BODY_FONT = "TeX Gyre TermesX" if TERMES_FILES else "Times New Roman"

plt.rcParams.update({
 "font.family": "serif", "font.serif": [BODY_FONT, "Times New Roman", "Times", "Nimbus Roman"],
 "mathtext.fontset": "custom", "mathtext.rm": BODY_FONT, "mathtext.it": f"{BODY_FONT}:italic",
 "mathtext.bf": f"{BODY_FONT}:bold", "mathtext.fallback": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
 "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
 "legend.fontsize": 7.5, "legend.frameon": False,
 "axes.edgecolor": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
 "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 4,
 "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
 "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
BLACK, DARK, MID, LIGHT = "black", "0.3", "0.55", "0.85"
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]
HATCHES = ["", "///", "...", "xx"]
SMALL = 7

RECORDED = {
    "research/paper-notebook.md": {
        "classifierReplayScores": 17920,
        "replayActivations": 32,
        "liveSelectiveScores": 294912,
        "liveSelectivePrompts": 6,
        "operatorScores": 49152,
        "projectionArrayScores": 842752,
        "classifierLuts": 1279,
        "classifierRegisters": 660,
        "classifierBram": 8,
        "classifierSlackNs": 2.071,
        "classifierHoldNs": 0.095,
        "missedTimingNs": 5.991,
        "reducerMissNs": 9.748,
        "classifierDsp": 0,
        "checkpointBytes": 425708,
        "storiesLayers": 5,
        "storiesHidden": 64,
        "storiesVocab": 512,
        "astraConstraintsMet": 56,
        "fixedPointPrompts": 7,
        "fixedPointDiverged": 2,
        "walkPhases": 3,
        "fixturePrompts": 2,
        "fixtureConditions": 4,
        "shapeChecks": 2,
        "bundleBytes": 701,
        "coldSuiteSeconds": 119.8,
        "storyStepSeconds": 0.94,
        "firstFlanPass": 0,
        "flanCases": 12,
        "replacementDirectPass": 12,
        "replacementLexicalPass": 9,
        "historySteps": 360,
        "historyCases": 15,
        "historyActive": 9,
        "historyArchived": 12,
        "historyOmittedZero": 5,
        "historyTargeted": 3,
        "flanEditBytes": 380,
        "flanEditRow": 852,
        "flanRowBytes": 512,
        "residentMiB": 16,
        "vocabSegments": 8,
        "weightUploadBeforeGB": 1.95,
        "weightUploadAfterGB": 1.14,
        "reducerRandomTests": 10000,
        "nativeQuantValues": 4854,
        "astraTopTwoLm": 6,
    },
    "docs/results.md": {
        "pollSecondsExact": 0.5,
        "pollSecondsQueue": 1,
        "targetSeconds": 10,
        "recordingReadings": 60,
        "verifyWidth": 16,
        "rateCandidateEarlyWarnings": 1,
    },
    "device datasheet": {
        "deviceLuts": 53200,
        "deviceBram": 140,
        "deviceDsp": 220,
        "armCores": 2,
    },
}


def load(path):
    return json.loads((ROOT / path).read_text())


def thousands(value):
    return f"{value:,}".replace(",", "{,}")


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_text(k, n):
    low, high = wilson(k, n)
    return f"[{low:.2f}, {high:.2f}]"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def token_names():
    data = (ROOT / "build/lm-inputs/tok512.bin").read_bytes()
    offset, names = 4, []
    while offset < len(data):
        length = int.from_bytes(data[offset + 4:offset + 8], "little")
        names.append(data[offset + 8:offset + 8 + length].decode("utf-8", "replace"))
        offset += 8 + length
    return names


def single_token_names():
    sentence_starts = {"The", "She", "They", "One", "Once"}
    return [name.strip() for name in token_names() if name.startswith(" ") and len(name.strip()) > 2
            and name.strip()[0].isupper() and name.strip().isalpha() and name.strip() not in sentence_starts]


def save(fig, name):
    fig.savefig(FIGURES / name, bbox_inches=fig.bbox_inches, metadata={"CreationDate": None})
    plt.close(fig)


def recorded_numbers(numbers):
    for values in RECORDED.values():
        for key, value in values.items():
            numbers[key] = thousands(value) if isinstance(value, int) and value >= 1000 else str(value)
    device = RECORDED["device datasheet"]
    notebook = RECORDED["research/paper-notebook.md"]
    report = (ROOT / "build/projection-queue-qualified/utilization.rpt").read_text()
    used = {name: int(re.search(rf"\| {name}\s*\|\s*(\d+)", report).group(1))
            for name in ("Slice LUTs", "Block RAM Tile", "DSPs")}
    final = {"finalLuts": used["Slice LUTs"], "finalBram": used["Block RAM Tile"], "finalDsp": used["DSPs"]}
    numbers.update({key: thousands(value) if value >= 1000 else str(value) for key, value in final.items()})
    numbers["finalLutShare"] = f"{100 * final['finalLuts'] / device['deviceLuts']:.0f}"
    numbers["finalLutFree"] = f"{100 - 100 * final['finalLuts'] / device['deviceLuts']:.0f}"
    numbers["finalBramShare"] = f"{100 * final['finalBram'] / device['deviceBram']:.0f}"
    numbers["finalDspShare"] = f"{100 * final['finalDsp'] / device['deviceDsp']:.1f}"
    authoring = load("build/lm001/authoring.json")
    numbers["fitConstraints"] = str(len(authoring["examples"]))
    astra = load("build/lm001/astra-response.json")["row"]
    numbers["astraChangedValues"] = str(sum(a != b for a, b in zip(astra, authoring["original_row"])))
    fixture = load("build/pynq/lm001-board-fixture.json")
    numbers["fixtureScores"] = thousands(sum(case["matched_scores_if_on_board"] for condition in fixture["conditions"].values()
                                             for case in condition["cases"]))
    middle = load("kit/evidence/physical-systolic-projection.json")
    numbers.update({"middleLuts": thousands(middle["physical_overlay_resources"]["lut"]),
                    "middleBram": str(middle["physical_overlay_resources"]["bram_tiles"]),
                    "middleDsp": str(middle["physical_overlay_resources"]["dsp"]),
                    "middleSlackNs": f"{middle['timing']['setup_slack_ns']:.3f}",
                    "middleHoldNs": f"{middle['timing']['hold_slack_ns']:.3f}"})
    pool = load("build/pool-study/observed/report.json")["conditions"]["pool-logistic-v1"]["total"]
    numbers.update({"poolEpisodes": str(pool["episodes"]), "poolWarned": str(pool["warned"]),
                    "poolMissed": str(pool["missed"]), "poolFalseAlerts": str(pool["false_alerts"])})
    numbers["classifierLutShare"] = f"{100 * notebook['classifierLuts'] / device['deviceLuts']:.1f}"
    numbers["rowBytes"] = str(notebook["storiesHidden"])


def walkthrough_figure(numbers):
    report = load("build/pynq/live-pynq-selective.json")
    prompt = report["prompt"]
    phases = report["phases"]
    rows = []
    for phase, label in (("original", "Base weights"), ("edited", "Row 326 replaced"), ("restored", "Rolled back")):
        text = phases[phase]["generated_text"]
        continuation = text[len(prompt):].strip()
        name, rest = continuation.split(".", 1)
        rows.append((label, name, rest.strip(), phases[phase]["observed_row_sha256"][:8],
                     phases[phase]["matched_scores"], phases[phase]["forward_passes"]))

    fig, ax = plt.subplots(figsize=COLUMN, layout="constrained")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")
    ax.text(0, 68, "Prompt:", fontsize=SMALL, fontweight="bold", va="center")
    ax.text(15, 68, f"\u201c{prompt}\u201d", fontsize=SMALL, va="center", style="italic")
    bottoms = (48, 25, 2)
    for (label, name, rest, row_hash, matched, passes), bottom, edited in zip(rows, bottoms, (False, True, False)):
        ax.add_patch(Rectangle((0, bottom), 100, 14, facecolor="white", edgecolor=BLACK, linewidth=0.7))
        if edited:
            ax.add_patch(Rectangle((0, bottom), 2.4, 14, facecolor=LIGHT, edgecolor=BLACK, linewidth=0.7, hatch="////"))
        ax.text(5, bottom + 9.6, label, fontsize=SMALL + 0.5, fontweight="bold", va="center")
        ax.text(5, bottom + 4.0, f"row 326 SHA-256 {row_hash}\u2026   {matched:,} of {matched:,} scores match",
                fontsize=SMALL, va="center")
        ax.text(97, bottom + 9.6, f"\u2026named {name}.", fontsize=9, va="center", ha="right", fontweight="bold")
    for (start_y, end_y), caption in zip(((47.5, 39.5), (24.5, 16.5)),
                                         (f"apply 64-byte row: {report['row_update_ms']['apply']:.2f} ms",
                                          f"restore original row: {report['row_update_ms']['restore']:.2f} ms")):
        ax.add_patch(FancyArrowPatch((22, start_y), (22, end_y), arrowstyle="-|>", mutation_scale=8, color=BLACK,
                                     linewidth=0.8, shrinkA=0, shrinkB=0))
        ax.text(25, (start_y + end_y) / 2, caption, fontsize=SMALL, va="center")
    save(fig, "walkthrough.pdf")
    numbers.update({
        "walkApplyMs": f"{report['row_update_ms']['apply']:.2f}",
        "walkRestoreMs": f"{report['row_update_ms']['restore']:.2f}",
        "walkPhaseScores": thousands(phases["original"]["matched_scores"]),
        "walkPasses": str(phases["original"]["forward_passes"]),
        "walkPreserved": str(report["selectivity"]["continuation_tokens_preserved"]),
        "walkLilySum": thousands(report["selectivity"]["original_best_other_sum_at_first_sample"]),
        "walkClassifierMs": f"{phases['original']['median_classifier_ms']:.2f}",
        "lilyToken": str(report["first_sampled_token_ids"][0]), "timToken": str(report["first_sampled_token_ids"][1]),
        "walkTimScore": thousands(report["target_sums_at_first_sample"][1]),
        "walkTimScoreBefore": thousands(report["target_sums_at_first_sample"][0]),
        "walkStorySeconds": f"{phases['edited']['generation_wall_seconds']:.2f}",
        "walkRowDiff": str(sum(a != b for a, b in zip(bytes.fromhex(report["edited_row_hex"]),
                                                     bytes.fromhex(load("build/lm001/conventional.patch.json")["payload_hex"])))),
        "walkFitConstraints": str(report["selectivity"]["fit_constraints"]),
    })


def box(ax, x, y, w, h, title, lines=(), fill="white", hatch=None, bold=True, size=SMALL):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=BLACK, linewidth=0.7, hatch=hatch))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal")
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 7.6 - 4.3 * i, line, ha="center", va="center", fontsize=SMALL)


def arrow(ax, start, end, both=False, label=None, offset=(0, 0), ha="center"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<|-|>" if both else "-|>", mutation_scale=7,
                                 color=BLACK, linewidth=0.7, shrinkA=0, shrinkB=0))
    if label:
        ax.text((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1], label, fontsize=SMALL,
                ha=ha, va="center")


def system_figure(numbers):
    exact = load("kit/evidence/physical-exact-backend-latency.json")
    floor = load("kit/evidence/diagnosis-latency-floor.json")
    fig, ax = plt.subplots(figsize=COLUMN, layout="constrained")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.axis("off")
    box(ax, 0, 50, 25, 24, "Mac app", ("question", "CSV upload", "answers", "row edits"))
    arrow(ax, (25, 61), (34, 61), both=True)
    ax.text(29.5, 64.5, "USB", fontsize=SMALL, ha="center", va="bottom")
    ax.add_patch(Rectangle((34, 0), 66, 80, facecolor="white", edgecolor=BLACK, linewidth=0.7, linestyle=(0, (3, 2))))
    ax.text(67, 76, "PYNQ-Z2 board (Zynq-7020)", ha="center", va="center", fontsize=SMALL + 0.5, fontweight="bold")
    box(ax, 38, 55, 58, 17, f"ARM Cortex-A9 \u00d72, {exact['cpu_mhz']} MHz",
        ("tokenize, attention, decoding,", "INT8 packing, row edits"))
    box(ax, 38, 21, 58, 27, f"Programmable logic, {exact['fabric_mhz']} MHz")
    for i, title in enumerate(("8\u00d78 INT8 array", "command queue", "vocab reducer (opt-in)", "telemetry scorer")):
        col, row = i % 2, i // 2
        ax.add_patch(Rectangle((40 + col * 28, 32.5 - row * 10), 26, 8, facecolor="white", hatch="////" if i == 0 else None,
                               edgecolor=BLACK, linewidth=1.1 if i == 0 else 0.6))
        if i == 0:
            ax.add_patch(Rectangle((42.6, 34.3), 20.8, 4.4, facecolor="white", edgecolor="none"))
        ax.text(40 + col * 28 + 13, 36.5 - row * 10, title, ha="center", va="center", fontsize=SMALL)
    box(ax, 38, 3, 58, 12, "DDR memory", ("weights and activations",))
    arrow(ax, (67, 55), (67, 48), both=True)
    ax.text(69, 51.5, "AXI", fontsize=SMALL, ha="left", va="center")
    arrow(ax, (67, 21), (67, 15), both=True)
    ax.text(69, 18, "one 64-bit read per clock", fontsize=SMALL, ha="left", va="center")
    save(fig, "system.pdf")
    bandwidth = 8 * exact["fabric_mhz"] * 1e6
    numbers.update({
        "cpuMhz": str(exact["cpu_mhz"]), "fabricMhz": str(exact["fabric_mhz"]),
        "readBandwidthMB": f"{bandwidth / 1e6:.0f}",
        "macLanes": str(floor["projection_array"]["mac_lanes"]),
    })


CASES = ("recovery", "growing", "healthy", "new")
CASE_LABELS = {"recovery": "Recovered", "growing": "Growing", "healthy": "Zero waiting", "new": "New wording"}
TIMERS = {"handoff": ("xx", LIGHT, "paired connection with handoff"), "board": ("///", "white", "board-local"),
          "mac": ("", "white", "Mac backend")}


def latency_numbers():
    systolic = load("kit/evidence/physical-systolic-diagnosis.json")
    cache = load("kit/evidence/physical-projection-cache.json")
    resident = load("kit/evidence/physical-projection-resident-head.json")
    queue = load("kit/evidence/physical-projection-queue-comparison.json")
    ladder = [
        ("Row-at-\na-time\nclassifier", systolic["baseline"]["complete_seconds"], None, "handoff"),
        ("Array,\ntwo query\nlayers", systolic["block_zero_queries"]["complete_seconds"], None, "board"),
        ("Array,\nall\nmatrices", systolic["all_learned_matrices"]["complete_seconds"], None, "board"),
        ("C++\npacking", cache["cases"][0]["complete_seconds"], cache["cases"][1]["complete_seconds"], "mac"),
        ("Resident\nvocab\nhead", resident["cases"][0]["complete_seconds"], resident["cases"][1]["complete_seconds"], "mac"),
        ("Queue +\nFPGA\nreducer", queue["queue_and_argmax"][0]["complete_seconds"], queue["queue_and_argmax"][1]["complete_seconds"], "mac"),
        ("Queue +\nsoftware\npick", queue["queue_only_saved"][0]["complete_seconds"], queue["queue_only_saved"][1]["complete_seconds"], "mac"),
    ]
    three = load("kit/evidence/open-question-three-physical.json")["cases"]
    verified = load("kit/evidence/open-question-verified-physical.json")["cases"]
    controls = load("kit/evidence/verified-decoder-weight-controls-physical.json")["new_wording"]
    exact = load("kit/evidence/physical-exact-backend-latency.json")
    scalar = {c: three[c]["elapsed_wall_seconds"] for c in CASES[:3]} | {"new": controls["scalar"]["elapsed_wall_seconds"]}
    verify = {c: verified[c]["run"]["elapsed_wall_seconds"] for c in CASES[:3]} | {"new": controls["verified"]["elapsed_wall_seconds"]}
    baseline = {c: exact["cases"][c]["baseline"]["complete_seconds"] for c in CASES}
    stage_runs = {name: exact["stages"][name]["runs"] for name in ("ppfusion1", "ppcombined1", "ppcombined2")}
    stages = [("Scalar\ndecoder", scalar), ("Verified\nproposals", verify), ("v2\nbaseline", baseline),
              ("Batched\nfusion", {c: stage_runs["ppfusion1"][c]["complete_seconds"] for c in CASES}),
              ("NEON +\nprompt\ncopy", {c: stage_runs["ppcombined1"][c]["complete_seconds"] for c in CASES}),
              ("Native\nquantize", {c: stage_runs["ppcombined2"][c]["complete_seconds"] for c in CASES})]
    first_text = {c: stage_runs["ppcombined2"][c]["first_partial_seconds"] for c in CASES}
    return ladder, stages, first_text, exact, queue


def latency_figure(numbers):
    ladder, stages, first_text, exact, queue = latency_numbers()
    floor = load("kit/evidence/diagnosis-latency-floor.json")
    ddr_floor = floor["projection_ddr_reads"]["cached_read_only_floor_s_at_50mhz"]
    mac_floor = floor["projection_array"]["full_answer_mac_only_floor_s"]

    fig, (left, right) = plt.subplots(1, 2, figsize=FULL, layout="constrained", gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(ladder))
    values = np.array([v for _, v, _, _ in ladder])
    for xi, (_, value, warm, method) in zip(x, ladder):
        hatch, fill, _ = TIMERS[method]
        left.bar(xi, value, width=0.6, facecolor=fill, edgecolor=BLACK, linewidth=0.7, hatch=hatch, zorder=2)
        left.text(xi, value * 1.15, f"{value:.1f}", ha="center", va="bottom", fontsize=SMALL)
        if warm is not None:
            left.plot(xi + 0.4, warm, marker="o", mfc="white", mec=BLACK, ms=3.2, mew=0.7, ls="none", zorder=4)
    left.set_yscale("log")
    left.set_ylim(1, 5000)
    left.set_xlim(-0.55, len(ladder) + 0.95)
    left.axhline(ddr_floor, color=BLACK, lw=0.8, ls="--", zorder=3)
    left.axhline(mac_floor, color=BLACK, lw=0.8, ls=":", zorder=3)
    for level, text in ((ddr_floor, f"memory-read floor\n{ddr_floor:.2f} s"), (mac_floor, f"arithmetic-only floor\n{mac_floor:.2f} s")):
        left.text(len(ladder) - 0.45, level * 1.1, text, ha="left", va="bottom", fontsize=SMALL, linespacing=1.1)
    left.set_xticks(x, [label for label, _, _, _ in ladder], fontsize=SMALL)
    left.set_ylabel("Complete answer [s], log scale")
    handles = [Rectangle((0, 0), 1, 1, facecolor=TIMERS[m][1], edgecolor=BLACK, linewidth=0.7, hatch=TIMERS[m][0],
                         label=f"timed: {TIMERS[m][2]}") for m in ("handoff", "board", "mac")]
    handles.append(plt.Line2D([], [], marker="o", mfc="white", mec=BLACK, ms=3.2, mew=0.7, ls="none", label="warm repeat"))
    left.legend(handles=handles, loc="upper right", handlelength=1.4, fontsize=SMALL)
    left.set_title("(a)", loc="left", fontsize=8, fontweight="bold", pad=6)

    xs = np.arange(len(stages))
    offsets = (-0.09, -0.03, 0.03, 0.09)
    for index, case in enumerate(CASES):
        ys = [stage[1][case] for stage in stages]
        xj = xs + offsets[index]
        right.plot(xj[:2], ys[:2], color=MID, ls=LINESTYLES[index], marker=MARKERS[index], mfc="white", mec=BLACK,
                   mew=0.7, lw=0.9, ms=3.5)
        right.plot(xj[2:], ys[2:], color=BLACK, ls=LINESTYLES[index], marker=MARKERS[index], mfc=BLACK, mec=BLACK,
                   lw=0.9, ms=3.5, label=CASE_LABELS[case])
    right.axhline(RECORDED["docs/results.md"]["targetSeconds"], color=BLACK, lw=0.8, ls=":")
    right.text(5.3, 10.6, "10 s target", fontsize=SMALL, ha="right", va="bottom")
    right.axvline(1.5, color=MID, lw=0.6, ls="-")
    right.text(0.5, 51, "original checkpoint", fontsize=SMALL, ha="center", style="italic")
    right.text(3.6, 51, "Queue history v2 checkpoint", fontsize=SMALL, ha="center", style="italic")
    right.set_xticks(xs, [stage[0] for stage in stages], fontsize=SMALL)
    right.set_ylim(0, 55)
    right.set_xlim(-0.35, 5.35)
    right.set_ylabel("Complete answer [s]")
    right.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), handlelength=2.6)
    right.set_title("(b)", loc="left", fontsize=8, fontweight="bold", pad=6)
    save(fig, "latency.pdf")

    final = stages[-1][1]
    warm = exact["stages"]["ppcombined2"]["warm_recovery_repeat"]["complete_seconds"]
    restart = exact["stages"]["ppqualified"]["runs"]["new"]["complete_seconds"]
    every_run = list(final.values()) + [warm, restart]
    best = exact["stages"]["ppcombined2"]["runs"]
    phases = best["recovery"]["phases"]
    installed = load("kit/evidence/installed-exact-backend-latency.json")["cases"]
    shown = {case: int(re.search(r"Complete answer ([\d,]+) ms", installed[case]["after_ui"]).group(1).replace(",", "")) / 1000
             for case in ("healthy", "sample")}
    numbers.update({
        "latBaseline": f"{values[0]:.1f}", "latTwoQuery": f"{values[1]:.1f}", "latArray": f"{values[2]:.1f}",
        "latCache": f"{values[3]:.1f}", "latResident": f"{values[4]:.1f}", "latQueue": f"{values[5]:.1f}",
        "latQueueSoft": f"{values[6]:.1f}",
        "latCacheWarm": f"{ladder[3][2]:.1f}", "latResidentWarm": f"{ladder[4][2]:.1f}", "latQueueWarm": f"{ladder[5][2]:.1f}",
        "latQueueSoftWarm": f"{ladder[6][2]:.1f}",
        "latBaselineMinutes": f"{values[0] / 60:.0f}",
        "latLadderSpeedup": f"{values[0] / values[5]:.0f}",
        "latArraySpeedup": f"{values[0] / values[2]:.1f}",
        "latBestMin": f"{min(final.values()):.2f}", "latBestMax": f"{max(final.values()):.2f}",
        "latBestMedian": f"{np.median(list(final.values())):.2f}",
        "latBestRounded": f"{np.median(list(final.values())):.0f}",
        "latAllMin": f"{min(every_run):.1f}", "latAllMax": f"{max(every_run):.1f}", "latAllRuns": str(len(every_run)),
        "latScalarMin": f"{min(stages[0][1].values()):.2f}", "latScalarMax": f"{max(stages[0][1].values()):.2f}",
        "latVerifyMin": f"{min(stages[1][1].values()):.2f}", "latVerifyMax": f"{max(stages[1][1].values()):.2f}",
        "latVtwoMin": f"{min(stages[2][1].values()):.2f}", "latVtwoMax": f"{max(stages[2][1].values()):.2f}",
        "firstTextMin": f"{min(first_text.values()):.2f}", "firstTextMax": f"{max(first_text.values()):.2f}",
        "firstTextNew": f"{first_text['new']:.2f}",
        "firstTextNewRestart": f"{exact['stages']['ppqualified']['runs']['new']['first_partial_seconds']:.2f}",
        "latWarmRepeat": f"{warm:.2f}",
        "latRecoveryFinal": f"{final['recovery']:.2f}",
        "latNewFinal": f"{final['new']:.2f}",
        "latNewRestart": f"{restart:.2f}",
        "latWarmDelta": f"{final['recovery'] - warm:.1f}",
        "latRestartDelta": f"{restart - final['new']:.1f}",
        "latInstalledHealthy": f"{shown['healthy']:.2f}", "latInstalledSample": f"{shown['sample']:.2f}",
        "phaseEncoder": f"{phases['encoder_stack_ns'] / 1e9:.2f}", "phaseDecoder": f"{phases['decoder_stacks_ns'] / 1e9:.2f}",
        "phaseVocabulary": f"{phases['vocabulary_ns'] / 1e9:.2f}",
        "phaseRoundtrip": f"{phases['device_roundtrip_ns'] / 1e9:.2f}",
        "phasePackWeights": f"{phases['pack_weights_ns'] / 1e9:.2f}",
        "phaseCommands": str(phases["commands"]),
        "projectionsCurrentMin": str(min(r["projections"] for r in best.values())),
        "projectionsCurrentMax": str(max(r["projections"] for r in best.values())),
        "projectionsFloorCase": thousands(load("kit/evidence/physical-systolic-diagnosis.json")["all_learned_matrices"]["fabric_projections"]),
        "liveMin": str(min(r["live_submissions"] for r in best.values())),
        "liveMax": str(max(r["live_submissions"] for r in best.values())),
        "floorMac": f"{mac_floor:.2f}", "floorDdr": f"{ddr_floor:.2f}",
        "floorSubmissions": thousands(floor["total_submissions"]),
        "weightGB": f"{floor['projection_ddr_reads']['weight_bytes'] / 1e9:.2f}",
        "activationNoReuseMB": f"{floor['projection_ddr_reads']['activation_bytes_without_block_ram_reuse'] / 1e6:.0f}",
        "activationReuseMB": f"{floor['projection_ddr_reads']['activation_bytes_with_block_ram_reuse'] / 1e6:.1f}",
        "promptTokens": str(floor["input_tokens"]), "decoderSteps": str(floor["decoder_steps_including_eos"]),
        "queueScores": thousands(queue["isolated_board_qualification"]["integer_queue_scores_checked"]),
        "queueSlackNs": f"{queue['routed_setup_slack_ns']:.3f}", "queueHoldNs": f"{queue['routed_hold_slack_ns']:.3f}",
        "tieWinner": thousands(queue["isolated_board_qualification"]["cross_segment_tie_winner"]),
    })
    numbers["latencyRows"] = latency_rows(stages, first_text)
    numbers["stageRows"] = stage_rows(stages)
    numbers["ladderRows"] = ladder_rows(ladder)


def latency_rows(stages, first_text):
    rows = []
    for case in CASES:
        cells = [f"{stage[1][case]:.2f}" for stage in stages]
        rows.append(f"{CASE_LABELS[case]} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[5]} & {first_text[case]:.2f} \\\\")
    return "\n".join(rows)


def stage_rows(stages):
    return "\n".join(f"{CASE_LABELS[case]} & " + " & ".join(f"{stage[1][case]:.2f}" for stage in stages) + " \\\\"
                     for case in CASES)


def prompt_rows(extended, protocol):
    def tex(prompt):
        return prompt.replace("&", "\\&")
    targets = extended["protocol"]["targets"]
    controls = extended["protocol"]["controls"]
    rows = [f"{i + 1} & {tex(t)} & {tex(c)}{'$^\\dagger$' if i < 11 else ''} \\\\" for i, (t, c) in enumerate(zip(targets, controls))]
    lm_targets = [c["prompt"] for c in protocol["cases"] if c["group"] == "target"]
    lm_controls = [c["prompt"] for c in protocol["cases"] if c["group"] == "control"]
    rows.append("\\midrule")
    rows += [f"L{i + 1} & {tex(t)} & {tex(c)} \\\\" for i, (t, c) in enumerate(zip(lm_targets, lm_controls))]
    return "\n".join(rows)


def ladder_rows(ladder):
    rows = []
    for label, cold, warm, method in ladder:
        name = label.replace("-\n", "-").replace("\n", " ")
        timer = TIMERS[method][2]
        warm_text = f"{warm:.1f}" if warm is not None else "{--}"
        rows.append(f"{name} & {cold:.1f} & {warm_text} & {timer} \\\\")
    return "\n".join(rows)


def utilization_figure(numbers):
    queue = load("kit/evidence/physical-projection-queue-comparison.json")
    captures = {c["mode"]: c for c in queue["isolated_board_qualification"]["real_capture_counters"]}
    projection = load("kit/evidence/physical-systolic-projection.json")["cases"]
    modes = ("encoder", "decoder")
    mac = [captures[m]["mac_feed_cycles"] / captures[m]["elapsed_cycles"] for m in modes]
    ddr = [captures[m]["ddr_read_beats"] / captures[m]["elapsed_cycles"] for m in modes]
    fig, ax = plt.subplots(figsize=COLUMN, layout="constrained")
    x = np.arange(len(modes))
    width = 0.34
    for offset, series, hatch, fill, label in ((-width / 2, mac, "///", "white", "multipliers had data"),
                                               (width / 2, ddr, "...", LIGHT, "memory delivered a 64-bit word")):
        ax.bar(x + offset, [100 * v for v in series], width=width, facecolor=fill, edgecolor=BLACK, linewidth=0.7,
               hatch=hatch, label=label)
        for xi, v in zip(x + offset, series):
            ax.text(xi, 100 * v + 1.2, f"{100 * v:.1f}%", ha="center", va="bottom", fontsize=SMALL)
    shapes = [captures[m]["shape"] for m in modes]
    ax.set_xticks(x, [f"Encoder projection\n{s['batch']}×{s['depth']}→{s['rows']}" for s in shapes[:1]] +
                  [f"Decoder projection\n{s['batch']}×{s['depth']}→{s['rows']}" for s in shapes[1:]])
    ax.set_ylabel("Share of elapsed cycles [%]")
    ax.set_ylim(0, 45)
    ax.set_xlim(-0.6, 1.6)
    ax.legend(loc="upper center", ncol=2, handlelength=1.6, columnspacing=1.2)
    save(fig, "utilization.pdf")

    def spread(mode, phase):
        samples = [s["projection_ns"] / 1e6 for s in projection[mode][phase]["samples"]]
        return np.median(samples), min(samples), max(samples), len(samples)

    for mode, key in (("encoder", "Enc"), ("decoder", "Dec")):
        for phase, label in (("before", "Before"), ("after", "After")):
            median, low, high, n = spread(mode, phase)
            numbers[f"proj{key}{label}"] = f"{median:.1f}" if median < 1000 else thousands(round(median))
            numbers[f"proj{key}{label}Range"] = f"{low:.1f} to {high:.1f}" if median < 1000 else f"{thousands(round(low))} to {thousands(round(high))}"
            numbers["projRuns"] = str(n)
    numbers.update({
        "macFeedEnc": f"{100 * mac[0]:.1f}", "macFeedDec": f"{100 * mac[1]:.1f}",
        "ddrBeatEnc": f"{100 * ddr[0]:.1f}", "ddrBeatDec": f"{100 * ddr[1]:.1f}",
    })


def edit_figure(numbers):
    extended = load("paper/experiments/lm001-extended.json")
    sweep = load("paper/experiments/contrastive-sweep.json")
    contrastive = load("paper/experiments/contrastive-fit.json")["summary"]
    controls = extended["protocol"]["controls"]
    targets_total = len(extended["protocol"]["targets"])
    naming = set(controls[:11])
    naming_total = len(naming)
    other_total = len(controls) - naming_total

    def summarise(result):
        hits = sum(c["group"] == "target" and c["first_token_id"] == TARGET for c in result["cases"])
        leaks = sum(c["group"] == "control" and c["prompt"] in naming and not c["unchanged"] for c in result["cases"])
        other = sum(c["group"] == "control" and c["prompt"] not in naming and not c["unchanged"] for c in result["cases"])
        return hits, leaks, other

    fig, (left, right) = plt.subplots(1, 2, figsize=FULL, layout="constrained", gridspec_kw={"width_ratios": [1.1, 1]})
    first_leak = {}
    for index, (author, name) in enumerate((("conventional", "Conventional"), ("astra", "Astra"))):
        rows = [r for r in extended["results"] if r["author"] == author]
        alpha = np.array([r["alpha"] for r in rows])
        stats = np.array([summarise(r) for r in rows])
        shade = BLACK if index == 0 else MID
        left.plot(alpha, stats[:, 0] / targets_total, color=shade, ls="-", marker=MARKERS[index], markevery=2,
                  mfc="white" if index else shade, mec=shade, lw=1.1)
        left.plot(alpha, stats[:, 1] / naming_total, color=shade, ls="--", marker=MARKERS[index], markevery=2,
                  mfc="white" if index else shade, mec=shade, lw=1.1)
        at_one = stats[alpha == 1.0][0]
        top_two = next(r for r in rows if r["alpha"] == 1.0)["target_top2"]
        numbers.update({f"{author}Hits": str(at_one[0]), f"{author}Leaks": str(at_one[1]),
                        f"{author}Other": str(at_one[2]), f"{author}MaxOther": str(stats[:, 2].max()),
                        f"{author}TopTwo": str(top_two)})
        leak_alpha = alpha[np.argmax(stats[:, 1] > 0)]
        hit_alpha = alpha[np.argmax(stats[:, 0] >= stats[:, 0].max() / 2)]
        first_leak[author] = (leak_alpha, hit_alpha)
        numbers[f"{author}FirstLeakAlpha"] = f"{leak_alpha:.1f}"
        numbers[f"{author}HalfHitAlpha"] = f"{hit_alpha:.1f}"
        numbers[f"{author}MaxHits"] = str(stats[:, 0].max())
        numbers[f"{author}MaxLeaks"] = str(stats[:, 1].max())
    left.axvline(1.0, color=MID, lw=0.6, zorder=0)
    left.text(0.97, 1.0, r"frozen row ($\alpha = 1$)", fontsize=SMALL, ha="right", va="top")
    left.set_xlabel(r"Edit strength $\alpha$  (row $= r_0 + \alpha\,(r_1 - r_0)$)")
    left.set_ylabel("Share of prompts")
    left.set_ylim(-0.03, 1.08)
    left.set_xlim(0, 2)
    handles = [
        plt.Line2D([], [], color=BLACK, ls="-", marker="o", mfc=BLACK, label="Conventional: Tim on targets"),
        plt.Line2D([], [], color=BLACK, ls="--", marker="o", mfc=BLACK, label="Conventional: renames"),
        plt.Line2D([], [], color=MID, ls="-", marker="s", mfc="white", mec=MID, label="Astra: Tim on targets"),
        plt.Line2D([], [], color=MID, ls="--", marker="s", mfc="white", mec=MID, label="Astra: renames"),
    ]
    left.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.02, 0.02), handlelength=2.0, fontsize=SMALL, handletextpad=0.5)
    left.set_title("(a)", loc="left", fontsize=8, fontweight="bold", pad=6)

    hit_floor = (17.5) / targets_total
    leak_ceiling = 2.5 / naming_total
    right.add_patch(Rectangle((-0.05, hit_floor), leak_ceiling + 0.05, 1.1 - hit_floor, facecolor=LIGHT,
                              edgecolor="none", zorder=0))
    right.text(-0.03, 1.08, "pass rule", fontsize=SMALL, va="top", ha="left")
    for positives, marker, fill, label in ((0, "s", "white", "Negatives only"), (3, "o", BLACK, "3 positives + negatives")):
        pts = [r for r in sweep if r["positives"] == positives]
        right.plot([r["extended_naming_changed"] / naming_total for r in pts],
                   [r["extended_target_hits"] / targets_total for r in pts], ls=":" if positives == 0 else "-",
                   color=DARK, lw=0.7, marker=marker, mfc=fill, mec=BLACK, label=label, zorder=3)
    for r in sweep:
        if (r["positives"], r["negatives"]) in ((0, 4), (3, 0), (3, 1), (3, 4)):
            dx, dy = {(0, 4): (6, -3), (3, 0): (4, 5), (3, 1): (2, -12), (3, 4): (-5, -11)}[(r["positives"], r["negatives"])]
            right.annotate(f"{r['positives']}+, {r['negatives']}−", (r["extended_naming_changed"] / naming_total,
                           r["extended_target_hits"] / targets_total), xytext=(dx, dy), textcoords="offset points",
                           fontsize=SMALL, ha="right" if dx < 0 else "left")
    for author, marker, label, dx, dy in (("conventional", "D", "Conventional", 6, 5), ("astra", "^", "Astra", -6, 4)):
        k, j = int(numbers[f"{author}Hits"]), int(numbers[f"{author}Leaks"])
        hit_low, hit_high = wilson(k, targets_total)
        leak_low, leak_high = wilson(j, naming_total)
        xpt, ypt = j / naming_total, k / targets_total
        right.errorbar(xpt, ypt, yerr=[[ypt - hit_low], [hit_high - ypt]], xerr=[[xpt - leak_low], [leak_high - xpt]],
                       fmt=marker, ms=5, mfc="white", mec=BLACK, ecolor=MID, elinewidth=0.6, capsize=1.5, zorder=4)
        right.annotate(label, (xpt, ypt), xytext=(dx, dy), textcoords="offset points", fontsize=SMALL,
                       ha="left" if dx > 0 else "right")
    ck, cj = contrastive["extended_target_hits"], contrastive["extended_naming_changed"]
    hit_low, hit_high = wilson(ck, targets_total)
    leak_low, leak_high = wilson(cj, naming_total)
    right.errorbar(cj / naming_total, ck / targets_total,
                   yerr=[[ck / targets_total - hit_low], [hit_high - ck / targets_total]],
                   xerr=[[cj / naming_total - leak_low], [leak_high - cj / naming_total]], fmt="none", ecolor=MID,
                   elinewidth=0.6, capsize=1.5, zorder=2)
    right.set_xlim(-0.05, 1.05)
    right.set_ylim(0.1, 1.1)
    right.set_xlabel("Renames (share of 11 naming controls)")
    right.set_ylabel("Targets selecting Tim (share of 24)")
    right.legend(loc="lower right", handlelength=2.2)
    right.set_title("(b)", loc="left", fontsize=8, fontweight="bold", pad=6)
    save(fig, "edits.pdf")

    conv = next(r for r in extended["results"] if r["author"] == "conventional" and r["alpha"] == 1.0)
    astr = next(r for r in extended["results"] if r["author"] == "astra" and r["alpha"] == 1.0)
    targets = [c["prompt"] for c in conv["cases"] if c["group"] == "target"]
    hit = lambda result: {c["prompt"]: c["first_token_id"] == TARGET for c in result["cases"]}
    hc, ha = hit(conv), hit(astr)
    b = sum(hc[p] and not ha[p] for p in targets)
    c = sum(ha[p] and not hc[p] for p in targets)
    numbers.update({
        "extTargets": str(targets_total), "extControls": str(len(controls)), "extNaming": str(naming_total),
        "extOther": str(other_total), "extAlphas": str(len(extended["protocol"]["alphas"])),
        "extNewTokens": str(extended["protocol"]["new_tokens"]),
        "alphaMax": f"{max(extended['protocol']['alphas']):.0f}",
        "alphaStep": f"{extended['protocol']['alphas'][1] - extended['protocol']['alphas'][0]:.1f}",
        "extOriginalHits": str(extended["original_target_hits"]),
        "convVsAstraB": str(b), "convVsAstraC": str(c), "convVsAstraP": f"{mcnemar_exact(b, c):.3f}",
        "contrastiveHits": str(ck), "contrastiveLeaks": str(cj),
        "contrastiveOther": str(contrastive["extended_other_changed"]),
        "contrastiveLmHits": str(contrastive["lm001_target_hits"]),
        "contrastiveLmControls": str(contrastive["lm001_controls_unchanged"]),
        "contrastiveConstraints": str(contrastive["constraints"]),
        "contrastiveLone": str(contrastive["row_l1_change"]),
        "positiveAuthoring": "3", "negativeAuthoring": "8",
        "criterionHits": "18", "criterionLeaks": "2", "criterionOther": "0",
    })
    for key, (k, n) in {"convHitCI": (int(numbers["conventionalHits"]), targets_total),
                        "astraHitCI": (int(numbers["astraHits"]), targets_total),
                        "contrHitCI": (ck, targets_total),
                        "convLeakCI": (int(numbers["conventionalLeaks"]), naming_total),
                        "astraLeakCI": (int(numbers["astraLeaks"]), naming_total),
                        "contrLeakCI": (cj, naming_total),
                        "lmConvCI": (5, 6), "lmAstraCI": (3, 6), "lmConvControlCI": (5, 6),
                        "contrLmCI": (contrastive["lm001_target_hits"], 6)}.items():
        numbers[key] = wilson_text(k, n)
    by_key = {(r["positives"], r["negatives"]): r for r in sweep}
    numbers.update({
        "sweepNegOnlyHits": str(by_key[(0, 8)]["extended_target_hits"]),
        "sweepNegOnlyLeaks": str(by_key[(0, 8)]["extended_naming_changed"]),
        "sweepPosOnlyHits": str(by_key[(3, 0)]["extended_target_hits"]),
        "sweepPosOnlyLeaks": str(by_key[(3, 0)]["extended_naming_changed"]),
        "sweepBestHits": str(by_key[(3, 4)]["extended_target_hits"]),
        "sweepBestLeaks": str(by_key[(3, 4)]["extended_naming_changed"]),
        "sweepFits": str(len(sweep)),
        "sweepPassing": str(sum(r["preregistered_success"] for r in sweep)),
        "sweepDistinctPassing": str(len({(r["row_l1_change"], r["extended_target_hits"], r["extended_naming_changed"],
                                          r["extended_other_changed"], r["lm001_target_hits"], r["lm001_controls_unchanged"])
                                         for r in sweep if r["preregistered_success"]})),
        "sweepBaseConstraints": str(by_key[(0, 0)]["constraints"]),
        "constraintsPerNegative": str(by_key[(0, 1)]["constraints"] - by_key[(0, 0)]["constraints"]),
        "sweepBaseLone": str(by_key[(0, 0)]["row_l1_change"]),
        "sweepBaseHits": str(by_key[(0, 0)]["extended_target_hits"]),
        "sweepBaseLeaks": str(by_key[(0, 0)]["extended_naming_changed"]),
    })
    base_cases = {c["prompt"]: c for c in next(r for r in extended["results"] if r["author"] == "conventional" and r["alpha"] == 0.0)["cases"]}
    naming_at_target = [p for p in controls[:naming_total] if base_cases[p]["first_token_id"] == TARGET]
    moved = [c for c in load("paper/experiments/contrastive-fit.json")["cases"] if c["group"] == "control" and not c["unchanged"]]
    numbers.update({"namingAlreadyTim": str(len(naming_at_target)),
                    "namingCanRename": str(naming_total - len(naming_at_target)),
                    "contrastiveMovedToken": str(moved[0]["first_token_id"]) if len(moved) == 1 else "",
                    "timmyToken": str(token_names().index(" Timmy")),
                    "singleTokenNames": str(len(single_token_names()))})
    numbers["promptRows"] = prompt_rows(extended, load("build/lm001/protocol.json"))
    numbers["sweepRows"] = "\n".join(
        f"{r['positives']} & {r['negatives']} & {r['constraints']} & {r['row_l1_change']} & "
        f"{r['extended_target_hits']}/{targets_total} & {r['extended_naming_changed']}/{naming_total} & "
        f"{r['extended_other_changed']}/{other_total} & {r['lm001_target_hits']}/6 & {r['lm001_controls_unchanged']}/6 & "
        f"{'yes' if r['preregistered_success'] else 'no'} \\\\" for r in sweep)


def lm001_numbers(numbers):
    software = load("build/lm001/lm001-software.json")["conditions"]
    pynq = load("build/lm001/lm001-pynq.json")
    protocol = load("build/lm001/protocol.json")
    same = all(software[name]["cases"][i]["sampled_token_ids"] == pynq["conditions"][name]["cases"][i]["sampled_token_ids"]
               for name in software for i in range(len(software[name]["cases"])))
    matched = sum(case.get("matched_scores_if_on_board", 0) for cond in pynq["conditions"].values() for case in cond["cases"])
    runs = sum(len(cond["cases"]) for cond in pynq["conditions"].values())
    numbers.update({"lmBoardMatchesSoftware": "yes" if same else "no", "lmBoardScores": thousands(matched),
                    "lmRuns": str(runs), "lmMaxTokens": str(protocol["max_sampled_tokens"]),
                    "lmTargets": str(sum(c["group"] == "target" for c in protocol["cases"])),
                    "lmControls": str(sum(c["group"] == "control" for c in protocol["cases"])),
                    "lmMinHits": str(protocol["success"]["minimum_targets_selecting_tim"]),
                    "lmMinControls": str(protocol["success"]["minimum_unchanged_control_sequences"])})
    for name, key in (("original", "Orig"), ("prompt_only", "Prompt"), ("conventional", "Conv"), ("astra", "Astra")):
        numbers[f"lm{key}Hits"] = str(software[name]["target_hits"])
        numbers[f"lm{key}Controls"] = str(software[name]["controls_unchanged"])


def quality_numbers(numbers):
    first = load("kit/evidence/blind-diagnosis-w8a8.json")
    replacement = load("kit/evidence/blind-diagnosis-replacement-w8a8.json")
    miss = load("kit/evidence/native-window-projection-quality-miss.json")
    parity = load("kit/evidence/history-prompt-copy-parity.json")
    edit = load("kit/evidence/physical-diagnosis-edit.json")
    numbers.update({
        "firstFlanPass": str(first["cases_passing_every_check"]), "flanCases": str(first["total_cases"]),
        "replacementLexicalPass": str(replacement["cases_passing_every_check"]),
        "missWaiting": str(miss["capture"]["latest"]["waiting"]),
        "missPeak": str(miss["capture"]["peak_waiting"]["waiting"]),
        "verifierBaselinePasses": str(parity["baseline_blocks"]), "verifierPasses": str(parity["prompt_copy_blocks"]),
        "verifierQuestions": str(len(parity["cases"])),
        "flanEditBytes": str(edit["changed_weight_bytes"]), "flanEditRow": str(edit["target_row"] if "target_row" in edit else edit["selected"]["target_row"]),
    })
    output = subprocess.run([sys.executable, "-m", "pinkprocess.fixed_point"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout
    windows = re.search(r"Replayed ([\d,]+) held-out windows", output).group(1).replace(",", "")
    flips = re.search(r"Prediction flips: (\d+)", output).group(1)
    floating = re.search(r"Floating point: warned (\d+)/(\d+)", output)
    fixed = re.search(r"Fixed point: warned (\d+)/(\d+)", output)
    floating_rest = re.search(r"Floating point: warned \d+/\d+, missed (\d+), false alerts (\d+)", output)
    fixed_rest = re.search(r"Fixed point: warned \d+/\d+, missed (\d+), false alerts (\d+)", output)
    numbers.update({"telemetryWindows": thousands(int(windows)), "telemetryFlips": flips,
                    "telemetryFloatWarned": floating.group(1), "telemetryFixedWarned": fixed.group(1),
                    "telemetryEpisodes": floating.group(2),
                    "telemetryFloatMissed": floating_rest.group(1), "telemetryFloatFalse": floating_rest.group(2),
                    "telemetryFixedMissed": fixed_rest.group(1), "telemetryFixedFalse": fixed_rest.group(2),
                    "telemetryMissShare": f"{100 * int(fixed_rest.group(1)) / int(floating.group(2)):.0f}"})
    numbers["verifyWidthOld"] = re.search(r"existing (\d+)-position verifier", parity["execution"]).group(1)
    review = load("kit/evidence/grounded-history-independent-review.json")["review"]
    original_states = re.match(r"(\d+)/(\d+) general cases interpreted current state correctly", review["original"])
    absent = re.search(r"Current count numerical citation absent on ids ([\d, ]+);", review["grounded_360"]).group(1)
    complete = re.search(r"Complete numeric current \+ peak (\d+)/(\d+)", review["grounded_360"])
    numbers.update({"originalStatesRight": original_states.group(1), "originalStatesTotal": original_states.group(2),
                    "historyCountMissing": str(len(absent.split(","))),
                    "historyCompleteNumeric": complete.group(1)})


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    numbers = {}
    recorded_numbers(numbers)
    walkthrough_figure(numbers)
    system_figure(numbers)
    latency_figure(numbers)
    utilization_figure(numbers)
    edit_figure(numbers)
    lm001_numbers(numbers)
    quality_numbers(numbers)
    TABLES.mkdir(parents=True, exist_ok=True)
    for table_name in ("latencyRows", "ladderRows", "sweepRows", "stageRows", "promptRows"):
        (TABLES / f"{table_name}.tex").write_text(numbers.pop(table_name) + "\n")
    lines = [f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in sorted(numbers.items())]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} macros written")


if __name__ == "__main__":
    main()
