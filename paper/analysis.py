"""Build every figure and number in the paper from the committed evidence files."""

import json
from math import comb, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
INK = "#1f2328"
GREY = "#8c959f"
LIGHT = "#d0d7de"
PINK = "#d6336c"
BLUE = "#2f6fb0"
TARGET = 326

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "pdf.fonttype": 42, "axes.titlelocation": "left", "axes.titleweight": "bold",
})


def load(path):
    return json.loads((ROOT / path).read_text())


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def latency_figure(numbers):
    systolic = load("kit/evidence/physical-systolic-diagnosis.json")
    cache = load("kit/evidence/physical-projection-cache.json")
    resident = load("kit/evidence/physical-projection-resident-head.json")
    floor = load("kit/evidence/diagnosis-latency-floor.json")
    ladder = [
        ("Classifier\nonly", systolic["baseline"]["complete_seconds"]),
        ("8×8 array,\nall layers", systolic["all_learned_matrices"]["complete_seconds"]),
        ("C++ pack,\ncache", cache["cases"][0]["complete_seconds"]),
        ("Resident\nvocab head", resident["cases"][0]["complete_seconds"]),
        ("Command\nqueue", 49.872),
    ]
    verified = load("kit/evidence/open-question-verified-physical.json")["cases"]
    exact = load("kit/evidence/physical-exact-backend-latency.json")["stages"]
    cases = ("recovery", "growing", "healthy", "new")
    labels = {"recovery": "Recovered", "growing": "Growing", "healthy": "Healthy", "new": "New wording"}
    scalar = {"recovery": 40.39, "growing": 43.41, "healthy": 42.44, "new": 42.43}
    verify = {c: verified[c]["run"]["elapsed_wall_seconds"] for c in cases[:3]} | {"new": 24.70}
    v2_before = {"recovery": 17.31, "growing": 19.22, "healthy": 18.29, "new": 21.71}
    stages = [("Scalar\ndecoder", scalar), ("Verified\nproposals", verify), ("v2\nbaseline", v2_before),
              ("Batched\nfusion", {c: exact["ppfusion1"]["runs"][c]["complete_seconds"] for c in cases}),
              ("NEON +\ncache", {c: exact["ppcombined1"]["runs"][c]["complete_seconds"] for c in cases}),
              ("Native\nquantization", {c: exact["ppcombined2"]["runs"][c]["complete_seconds"] for c in cases})]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(len(ladder))
    values = np.array([v for _, v in ladder])
    left.bar(x, values, width=0.55, color=[GREY] + [INK] * (len(ladder) - 1))
    for xi, v in zip(x, values):
        left.text(xi, v * 1.12, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    left.set_yscale("log")
    left.set_ylim(1, 6000)
    left.axhline(floor["projection_ddr_reads"]["cached_read_only_floor_s_at_50mhz"], color=PINK, lw=0.8, ls="--",
                 label="DDR weight-read floor (6.2 s)", zorder=0)
    left.axhline(floor["projection_array"]["full_answer_mac_only_floor_s"], color=PINK, lw=0.8, ls=":",
                 label="MAC-only floor (1.9 s)", zorder=0)
    left.legend(loc="upper right", handlelength=2.0)
    left.set_xticks(x, [l for l, _ in ladder])
    left.set_ylabel("Complete answer (s, log)")
    left.set_title("a  One saved incident, original weights")

    xs = np.arange(len(stages))
    markers = {"recovery": "o", "growing": "s", "healthy": "^", "new": "D"}
    for case in cases:
        ys = [s[1][case] for s in stages]
        right.plot(xs[:2], ys[:2], color=GREY, lw=0.9, marker=markers[case], ms=3.2, mfc="white")
        right.plot(xs[2:], ys[2:], color=INK, lw=0.9, marker=markers[case], ms=3.2, label=labels[case])
    right.axhline(10, color=PINK, lw=0.8, ls="--")
    right.text(5.15, 10.6, "10 s target", color=PINK, fontsize=6.5, ha="right")
    right.axvline(1.5, color=LIGHT, lw=0.8)
    right.text(0.5, 44.5, "original checkpoint", color=GREY, fontsize=6.5, ha="center")
    right.text(3.5, 44.5, "Queue history v2 checkpoint", color=INK, fontsize=6.5, ha="center")
    right.set_xticks(xs, [s[0] for s in stages])
    right.set_ylim(0, 48)
    right.set_ylabel("Complete answer (s)")
    right.set_title("b  Three recordings and one new wording")
    right.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), ncol=1, handlelength=1.6)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGURES / "latency.pdf")
    numbers.update({
        "latBaseline": f"{values[0]:.1f}", "latArray": f"{values[1]:.1f}", "latQueue": f"{values[-1]:.1f}",
        "latLadderSpeedup": f"{values[0] / values[-1]:.0f}",
        "latBestMin": f"{min(stages[-1][1].values()):.2f}", "latBestMax": f"{max(stages[-1][1].values()):.2f}",
        "latScalarMin": f"{min(scalar.values()):.2f}", "latScalarMax": f"{max(scalar.values()):.2f}",
        "latVerifyMin": f"{min(verify.values()):.2f}", "latVerifyMax": f"{max(verify.values()):.2f}",
        "floorMac": f"{floor['projection_array']['full_answer_mac_only_floor_s']:.2f}",
        "floorDdr": f"{floor['projection_ddr_reads']['cached_read_only_floor_s_at_50mhz']:.2f}",
        "weightGB": f"{floor['projection_ddr_reads']['weight_bytes'] / 1e9:.2f}",
    })


def edit_figure(numbers):
    extended = load("paper/experiments/lm001-extended.json")
    sweep = load("paper/experiments/contrastive-sweep.json")
    contrastive = load("paper/experiments/contrastive-fit.json")["summary"]
    controls = extended["protocol"]["controls"]
    naming = set(controls[:11])

    def summarise(result):
        hits = sum(c["group"] == "target" and c["first_token_id"] == TARGET for c in result["cases"])
        leaks = sum(c["group"] == "control" and c["prompt"] in naming and not c["unchanged"] for c in result["cases"])
        other = sum(c["group"] == "control" and c["prompt"] not in naming and not c["unchanged"] for c in result["cases"])
        return hits, leaks, other

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.25, 1]})
    for author, colour, name in (("conventional", INK, "Conventional fit"), ("astra", PINK, "Astra proposal")):
        rows = [r for r in extended["results"] if r["author"] == author]
        alpha = np.array([r["alpha"] for r in rows])
        stats = np.array([summarise(r) for r in rows])
        left.plot(alpha, stats[:, 0] / 24, color=colour, lw=1.1, label=f"{name}: target prompts")
        left.plot(alpha, stats[:, 1] / 11, color=colour, lw=1.1, ls="--", label=f"{name}: naming controls")
        at_one = stats[alpha == 1.0][0]
        numbers.update({f"{author}Hits": str(at_one[0]), f"{author}Leaks": str(at_one[1]),
                        f"{author}Other": str(at_one[2]), f"{author}MaxOther": str(stats[:, 2].max())})
    left.axvline(1.0, color=LIGHT, lw=0.8, zorder=0)
    left.text(0.97, 1.0, "frozen row", color=GREY, fontsize=6.5, ha="right")
    left.set_xlabel(r"Edit strength $\alpha$ (row $= r_0 + \alpha\,(r_1 - r_0)$)")
    left.set_ylabel("Share selecting Tim / changed")
    left.set_ylim(-0.03, 1.05)
    left.set_xlim(0, 2)
    left.set_title("a  Strength sweep on 48 new prompts")
    left.legend(loc="lower right", handlelength=1.8, borderaxespad=0.1)

    for row in sweep:
        colour = BLUE if row["positives"] else GREY
        right.scatter(row["extended_naming_changed"] / 11, row["extended_target_hits"] / 24, s=14, color=colour,
                      zorder=3, edgecolor="white", lw=0.4)
    for row in sweep:
        if row["positives"] == 3 and row["negatives"] in (0, 1, 4):
            right.annotate(f"+{row['negatives']} neg", (row["extended_naming_changed"] / 11, row["extended_target_hits"] / 24),
                           xytext=(4, -9 if row["negatives"] == 4 else 3), textcoords="offset points", fontsize=6.2, color=BLUE)
    conventional = (int(numbers["conventionalLeaks"]) / 11, int(numbers["conventionalHits"]) / 24)
    astra = (int(numbers["astraLeaks"]) / 11, int(numbers["astraHits"]) / 24)
    right.scatter(*conventional, s=34, marker="s", color=INK, zorder=4, label="Conventional (LM001)")
    right.scatter(*astra, s=34, marker="D", color=PINK, zorder=4, label="Astra (LM001)")
    right.scatter([], [], s=14, color=GREY, label="Negatives only")
    right.scatter([], [], s=14, color=BLUE, label="3 positives + negatives")
    right.set_xlim(-0.05, 1.0)
    right.set_ylim(0.1, 1.05)
    right.set_xlabel("Naming controls changed (of 11)")
    right.set_ylabel("Target prompts selecting Tim (of 24)")
    right.set_title("b  Contrastive constraints")
    right.legend(loc="lower left", handlelength=1.0, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGURES / "edits.pdf")

    conv = next(r for r in extended["results"] if r["author"] == "conventional" and r["alpha"] == 1.0)
    astr = next(r for r in extended["results"] if r["author"] == "astra" and r["alpha"] == 1.0)
    targets = [c["prompt"] for c in conv["cases"] if c["group"] == "target"]
    hit = lambda result: {c["prompt"]: c["first_token_id"] == TARGET for c in result["cases"]}
    hc, ha = hit(conv), hit(astr)
    b = sum(hc[p] and not ha[p] for p in targets)
    c = sum(ha[p] and not hc[p] for p in targets)
    numbers.update({
        "extTargets": "24", "extControls": "24", "extNaming": "11", "extOther": "13",
        "convVsAstraB": str(b), "convVsAstraC": str(c), "convVsAstraP": f"{mcnemar_exact(b, c):.4f}",
        "contrastiveHits": str(contrastive["extended_target_hits"]),
        "contrastiveLeaks": str(contrastive["extended_naming_changed"]),
        "contrastiveOther": str(contrastive["extended_other_changed"]),
        "contrastiveLmHits": str(contrastive["lm001_target_hits"]),
        "contrastiveLmControls": str(contrastive["lm001_controls_unchanged"]),
        "contrastiveConstraints": str(contrastive["constraints"]),
        "contrastiveLone": str(contrastive["row_l1_change"]),
    })
    for key, (k, n) in {"convHitCI": (int(numbers["conventionalHits"]), 24), "astraHitCI": (int(numbers["astraHits"]), 24),
                        "contrHitCI": (contrastive["extended_target_hits"], 24),
                        "convLeakCI": (int(numbers["conventionalLeaks"]), 11), "astraLeakCI": (int(numbers["astraLeaks"]), 11),
                        "contrLeakCI": (contrastive["extended_naming_changed"], 11),
                        "lmConvCI": (5, 6), "lmAstraCI": (3, 6)}.items():
        lo, hi = wilson(k, n)
        numbers[key] = f"{100 * lo:.0f}--{100 * hi:.0f}\\%"


def lm001_numbers(numbers):
    software = load("build/lm001/lm001-software.json")["conditions"]
    pynq = load("build/lm001/lm001-pynq.json")
    same = all(software[name]["cases"][i]["sampled_token_ids"] == pynq["conditions"][name]["cases"][i]["sampled_token_ids"]
               for name in software for i in range(12))
    matched = sum(case.get("matched_scores_if_on_board", 0) for cond in pynq["conditions"].values() for case in cond["cases"])
    numbers.update({"lmBoardMatchesSoftware": "yes" if same else "no", "lmBoardScores": f"{matched:,}".replace(",", "{,}")})


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    numbers = {}
    latency_figure(numbers)
    edit_figure(numbers)
    lm001_numbers(numbers)
    lines = [f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in sorted(numbers.items())]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
