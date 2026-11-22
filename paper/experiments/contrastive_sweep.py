"""Exploratory: vary the number of naming negatives and positives in the contrastive fit."""

import json
import sys
from pathlib import Path

import paper.experiments.contrastive_fit as cf

NEGATIVES, POSITIVES = cf.NEGATIVE_AUTHORING, cf.POSITIVE_AUTHORING
out = Path(sys.argv[-1])
base_args = sys.argv[1:-1]
rows = []
for positives in (0, 3):
    for negatives in (0, 1, 2, 4, 6, 8):
        cf.NEGATIVE_AUTHORING, cf.POSITIVE_AUTHORING = NEGATIVES[:negatives], POSITIVES[:positives]
        target = out.with_name(f"contrastive-p{positives}-n{negatives}.json")
        sys.argv = ["contrastive_fit", *base_args, "--out", str(target)]
        cf.main()
        rows.append({"positives": positives, "negatives": negatives, **json.loads(target.read_text())["summary"]})
        target.unlink()
out.write_text(json.dumps(rows, indent=1) + "\n")
