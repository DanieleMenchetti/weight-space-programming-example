"""Forward-pass tool handed to the LLM (copied into each workspace as forward.py).

It is pure Python on purpose: no torch, no numpy, so the LLM's Bash tool can run it
with any Python interpreter, and every line of the "debugger" is readable.

Usage:  python forward.py <weights.json> "x1,y1 x2,y2 ..."
Prints, for each point, the hidden ReLU activations, the logit and P(class 1).
Each invocation consumes one call from the budget in budget.json.
"""

import json
import math
import sys
from pathlib import Path

MAX_POINTS_PER_CALL = 32
LOG = Path("forward_log.jsonl")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    max_calls = json.loads(Path("budget.json").read_text())["max_forward_calls"]
    used = len(LOG.read_text().splitlines()) if LOG.exists() else 0
    if used >= max_calls:
        print(f"Forward-pass budget exhausted ({used}/{max_calls} calls used).")
        return 1

    weights_file = sys.argv[1]
    raw = " ".join(sys.argv[2:]).replace(";", " ").split()
    points = [tuple(float(c) for c in p.split(",")) for p in raw][:MAX_POINTS_PER_CALL]

    p = json.loads(Path(weights_file).read_text())["params"]
    W1, b1, W2, b2 = p["hidden.weight"], p["hidden.bias"], p["out.weight"][0], p["out.bias"][0]

    print(f"call {used + 1}/{max_calls}  weights={weights_file}")
    print(f"{'x':>7} {'y':>7} | hidden activations | {'logit':>8} {'P(1)':>6}")
    for x, y in points:
        h = [max(0.0, w[0] * x + w[1] * y + b) for w, b in zip(W1, b1)]
        logit = sum(a * w for a, w in zip(h, W2)) + b2
        prob = 1 / (1 + math.exp(-max(-60.0, min(60.0, logit))))
        hs = " ".join(f"{a:.2f}" for a in h)
        print(f"{x:7.3f} {y:7.3f} | {hs} | {logit:8.3f} {prob:6.3f}")

    with LOG.open("a") as f:
        f.write(json.dumps({"weights": weights_file, "points": points}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
