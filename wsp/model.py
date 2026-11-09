"""Tiny 2 -> H -> 1 ReLU network, toy 2D tasks, and weight (de)serialization.

Weights are exchanged with the LLM as human-readable JSON whose keys match the
PyTorch state_dict (hidden.weight, hidden.bias, out.weight, out.bias).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- tasks

RULES = {
    # class 1 inside a diamond centred at the origin
    "diamond": lambda x, y: (x.abs() + y.abs()) < 0.8,
    # class 1 in the 1st and 3rd quadrant (an XOR of the two signs)
    "quadrants": lambda x, y: (x * y) > 0,
}


def make_dataset(rule: str, n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform points in [-1, 1]^2 labelled by one of RULES."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g) * 2 - 1
    y = RULES[rule](X[:, 0], X[:, 1]).float()
    return X, y


# --------------------------------------------------------------------------- model

class TinyMLP(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.hidden = nn.Linear(2, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits; probability of class 1 is sigmoid(logit)."""
        return self.out(torch.relu(self.hidden(x))).squeeze(-1)


def train(model: TinyMLP, X, y, steps: int = 3000, lr: float = 0.02, seed: int = 0,
          target_acc: float | None = None, X_eval=None, y_eval=None) -> int:
    """Full-batch Adam. If target_acc is given, stop as soon as accuracy on
    (X_eval, y_eval) reaches it and return the number of steps used."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
        if target_acc is not None and accuracy(model, X_eval, y_eval) >= target_acc:
            return step
    return steps


@torch.no_grad()
def accuracy(model: TinyMLP, X, y) -> float:
    return ((model(X) > 0).float() == y).float().mean().item()


# --------------------------------------------------------------------------- weights <-> JSON

def weights_to_dict(model: TinyMLP, decimals: int = 3) -> dict:
    return {
        "architecture": (
            f"input(2: x, y) -> hidden = relu(hidden.weight @ [x, y] + hidden.bias) "
            f"({model.hidden.out_features} neurons) -> logit = out.weight @ hidden + out.bias; "
            f"P(class 1) = sigmoid(logit)"
        ),
        "params": {k: _round(v.tolist(), decimals) for k, v in model.state_dict().items()},
    }


def save_weights(model: TinyMLP, path: str | Path, decimals: int = 3) -> None:
    Path(path).write_text(json.dumps(weights_to_dict(model, decimals), indent=2))


def load_weights(path: str | Path) -> TinyMLP:
    params = json.loads(Path(path).read_text())["params"]
    model = TinyMLP(hidden=len(params["hidden.bias"]))
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in params.items()})
    return model


def round_model(model: TinyMLP, decimals: int = 3) -> TinyMLP:
    """Round weights exactly as they appear in the JSON the LLM sees."""
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.round(p * 10**decimals) / 10**decimals)
    return model


def changed_weights(a: TinyMLP, b: TinyMLP, tol: float = 1e-6) -> tuple[int, int]:
    """(number of parameters that differ, total parameters)."""
    if a.hidden.out_features != b.hidden.out_features:
        n = sum(p.numel() for p in b.parameters())
        return n, n
    sa, sb = a.state_dict(), b.state_dict()
    diff = sum(int(((sa[k] - sb[k]).abs() > tol).sum()) for k in sa)
    return diff, sum(v.numel() for v in sa.values())


def _round(v, d):
    return [_round(x, d) for x in v] if isinstance(v, list) else round(v, d)


# --------------------------------------------------------------------------- plotting

@torch.no_grad()
def plot_decision(model: TinyMLP, rule: str, ax=None, title: str = "", res: int = 200):
    """Model's class-1 region (shaded) vs. the true boundary (dashed)."""
    ax = ax or plt.gca()
    t = torch.linspace(-1, 1, res)
    xx, yy = torch.meshgrid(t, t, indexing="xy")
    grid = torch.stack([xx.flatten(), yy.flatten()], 1)
    pred = (model(grid) > 0).float().reshape(res, res)
    truth = RULES[rule](xx, yy).float()
    ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5], colors=["#f2f2f2", "#7aa6d6"])
    ax.contour(xx, yy, truth, levels=[0.5], colors="k", linestyles="--", linewidths=1)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_title(title, fontsize=10)
    return ax
