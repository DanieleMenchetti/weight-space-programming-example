# Weight-Space Programming: a tiny, CPU-only example

An LLM edits a neural network **by reading and writing its weights directly**, the way an engineer edits source code. There's no gradient descent and no training loop.

In this repository the editor is **Claude Opus 5.5**, run headless through **Claude Code**. The networks are tiny **PyTorch** MLPs that run on any CPU in seconds. Everything is in one notebook: [`weight_space_programming.ipynb`](weight_space_programming.ipynb).

> The term comes from MIT CSAIL's November 2026 blog post *"Sparks of Weight-Space Programming"*. There, a frontier LLM repaired RL policies, wrote game-playing networks from scratch and fixed a looping bug in a 0.8B-parameter LLM. This repo is a small-scale, fully inspectable reproduction of the idea, not of those results.

## The three experiments

| | Experiment | What Claude gets | What Claude must produce | Result |
|---|---|---|---|---|
| 1 | **READ** | the 33 weights of a trained 2→8→1 net, **without being told the task**, and no tools to run it | an explanation of what the net computes | Recovered the exact rule `\|x\| + \|y\| < 0.8` (a diamond), neuron by neuron |
| 2 | **REPAIR** | the same net with **one planted bug** (a sign flip), the spec, 8 failing inputs, and a 10-call forward-pass tool | fixed weights | Found the corrupted weight, changed **1/33 numbers**, accuracy 66.4% → **99.8%**, original weights restored exactly, using 2 forward calls |
| 3 | **WRITE** | an all-zero 2→4→1 net and the spec "class 1 if and only if `x·y > 0`" (XOR) | all 17 weights, from scratch | `logit = \|x+y\| − \|x−y\|`, **100%** accuracy with 0 training steps |

**Gradient baselines** (same architectures, Adam, 2000 labelled points):
- **REPAIR:** fine-tuning the broken net for 5000 steps reaches 97.8% and changes 29/33 weights.
- **WRITE:** training 2→4→1 from scratch for 3000 steps, across 5 seeds, gives 83.0%–99.3%.

Each run cost about $0.12–0.21 (`total_cost_usd` as reported by Claude Code) and took under a minute.

## How the harness works

```
notebook ──► wsp/harness.py ──► runs/<experiment>/   (the only folder Claude can see)
                                  ├─ prompt.txt
                                  ├─ weights.json        weights as JSON (PyTorch state_dict keys)
                                  ├─ forward.py          pure-Python forward pass with a call budget
                                  ├─ budget.json
                                  └─ ← Claude writes repaired.json / written.json / notes.md here
```

Claude is invoked as:

```bash
claude -p --model claude-opus-5-5 --effort high --output-format json \
       --restricted --tools Read,Write,Bash --permission-mode dontAsk \
       --allowedTools Read Write "Bash(python forward.py *)"
```

- **`--restricted`** confines the file tools to the experiment folder, so Claude can't peek at the notebook, where the planted bug is visible.
- **`--permission-mode dontAsk`** with a single Bash allow rule means `forward.py` is the only program Claude can run. It can't train, write scripts or run an optimizer. It must reason about the numbers.
- **`forward.py`** ([`wsp/forward_tool.py`](wsp/forward_tool.py)) is the LLM's "debugger". It prints hidden activations, the logit and the probability for the points Claude chooses, and refuses calls once the budget is used up. It is checked against the PyTorch model.
- Claude's run metadata and final message (`claude_result.json`), prompts, forward-call logs and outputs are committed under [`runs/`](runs/), so you can audit everything without re-running.

## Run it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook weight_space_programming.ipynb
```

By default the notebook uses the cached Claude results in `runs/`, so it needs no API usage. To make Claude do the work again, install [Claude Code](https://claude.com/claude-code), log in, and set `RERUN = True` in the first code cell. The cache is also bypassed automatically whenever you change a prompt. LLM outputs are not deterministic, so your runs may differ.

## Repository layout

```
weight_space_programming.ipynb   the walkthrough: train → READ → REPAIR → WRITE → results
wsp/model.py                     TinyMLP, toy datasets, training, weights <-> JSON, plots
wsp/forward_tool.py              the forward-pass tool given to Claude (copied as forward.py)
wsp/harness.py                   runs `claude -p` in a sandboxed per-experiment folder, caches results
runs/                            prompts, inputs and everything Claude produced
```

## Notes and limits

- This is a toy. With 33 parameters, each hidden ReLU is a readable half-plane. Real networks are far less legible, and the original work needed sandboxes and forward-pass budgets to scale.
- The planted bug is **localized**, which is the case where weight-space programming works best. The MIT authors report it struggles with distributed faults and can't add missing capabilities.
- During development, Claude once tried `sed ... && python forward.py ...` in a single command. The sandbox denied the whole command, and Claude finished the repair by hand calculation instead. The prompt now states that the shell accepts only standalone `forward.py` calls.

**Ideas to extend:** distributed bugs (noise on every weight), larger hidden layers, removing the forward tool, MNIST-scale MLPs, or comparing editor models.
