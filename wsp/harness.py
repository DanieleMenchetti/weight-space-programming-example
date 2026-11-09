"""Run Claude Opus 5.5 through Claude Code (headless `claude -p`) as the weight editor.

Each experiment gets its own workspace folder under runs/<name>/ containing only
what the LLM is allowed to see (weights JSON, optional forward.py + budget).
Claude runs with that folder as its working directory, in `--restricted` mode
(file tools cannot leave the folder), with a minimal tool set and permission
mode `dontAsk`, so anything not explicitly allowed is denied. The only shell
command it may run is `python forward.py ...`.

Results are cached in runs/<name>/claude_result.json so the notebook can be
re-executed (or read on GitHub) without spending tokens again. Claude is called
again only if the prompt changed or rerun=True.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

MODEL = "claude-opus-5-5"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
FORWARD_TOOL = Path(__file__).resolve().parent / "forward_tool.py"


def run_experiment(
    name: str,
    prompt: str,
    files: dict[str, str],
    forward_budget: int | None = None,
    effort: str = "high",
    rerun: bool = False,
    timeout_s: int = 1800,
) -> dict:
    """Prepare runs/<name>/, run Claude Code on it, return the parsed result.

    files: mapping filename -> text content to place in the workspace.
    forward_budget: if set, the workspace gets forward.py and Claude may run
        `python forward.py ...` at most this many times.
    """
    ws = RUNS_DIR / name
    cached = ws / "claude_result.json"
    prompt_file = ws / "prompt.txt"
    same_prompt = prompt_file.exists() and prompt_file.read_text(encoding="utf-8") == prompt
    if cached.exists() and same_prompt and not rerun:
        print(f"[cache] using {cached.relative_to(RUNS_DIR.parent)} (rerun=True to call Claude again)")
        return json.loads(cached.read_text(encoding="utf-8"))

    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    for fname, content in files.items():
        (ws / fname).write_text(content, encoding="utf-8")
    (ws / "prompt.txt").write_text(prompt, encoding="utf-8")

    tools = ["Read", "Write"]
    allowed = ["Read", "Write"]
    if forward_budget:
        shutil.copy(FORWARD_TOOL, ws / "forward.py")
        (ws / "budget.json").write_text(json.dumps({"max_forward_calls": forward_budget}))
        tools.append("Bash")
        allowed += ["Bash(python forward.py *)", "Bash(python -I forward.py *)"]

    cmd = [
        "claude", "-p",
        "--model", MODEL,
        "--effort", effort,
        "--output-format", "json",
        "--restricted",            # file tools confined to the workspace; user settings ignored
        "--tools", ",".join(tools),
        "--permission-mode", "dontAsk",
        "--no-session-persistence",
        "--allowedTools", *allowed,
    ]
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
    print(f"[claude] running {MODEL} in {ws.relative_to(RUNS_DIR.parent)} ... (this can take a few minutes)")
    proc = subprocess.run(cmd, input=prompt, cwd=ws, capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout_s, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited with {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    result = json.loads(proc.stdout)
    log = ws / "forward_log.jsonl"
    result["forward_calls_used"] = len(log.read_text().splitlines()) if log.exists() else 0
    cached.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summary(result: dict) -> str:
    keys = ("num_turns", "duration_ms", "total_cost_usd", "forward_calls_used")
    return ", ".join(f"{k}={result.get(k)}" for k in keys)
