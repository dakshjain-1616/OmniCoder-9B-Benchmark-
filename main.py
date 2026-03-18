"""
OmniCoder-9B Benchmarking Harness
Compares OmniCoder-9B against other local coding LLMs via Ollama.

Improvements:
- Robust multi-pattern regex code extraction from Markdown responses
- AST-based syntax validation before subprocess execution
- JSON caching layer to avoid redundant LLM inference
- Strict system prompt to force clean Python-only output
- Modernized Tailwind CSS HTML report with expandable error panels
"""

import argparse
import ast
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strict system prompt: forces models to emit ONLY executable Python code
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Your task is to write ONLY valid, executable Python code — nothing else. "
    "Do NOT include any explanation, commentary, markdown prose, or conversational text. "
    "Do NOT use triple-backtick fences. "
    "Output ONLY the raw Python source code that solves the problem. "
    "The code must be self-contained and importable. "
    "Define the requested function(s) exactly as specified."
)


# ---------------------------------------------------------------------------
# Config & task loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "./config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_tasks(tasks_dir: str) -> list[dict]:
    """Load all task JSON files from the tasks directory."""
    tasks = []
    tasks_path = Path(tasks_dir)
    for task_file in sorted(tasks_path.glob("task_*.json")):
        with open(task_file, "r") as f:
            tasks.append(json.load(f))
    logger.info(f"Loaded {len(tasks)} tasks from {tasks_dir}")
    return tasks


# ---------------------------------------------------------------------------
# JSON caching layer
# ---------------------------------------------------------------------------

def _cache_key(model: str, prompt: str) -> str:
    """Generate a deterministic cache key from model name and prompt."""
    raw = f"{model}::{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(cache_path: str) -> dict:
    """Load the LLM response cache from disk (returns empty dict if missing)."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            logger.debug(f"Cache loaded: {len(data)} entries from {cache_path}")
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Cache file unreadable ({exc}), starting fresh.")
    return {}


def save_cache(cache: dict, cache_path: str) -> None:
    """Persist the LLM response cache to disk."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# Code extraction — robust multi-pattern regex
# ---------------------------------------------------------------------------

def extract_code(response_text: str, entry_point: str) -> str:
    """
    Extract the best Python code block from a model response.

    Strategy (in order of preference):
    1. ```python ... ``` fenced block containing the entry_point
    2. Any ``` ... ``` fenced block containing the entry_point
    3. Longest fenced block overall
    4. Lines starting from a def/import/from statement
    5. Raw response as fallback
    """
    # Pattern priority: explicit python tag first, then generic fence
    fence_patterns = [
        r"```python\s*\n(.*?)```",
        r"```py\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
        r"```(.*?)```",
    ]

    def _clean_block(block: str) -> str:
        """Strip leading language identifier line (e.g. 'python') from a code block."""
        lines = block.split("\n")
        if lines and re.match(r"^\s*(python|py|python3)\s*$", lines[0], re.IGNORECASE):
            lines = lines[1:]
        return "\n".join(lines).strip()

    all_blocks: list[str] = []
    for pattern in fence_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        all_blocks.extend(_clean_block(m) for m in matches if m.strip())

    if all_blocks:
        # Prefer blocks that contain the entry_point function definition
        ep_blocks = [b for b in all_blocks if f"def {entry_point}" in b]
        if ep_blocks:
            return max(ep_blocks, key=len)
        # Fall back to the longest block
        return max(all_blocks, key=len)

    # No fenced blocks — extract from raw lines
    lines = response_text.split("\n")
    code_lines: list[str] = []
    capturing = False
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith(f"def {entry_point}")
            or stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("class ")
        ):
            capturing = True
        if capturing:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return response_text.strip()


# ---------------------------------------------------------------------------
# AST-based syntax validation
# ---------------------------------------------------------------------------

def check_syntax(code: str) -> tuple[bool, str]:
    """
    Validate Python syntax using the AST parser.

    Returns:
        (is_valid, error_message) — error_message is empty string on success.
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Subprocess execution sandbox
# ---------------------------------------------------------------------------

def execute_code_with_test(
    code: str,
    test_input: str,
    expected_output: str,
    entry_point: str,
    timeout: int = 10,
) -> tuple[bool, str, str]:
    """
    Execute generated code against a single test case in a subprocess sandbox.

    Returns:
        (passed, actual_output, stderr_output)
    """
    test_harness = f"""\
{code}

import sys

try:
    result = {entry_point}({test_input})
    print(repr(result))
except Exception as exc:
    print(f"ERROR: {{exc}}", file=sys.stderr)
    sys.exit(1)
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_harness)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stderr_out = proc.stderr.strip()
        if proc.returncode != 0:
            return False, "", f"Runtime error: {stderr_out}"

        actual = proc.stdout.strip()
        # Normalised comparison via literal_eval
        try:
            actual_val = ast.literal_eval(actual)
            expected_val = ast.literal_eval(expected_output)
            passed = actual_val == expected_val
        except Exception:
            passed = actual.strip() == expected_output.strip()

        return passed, actual, stderr_out
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out"
    except Exception as exc:
        return False, "", str(exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Ollama streaming query (with system prompt injection)
# ---------------------------------------------------------------------------

def query_ollama_streaming(
    base_url: str,
    model: str,
    prompt: str,
    timeout: int = 120,
) -> dict:
    """
    Query Ollama with streaming to capture TTFT and tokens/sec metrics.
    Injects a strict system prompt to improve code-only output quality.
    """
    url = f"{base_url}/api/generate"
    # Append /no_think so Qwen3-based models (e.g. OmniCoder) emit code directly
    # instead of spending their entire token budget on internal chain-of-thought.
    # Non-Qwen3 models treat this as plain text and ignore it harmlessly.
    payload = {
        "model": model,
        "prompt": prompt + " /no_think",
        "system": SYSTEM_PROMPT,
        "stream": True,
        "options": {"temperature": 0.05, "num_predict": 8192},
    }

    start_time = time.time()
    first_token_time = None
    full_response = ""
    token_count = 0

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = chunk.get("response", "")
                if token and first_token_time is None:
                    first_token_time = time.time()

                full_response += token
                token_count += 1

                if chunk.get("done", False):
                    if "eval_count" in chunk:
                        token_count = chunk["eval_count"]
                    break

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout querying model {model}")
        return {"error": "timeout", "response": full_response}
    except Exception as exc:
        logger.error(f"Error querying {model}: {exc}")
        return {"error": str(exc), "response": ""}

    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else total_time_ms
    elapsed_sec = end_time - start_time
    tps = token_count / elapsed_sec if elapsed_sec > 0 else 0

    return {
        "response": full_response,
        "ttft_ms": round(ttft_ms, 2),
        "total_time_ms": round(total_time_ms, 2),
        "latency_ms": round(total_time_ms, 2),
        "tokens_per_second": round(tps, 2),
        "token_count": token_count,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    model_name: str,
    task: dict,
    config: dict,
    cache: dict,
) -> dict:
    """
    Run a single task against a model and return full metrics.
    Uses the JSON cache to skip redundant LLM calls.
    """
    base_url = config["ollama"]["base_url"]
    exec_timeout = config["benchmark"]["execution_timeout"]
    ollama_timeout = config["ollama"]["timeout"]
    cache_path = config["benchmark"].get("cache_file", "./results/llm_cache.json")

    logger.info(f"  Task: {task['name']} | Model: {model_name}")

    prompt = task["prompt"]
    key = _cache_key(model_name, prompt)

    # --- Cache lookup ---
    if key in cache:
        logger.info("    [CACHE HIT] Using cached LLM response.")
        result = cache[key]
    else:
        result = query_ollama_streaming(base_url, model_name, prompt, ollama_timeout)
        if not (result.get("error") and not result.get("response")):
            cache[key] = result
            save_cache(cache, cache_path)

    # --- Handle hard failure ---
    if result.get("error") and not result.get("response"):
        return {
            "model": model_name,
            "task_id": task["id"],
            "task_name": task["name"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "correctness_score": 0,
            "latency_ms": 0,
            "tokens_per_second": 0,
            "response_length": 0,
            "syntax_validity": False,
            "syntax_error": "",
            "pass_at_1": False,
            "passed_tests": 0,
            "total_tests": len(task.get("test_cases", [])),
            "error": result.get("error"),
            "extracted_code": "",
            "raw_response": "",
            "test_details": [],
        }

    response_text = result["response"]
    entry_point = task.get("entry_point", "solution")

    # --- Extract code ---
    code = extract_code(response_text, entry_point)

    # --- AST syntax check ---
    syntax_valid, syntax_error = check_syntax(code)

    # --- Execute test cases ---
    test_cases = task.get("test_cases", [])
    passed_tests = 0
    total_tests = len(test_cases)
    test_details: list[dict] = []

    if syntax_valid and task.get("evaluation_type") == "execution" and total_tests > 0:
        for tc in test_cases:
            passed, actual, stderr = execute_code_with_test(
                code,
                tc["input"],
                tc["expected_output"],
                entry_point,
                exec_timeout,
            )
            if passed:
                passed_tests += 1
            test_details.append(
                {
                    "input": tc["input"],
                    "expected": tc["expected_output"],
                    "actual": actual,
                    "passed": passed,
                    "stderr": stderr,
                }
            )
    elif not syntax_valid:
        # Record syntax failure for every test case
        for tc in test_cases:
            test_details.append(
                {
                    "input": tc["input"],
                    "expected": tc["expected_output"],
                    "actual": "",
                    "passed": False,
                    "stderr": syntax_error,
                }
            )

    correctness_score = (
        int(passed_tests / total_tests * 100) if total_tests > 0 else (50 if syntax_valid else 0)
    )
    pass_at_1 = passed_tests == total_tests and total_tests > 0

    return {
        "model": model_name,
        "task_id": task["id"],
        "task_name": task["name"],
        "category": task["category"],
        "difficulty": task["difficulty"],
        "correctness_score": correctness_score,
        "latency_ms": round(result.get("latency_ms", 0), 2),
        "tokens_per_second": round(result.get("tokens_per_second", 0), 2),
        "response_length": len(response_text),
        "syntax_validity": syntax_valid,
        "syntax_error": syntax_error,
        "pass_at_1": pass_at_1,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "error": result.get("error"),
        "extracted_code": code,
        "raw_response": response_text,
        "test_details": test_details,
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(config: dict, tasks: list[dict]) -> list[dict]:
    """Run the full benchmark across all models and tasks, using the cache."""
    cache_path = config["benchmark"].get("cache_file", "./results/llm_cache.json")
    cache = load_cache(cache_path)

    results = []
    models = config["models"]
    total = len(models) * len(tasks)
    done = 0

    for model_cfg in models:
        model_name = model_cfg["name"]
        display_name = model_cfg.get("display_name", model_name)
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking model: {display_name} ({model_name})")
        logger.info(f"{'='*60}")

        for task in tasks:
            done += 1
            logger.info(f"[{done}/{total}] Running task: {task['name']}")
            result = evaluate_task(model_name, task, config, cache)
            result["display_name"] = display_name
            results.append(result)
            logger.info(
                f"  -> Score: {result['correctness_score']}/100 | "
                f"Latency: {result['latency_ms']:.0f}ms | "
                f"TPS: {result['tokens_per_second']:.1f} | "
                f"Pass@1: {result['pass_at_1']} | "
                f"Syntax: {'✓' if result['syntax_validity'] else '✗'}"
            )

    return results


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(results: list[dict], results_dir: str) -> str:
    """Save raw results to JSON (strips large raw_response to keep file lean)."""
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "benchmark_results.json")
    # Save a slimmed copy (no raw_response) for the results file
    slim = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "raw_response"}
        slim.append(row)
    with open(output_path, "w") as f:
        json.dump(slim, f, indent=2)
    logger.info(f"Raw results saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# HTML report — Tailwind CSS, modern minimalist design
# ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    """HTML-escape a string for safe embedding."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def generate_report(results: list[dict], config: dict) -> str:
    """
    Generate a modernised Tailwind CSS HTML report with:
    - Executive summary cards
    - 5 Plotly charts (scatter, bars, heatmap)
    - Summary table
    - Per-task detail table with expandable error/diff panels
    """
    import pandas as pd
    import plotly.graph_objects as go

    df = pd.DataFrame(results)
    output_file = config["report"]["output_file"]
    title = config["report"]["title"]
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # ---- Aggregate per model ----
    summary = (
        df.groupby(["display_name", "model"])
        .agg(
            avg_correctness=("correctness_score", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            avg_tps=("tokens_per_second", "mean"),
            avg_response_length=("response_length", "mean"),
            syntax_validity_rate=("syntax_validity", "mean"),
            pass_at_1_rate=("pass_at_1", "mean"),
            total_tasks=("task_id", "count"),
        )
        .reset_index()
    )
    summary["avg_correctness"] = summary["avg_correctness"].round(1)
    summary["avg_latency_ms"] = summary["avg_latency_ms"].round(1)
    summary["avg_tps"] = summary["avg_tps"].round(1)
    summary["syntax_validity_pct"] = (summary["syntax_validity_rate"] * 100).round(1)
    summary["pass_at_1_pct"] = (summary["pass_at_1_rate"] * 100).round(1)
    summary["is_primary"] = summary["model"].apply(
        lambda m: next((c["is_primary"] for c in config["models"] if c["name"] == m), False)
    )
    summary = summary.sort_values("is_primary", ascending=False).reset_index(drop=True)

    PRIMARY_COLOR = "#f97316"   # Tailwind orange-500
    SECONDARY_COLOR = "#3b82f6" # Tailwind blue-500
    COLORS = [PRIMARY_COLOR if r["is_primary"] else SECONDARY_COLOR for _, r in summary.iterrows()]

    # ---- Chart 1: Accuracy vs Speed scatter ----
    fig1 = go.Figure()
    for _, row in summary.iterrows():
        c = PRIMARY_COLOR if row["is_primary"] else SECONDARY_COLOR
        sz = 22 if row["is_primary"] else 14
        fig1.add_trace(go.Scatter(
            x=[row["avg_latency_ms"]],
            y=[row["avg_correctness"]],
            mode="markers+text",
            name=row["display_name"],
            text=[row["display_name"]],
            textposition="top center",
            marker=dict(size=sz, color=c, line=dict(width=2, color="white")),
        ))
    fig1.update_layout(
        title=dict(text="Accuracy vs Speed", font=dict(size=16)),
        xaxis_title="Average Latency (ms)",
        yaxis_title="Avg Correctness Score",
        yaxis=dict(range=[0, 110]),
        template="plotly_white",
        height=460,
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )

    # ---- Chart 2: Tokens/sec bar ----
    fig2 = go.Figure(go.Bar(
        x=summary["display_name"],
        y=summary["avg_tps"],
        marker_color=COLORS,
        text=summary["avg_tps"].apply(lambda x: f"{x:.1f}"),
        textposition="outside",
    ))
    fig2.update_layout(
        title=dict(text="Tokens Per Second", font=dict(size=16)),
        xaxis_title="Model",
        yaxis_title="Tokens / Second",
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=30, t=60, b=50),
    )

    # ---- Chart 3: Correctness bar ----
    fig3 = go.Figure(go.Bar(
        x=summary["display_name"],
        y=summary["avg_correctness"],
        marker_color=COLORS,
        text=summary["avg_correctness"].apply(lambda x: f"{x:.1f}"),
        textposition="outside",
    ))
    fig3.update_layout(
        title=dict(text="Average Correctness Score", font=dict(size=16)),
        xaxis_title="Model",
        yaxis_title="Score (0–100)",
        yaxis=dict(range=[0, 115]),
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=30, t=60, b=50),
    )

    # ---- Chart 4: Pass@1 bar ----
    fig4 = go.Figure(go.Bar(
        x=summary["display_name"],
        y=summary["pass_at_1_pct"],
        marker_color=COLORS,
        text=summary["pass_at_1_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    fig4.update_layout(
        title=dict(text="Pass@1 Rate (%)", font=dict(size=16)),
        xaxis_title="Model",
        yaxis_title="Pass@1 (%)",
        yaxis=dict(range=[0, 120]),
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=30, t=60, b=50),
    )

    # ---- Chart 5: Category heatmap ----
    cat_pivot = (
        df.groupby(["display_name", "category"])["correctness_score"]
        .mean()
        .unstack(fill_value=0)
    )
    fig5 = go.Figure(go.Heatmap(
        z=cat_pivot.values,
        x=cat_pivot.columns.tolist(),
        y=cat_pivot.index.tolist(),
        colorscale="RdYlGn",
        text=[[f"{v:.0f}" for v in row] for row in cat_pivot.values],
        texttemplate="%{text}",
        showscale=True,
        zmin=0,
        zmax=100,
    ))
    fig5.update_layout(
        title=dict(text="Correctness by Model × Category", font=dict(size=16)),
        xaxis_title="Category",
        yaxis_title="Model",
        template="plotly_white",
        height=380,
        margin=dict(l=160, r=30, t=60, b=80),
    )

    # ---- Summary table rows ----
    summary_rows = ""
    for _, row in summary.iterrows():
        ring = "ring-2 ring-orange-400" if row["is_primary"] else ""
        bold = "font-semibold" if row["is_primary"] else ""
        badge = '<span class="ml-2 text-xs bg-orange-100 text-orange-700 px-2 py-0.5 rounded-full">primary</span>' if row["is_primary"] else ""
        summary_rows += f"""
        <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors {ring}">
          <td class="px-4 py-3 {bold}">{_escape(row['display_name'])}{badge}</td>
          <td class="px-4 py-3 text-center font-mono">{row['avg_correctness']}</td>
          <td class="px-4 py-3 text-center font-mono">{row['avg_latency_ms']:.0f}</td>
          <td class="px-4 py-3 text-center font-mono">{row['avg_tps']}</td>
          <td class="px-4 py-3 text-center font-mono">{row['syntax_validity_pct']}%</td>
          <td class="px-4 py-3 text-center font-mono">{row['pass_at_1_pct']}%</td>
          <td class="px-4 py-3 text-center font-mono">{int(row['total_tasks'])}</td>
        </tr>"""

    # ---- Per-task detail rows with expandable error panels ----
    task_rows = ""
    for idx, row in df.sort_values(["display_name", "task_id"]).iterrows():
        sv_badge = (
            '<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">✓ Valid</span>'
            if row["syntax_validity"]
            else '<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800">✗ Invalid</span>'
        )
        pa_badge = (
            '<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">✓ Pass</span>'
            if row["pass_at_1"]
            else '<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800">✗ Fail</span>'
        )
        score_color = (
            "text-green-700" if row["correctness_score"] >= 80
            else "text-yellow-700" if row["correctness_score"] >= 40
            else "text-red-700"
        )

        # Build expandable detail panel
        detail_id = f"detail_{idx}"
        has_failures = not row["pass_at_1"] or not row["syntax_validity"]
        toggle_btn = ""
        detail_panel = ""

        if has_failures:
            toggle_btn = f'<button onclick="toggleDetail(\'{detail_id}\')" class="ml-2 text-xs text-blue-600 hover:text-blue-800 underline">details</button>'

            # Extracted code block
            code_html = f'<pre class="bg-slate-900 text-green-300 rounded-lg p-3 text-xs overflow-x-auto whitespace-pre-wrap mt-2">{_escape(row.get("extracted_code", ""))}</pre>'

            # Syntax error
            syntax_html = ""
            if not row["syntax_validity"] and row.get("syntax_error"):
                syntax_html = f'<div class="mt-2"><span class="text-xs font-semibold text-red-600">Syntax Error:</span><pre class="bg-red-50 text-red-800 rounded p-2 text-xs mt-1">{_escape(row["syntax_error"])}</pre></div>'

            # Test case diffs
            test_html = ""
            test_details = row.get("test_details", [])
            if test_details:
                test_html = '<div class="mt-3"><span class="text-xs font-semibold text-slate-600">Test Case Results:</span><div class="mt-1 space-y-2">'
                for ti, td in enumerate(test_details):
                    tc_color = "border-green-300 bg-green-50" if td["passed"] else "border-red-300 bg-red-50"
                    tc_icon = "✓" if td["passed"] else "✗"
                    tc_icon_color = "text-green-600" if td["passed"] else "text-red-600"
                    stderr_html = ""
                    if td.get("stderr"):
                        stderr_html = f'<div class="mt-1"><span class="text-xs text-slate-500">stderr:</span><pre class="text-xs text-red-700 bg-red-50 rounded p-1 mt-0.5">{_escape(td["stderr"])}</pre></div>'
                    test_html += f"""
                    <div class="border rounded p-2 text-xs {tc_color}">
                      <div class="flex items-center gap-2">
                        <span class="font-bold {tc_icon_color}">{tc_icon} Test {ti+1}</span>
                        <span class="text-slate-500">Input: <code class="bg-white px-1 rounded">{_escape(td['input'])}</code></span>
                      </div>
                      <div class="mt-1 grid grid-cols-2 gap-2">
                        <div><span class="text-slate-500">Expected:</span><pre class="bg-white rounded p-1 mt-0.5">{_escape(td['expected'])}</pre></div>
                        <div><span class="text-slate-500">Actual:</span><pre class="bg-white rounded p-1 mt-0.5">{_escape(td['actual']) if td['actual'] else '<em class="text-slate-400">no output</em>'}</pre></div>
                      </div>
                      {stderr_html}
                    </div>"""
                test_html += "</div></div>"

            detail_panel = f"""
            <tr id="{detail_id}" class="hidden bg-slate-50">
              <td colspan="9" class="px-6 py-4">
                <div class="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                  <h4 class="text-sm font-semibold text-slate-700 mb-2">📋 Extracted Code</h4>
                  {code_html}
                  {syntax_html}
                  {test_html}
                </div>
              </td>
            </tr>"""

        task_rows += f"""
        <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
          <td class="px-4 py-3 text-sm font-medium text-slate-700">{_escape(row['display_name'])}</td>
          <td class="px-4 py-3 text-sm">{_escape(row['task_name'])}{toggle_btn}</td>
          <td class="px-4 py-3 text-xs text-slate-500">{_escape(row['category'])}</td>
          <td class="px-4 py-3 text-xs"><span class="px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">{_escape(row['difficulty'])}</span></td>
          <td class="px-4 py-3 text-center font-mono font-semibold {score_color}">{row['correctness_score']}</td>
          <td class="px-4 py-3 text-center font-mono text-sm">{row['latency_ms']:.0f}</td>
          <td class="px-4 py-3 text-center font-mono text-sm">{row['tokens_per_second']:.1f}</td>
          <td class="px-4 py-3 text-center">{sv_badge}</td>
          <td class="px-4 py-3 text-center">{pa_badge}</td>
        </tr>
        {detail_panel}"""

    # ---- Executive summary KPI cards ----
    primary_row = summary[summary["is_primary"]]
    omni_score = f"{primary_row['avg_correctness'].values[0]:.1f}" if len(primary_row) > 0 else "N/A"
    omni_pass = f"{primary_row['pass_at_1_pct'].values[0]:.1f}%" if len(primary_row) > 0 else "N/A"
    total_evals = len(df)
    syntax_overall = f"{df['syntax_validity'].mean()*100:.1f}%"

    gen_time = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    # ---- Assemble HTML ----
    html = f"""<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{_escape(title)}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    body {{ font-family: 'Inter', sans-serif; }}
    code, pre, .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
    .chart-card {{ transition: box-shadow 0.2s ease; }}
    .chart-card:hover {{ box-shadow: 0 8px 30px rgba(0,0,0,0.08); }}
  </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen">

<!-- ── Top nav bar ── -->
<header class="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
  <div class="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <span class="text-2xl">🤖</span>
      <span class="font-semibold text-slate-800 text-lg">LLM Coding Benchmark</span>
    </div>
    <span class="text-xs text-slate-400">{gen_time}</span>
  </div>
</header>

<main class="max-w-7xl mx-auto px-6 py-10 space-y-12">

  <!-- ── Hero ── -->
  <section>
    <h1 class="text-3xl font-bold text-slate-900 tracking-tight">{_escape(title)}</h1>
    <p class="mt-2 text-slate-500 text-sm">
      {len(summary)} models &nbsp;·&nbsp; {len(df['task_id'].unique())} tasks &nbsp;·&nbsp; {total_evals} total evaluations
    </p>
  </section>

  <!-- ── KPI cards ── -->
  <section>
    <h2 class="text-xl font-semibold text-slate-700 mb-4">Executive Summary</h2>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div class="bg-white rounded-2xl shadow-sm border border-slate-100 p-5">
        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Models Evaluated</p>
        <p class="mt-2 text-4xl font-bold text-slate-800">{len(summary)}</p>
      </div>
      <div class="bg-white rounded-2xl shadow-sm border border-slate-100 p-5">
        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">OmniCoder Avg Score</p>
        <p class="mt-2 text-4xl font-bold text-orange-500">{omni_score}</p>
      </div>
      <div class="bg-white rounded-2xl shadow-sm border border-slate-100 p-5">
        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">OmniCoder Pass@1</p>
        <p class="mt-2 text-4xl font-bold text-orange-500">{omni_pass}</p>
      </div>
      <div class="bg-white rounded-2xl shadow-sm border border-slate-100 p-5">
        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Overall Syntax Valid</p>
        <p class="mt-2 text-4xl font-bold text-blue-500">{syntax_overall}</p>
      </div>
    </div>
  </section>

  <!-- ── Summary table ── -->
  <section>
    <h2 class="text-xl font-semibold text-slate-700 mb-4">Model Performance Summary</h2>
    <div class="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
      <table class="w-full text-sm">
        <thead class="bg-slate-800 text-white">
          <tr>
            <th class="px-4 py-3 text-left font-medium">Model</th>
            <th class="px-4 py-3 text-center font-medium">Avg Score</th>
            <th class="px-4 py-3 text-center font-medium">Avg Latency (ms)</th>
            <th class="px-4 py-3 text-center font-medium">Avg TPS</th>
            <th class="px-4 py-3 text-center font-medium">Syntax Valid</th>
            <th class="px-4 py-3 text-center font-medium">Pass@1</th>
            <th class="px-4 py-3 text-center font-medium">Tasks</th>
          </tr>
        </thead>
        <tbody>
          {summary_rows}
        </tbody>
      </table>
    </div>
  </section>

  <!-- ── Charts ── -->
  <section>
    <h2 class="text-xl font-semibold text-slate-700 mb-4">Performance Charts</h2>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="chart-card bg-white rounded-2xl shadow-sm border border-slate-100 p-4 lg:col-span-2" id="chart1"></div>
      <div class="chart-card bg-white rounded-2xl shadow-sm border border-slate-100 p-4" id="chart2"></div>
      <div class="chart-card bg-white rounded-2xl shadow-sm border border-slate-100 p-4" id="chart3"></div>
      <div class="chart-card bg-white rounded-2xl shadow-sm border border-slate-100 p-4" id="chart4"></div>
      <div class="chart-card bg-white rounded-2xl shadow-sm border border-slate-100 p-4" id="chart5"></div>
    </div>
  </section>

  <!-- ── Per-task detail table ── -->
  <section>
    <h2 class="text-xl font-semibold text-slate-700 mb-1">Per-Task Results</h2>
    <p class="text-xs text-slate-400 mb-4">Click <span class="text-blue-600 underline">details</span> on any failed row to inspect extracted code, expected vs actual output, and stderr.</p>
    <div class="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-x-auto">
      <table class="w-full text-sm">
        <thead class="bg-slate-800 text-white">
          <tr>
            <th class="px-4 py-3 text-left font-medium">Model</th>
            <th class="px-4 py-3 text-left font-medium">Task</th>
            <th class="px-4 py-3 text-left font-medium">Category</th>
            <th class="px-4 py-3 text-left font-medium">Difficulty</th>
            <th class="px-4 py-3 text-center font-medium">Score</th>
            <th class="px-4 py-3 text-center font-medium">Latency (ms)</th>
            <th class="px-4 py-3 text-center font-medium">TPS</th>
            <th class="px-4 py-3 text-center font-medium">Syntax</th>
            <th class="px-4 py-3 text-center font-medium">Pass@1</th>
          </tr>
        </thead>
        <tbody>
          {task_rows}
        </tbody>
      </table>
    </div>
  </section>

</main>

<footer class="mt-16 border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-400">
  Generated by OmniCoder Benchmark Harness &nbsp;·&nbsp; {gen_time}
</footer>

<script>
  // ── Render Plotly charts ──
  var fig1 = {fig1.to_json()};
  var fig2 = {fig2.to_json()};
  var fig3 = {fig3.to_json()};
  var fig4 = {fig4.to_json()};
  var fig5 = {fig5.to_json()};

  var cfg = {{responsive: true, displayModeBar: false}};
  Plotly.newPlot('chart1', fig1.data, fig1.layout, cfg);
  Plotly.newPlot('chart2', fig2.data, fig2.layout, cfg);
  Plotly.newPlot('chart3', fig3.data, fig3.layout, cfg);
  Plotly.newPlot('chart4', fig4.data, fig4.layout, cfg);
  Plotly.newPlot('chart5', fig5.data, fig5.layout, cfg);

  // ── Toggle expandable detail rows ──
  function toggleDetail(id) {{
    var el = document.getElementById(id);
    if (el) {{
      el.classList.toggle('hidden');
    }}
  }}
</script>
</body>
</html>"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML report saved to {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry point for the benchmarking CLI."""
    parser = argparse.ArgumentParser(
        description="OmniCoder-9B vs Competitors: Coding LLM Benchmark Harness"
    )
    parser.add_argument("--config", default="./config.yaml", help="Path to config.yaml")
    parser.add_argument("--models", nargs="+", help="Override model list (space-separated)")
    parser.add_argument("--tasks", nargs="+", help="Override task IDs to run (e.g. task_01 task_02)")
    parser.add_argument("--output", default=None, help="Override output report path")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation")
    parser.add_argument("--report-only", action="store_true",
                        help="Re-generate report from existing results JSON without re-running inference")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache and re-run all LLM calls")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.models:
        config["models"] = [{"name": m, "display_name": m, "is_primary": False} for m in args.models]
    if args.output:
        config["report"]["output_file"] = args.output
    if args.no_cache:
        cache_path = config["benchmark"].get("cache_file", "./results/llm_cache.json")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info("Cache cleared (--no-cache flag).")

    results_dir = config["benchmark"]["results_dir"]
    results_json = os.path.join(results_dir, "benchmark_results.json")

    # ---- Report-only mode: load existing results and regenerate ----
    if args.report_only:
        if not os.path.exists(results_json):
            logger.error(f"No results file found at {results_json}. Run benchmark first.")
            sys.exit(1)
        with open(results_json, "r") as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} results from {results_json}")
        report_path = generate_report(results, config)
        logger.info(f"✅ Report regenerated: {report_path}")
        return

    # ---- Full benchmark run ----
    all_tasks = load_tasks(config["benchmark"]["tasks_dir"])
    if args.tasks:
        all_tasks = [t for t in all_tasks if t["id"] in args.tasks]

    logger.info(f"Starting benchmark: {len(config['models'])} models × {len(all_tasks)} tasks")

    results = run_benchmark(config, all_tasks)
    save_results(results, results_dir)

    if not args.skip_report:
        report_path = generate_report(results, config)
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Benchmark complete! Report: {report_path}")
        logger.info(f"{'='*60}")
    else:
        logger.info("Report generation skipped.")

    # ---- Console summary ----
    from collections import defaultdict
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    model_scores: dict[str, list] = defaultdict(list)
    for r in results:
        model_scores[r.get("display_name", r["model"])].append(r["correctness_score"])
    for model, scores in sorted(model_scores.items()):
        avg = sum(scores) / len(scores)
        print(f"  {model:30s}: avg_score={avg:.1f}/100, tasks={len(scores)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
