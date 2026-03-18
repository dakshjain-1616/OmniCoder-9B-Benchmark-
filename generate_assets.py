"""
Generate high-quality benchmark visualization PNG images from benchmark_results.json.
Saves charts to /root/benchmark17/assets/ directory.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "benchmark_results.json")
ASSETS_DIR   = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Colour palette (one per model, consistent across all charts) ───────────────
MODEL_COLORS = {
    "OmniCoder-9B":          "#6C63FF",   # purple  – primary model
    "Qwen3-8B":              "#00C9A7",   # teal
    "Llama3.1-8B":           "#FF6B6B",   # coral
    "DeepSeek-Coder-V2-16B": "#FFA94D",   # amber
    "StarCoder2-15B":        "#4DABF7",   # sky-blue
}

MODEL_ORDER = [
    "OmniCoder-9B",
    "Qwen3-8B",
    "Llama3.1-8B",
    "DeepSeek-Coder-V2-16B",
    "StarCoder2-15B",
]

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0D1117",
    "axes.facecolor":    "#161B22",
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   "#C9D1D9",
    "axes.titlecolor":   "#F0F6FC",
    "xtick.color":       "#8B949E",
    "ytick.color":       "#8B949E",
    "text.color":        "#C9D1D9",
    "grid.color":        "#21262D",
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "legend.facecolor":  "#161B22",
    "legend.edgecolor":  "#30363D",
    "legend.labelcolor": "#C9D1D9",
})


def load_data(path: str) -> pd.DataFrame:
    """Load benchmark JSON and return a tidy DataFrame."""
    with open(path) as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    # Ensure model order
    df["display_name"] = pd.Categorical(df["display_name"], categories=MODEL_ORDER, ordered=True)
    return df


def per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-model metrics."""
    agg = (
        df.groupby("display_name", observed=True)
        .agg(
            avg_correctness=("correctness_score", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            avg_tps=("tokens_per_second", "mean"),
            pass_rate=("pass_at_1", "mean"),
            syntax_rate=("syntax_validity", "mean"),
            total_tasks=("task_id", "count"),
        )
        .reset_index()
    )
    agg["pass_rate_pct"]   = agg["pass_rate"]   * 100
    agg["syntax_rate_pct"] = agg["syntax_rate"] * 100
    return agg


# ── Chart 1 – Accuracy (Correctness Score) Bar Chart ──────────────────────────
def chart_accuracy(summary: pd.DataFrame, out_path: str) -> None:
    """Grouped bar chart: average correctness score per model."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0D1117")

    models  = summary["display_name"].tolist()
    scores  = summary["avg_correctness"].tolist()
    colors  = [MODEL_COLORS[m] for m in models]
    x       = np.arange(len(models))
    bars    = ax.bar(x, scores, color=colors, width=0.55, zorder=3,
                     edgecolor="#0D1117", linewidth=1.2)

    # Value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{score:.1f}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
            color="#F0F6FC",
        )

    # Highlight primary model
    primary_idx = models.index("OmniCoder-9B") if "OmniCoder-9B" in models else 0
    bars[primary_idx].set_edgecolor("#F0F6FC")
    bars[primary_idx].set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Average Correctness Score (0–100)", fontsize=12)
    ax.set_title("Average Correctness Score by Model", fontsize=16, fontweight="bold",
                 color="#F0F6FC", pad=18)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # Subtitle annotation
    fig.text(0.5, 0.01,
             "Higher is better  •  Score = % of test cases passed per task",
             ha="center", fontsize=10, color="#8B949E")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ── Chart 2 – Tokens Per Second Bar Chart ─────────────────────────────────────
def chart_tokens_per_second(summary: pd.DataFrame, out_path: str) -> None:
    """Horizontal bar chart: average tokens/sec per model."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0D1117")

    models = summary["display_name"].tolist()
    tps    = summary["avg_tps"].tolist()
    colors = [MODEL_COLORS[m] for m in models]
    y      = np.arange(len(models))

    bars = ax.barh(y, tps, color=colors, height=0.55, zorder=3,
                   edgecolor="#0D1117", linewidth=1.2)

    # Value labels
    for bar, val in zip(bars, tps):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} tok/s",
            va="center", fontsize=12, fontweight="bold",
            color="#F0F6FC",
        )

    # Highlight primary
    primary_idx = models.index("OmniCoder-9B") if "OmniCoder-9B" in models else 0
    bars[primary_idx].set_edgecolor("#F0F6FC")
    bars[primary_idx].set_linewidth(2.5)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlim(0, max(tps) * 1.22)
    ax.set_xlabel("Average Tokens Per Second", fontsize=12)
    ax.set_title("⚡  Inference Speed: Tokens Per Second", fontsize=16, fontweight="bold",
                 color="#F0F6FC", pad=18)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.5, 0.01,
             "Higher is better  •  Measured via Ollama streaming API",
             ha="center", fontsize=10, color="#8B949E")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ── Chart 3 – Accuracy vs Latency Scatter Plot ────────────────────────────────
def chart_accuracy_vs_latency(summary: pd.DataFrame, out_path: str) -> None:
    """Bubble scatter: accuracy (y) vs avg latency (x), bubble size = TPS."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0D1117")

    for _, row in summary.iterrows():
        model  = row["display_name"]
        color  = MODEL_COLORS[model]
        size   = (row["avg_tps"] / summary["avg_tps"].max()) * 1800 + 200

        ax.scatter(
            row["avg_latency_ms"] / 1000,
            row["avg_correctness"],
            s=size, color=color, alpha=0.85,
            edgecolors="#F0F6FC", linewidths=1.5, zorder=4,
        )
        # Label offset
        offset_x = 0.15
        offset_y = 1.5
        ax.annotate(
            model,
            xy=(row["avg_latency_ms"] / 1000, row["avg_correctness"]),
            xytext=(row["avg_latency_ms"] / 1000 + offset_x,
                    row["avg_correctness"] + offset_y),
            fontsize=10, fontweight="bold", color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    ax.set_xlabel("Average Latency (seconds)", fontsize=12)
    ax.set_ylabel("Average Correctness Score (0–100)", fontsize=12)
    ax.set_title("🔬  Accuracy vs. Latency  (bubble size ∝ tokens/sec)",
                 fontsize=16, fontweight="bold", color="#F0F6FC", pad=18)
    ax.yaxis.grid(True, zorder=0)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # Quadrant annotations
    ax.axhline(y=50, color="#30363D", linestyle="--", linewidth=1, zorder=2)
    ax.axvline(x=summary["avg_latency_ms"].mean() / 1000,
               color="#30363D", linestyle="--", linewidth=1, zorder=2)

    fig.text(0.5, 0.01,
             "Top-left = best (high accuracy, low latency)  •  Bubble size = tokens/sec",
             ha="center", fontsize=10, color="#8B949E")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ── Chart 4 – Pass@1 Rate by Category Heatmap ─────────────────────────────────
def chart_pass_rate_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """Heatmap: pass@1 rate per model × task category."""
    pivot = (
        df.groupby(["display_name", "category"], observed=True)["pass_at_1"]
        .mean()
        .unstack(fill_value=0)
    )
    # Reorder rows
    pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")

    data   = pivot.values * 100
    im     = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    # Cell annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt_color = "black" if 30 < val < 80 else "white"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pass@1 Rate (%)", color="#C9D1D9", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="#C9D1D9")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#C9D1D9")

    ax.set_title("📊  Pass@1 Rate by Model & Category (%)",
                 fontsize=16, fontweight="bold", color="#F0F6FC", pad=18)

    fig.text(0.5, 0.01,
             "Green = 100% pass  •  Red = 0% pass  •  Per-category average across all tasks",
             ha="center", fontsize=10, color="#8B949E")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ── Chart 5 – Radar / Spider Chart ────────────────────────────────────────────
def chart_radar(summary: pd.DataFrame, out_path: str) -> None:
    """Radar chart comparing models across 4 normalised dimensions."""
    categories  = ["Correctness", "Speed\n(TPS)", "Pass Rate", "Syntax\nValidity"]
    N           = len(categories)
    angles      = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles     += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    max_tps = summary["avg_tps"].max()

    for _, row in summary.iterrows():
        model  = row["display_name"]
        color  = MODEL_COLORS[model]
        values = [
            row["avg_correctness"] / 100,
            row["avg_tps"] / max_tps,
            row["pass_rate_pct"] / 100,
            row["syntax_rate_pct"] / 100,
        ]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2.2, linestyle="solid", label=model)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, color="#C9D1D9")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=9, color="#8B949E")
    ax.grid(color="#30363D", linewidth=0.8)
    ax.spines["polar"].set_color("#30363D")

    ax.set_title("🕸️  Multi-Dimensional Model Comparison",
                 fontsize=16, fontweight="bold", color="#F0F6FC", pad=28)

    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        fontsize=10, framealpha=0.9,
    )

    fig.text(0.5, 0.01,
             "All axes normalised to [0, 1]  •  Larger area = better overall performance",
             ha="center", fontsize=10, color="#8B949E")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    """Load data and generate all benchmark visualisation PNGs."""
    print(f"\n📂  Loading results from: {RESULTS_FILE}")
    df      = load_data(RESULTS_FILE)
    summary = per_model_summary(df)

    print(f"\n📊  Model summary:\n{summary[['display_name','avg_correctness','avg_tps','pass_rate_pct']].to_string(index=False)}\n")
    print(f"🖼️   Generating charts → {ASSETS_DIR}\n")

    chart_accuracy(
        summary,
        os.path.join(ASSETS_DIR, "accuracy_scores.png"),
    )
    chart_tokens_per_second(
        summary,
        os.path.join(ASSETS_DIR, "tokens_per_second.png"),
    )
    chart_accuracy_vs_latency(
        summary,
        os.path.join(ASSETS_DIR, "accuracy_vs_latency.png"),
    )
    chart_pass_rate_heatmap(
        df,
        os.path.join(ASSETS_DIR, "pass_rate_heatmap.png"),
    )
    chart_radar(
        summary,
        os.path.join(ASSETS_DIR, "radar_comparison.png"),
    )

    print(f"\n✅  All charts saved to {ASSETS_DIR}")
    print("   Files:")
    for f in sorted(os.listdir(ASSETS_DIR)):
        size = os.path.getsize(os.path.join(ASSETS_DIR, f))
        print(f"   • {f}  ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
