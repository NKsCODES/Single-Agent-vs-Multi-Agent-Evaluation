"""Write benchmark results: raw JSON lines, a summary JSON, a Markdown report and charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from archeval.evaluation.economics import Summary, break_even, pick, summarize, verdict
from archeval.evaluation.runner import BenchmarkRun
from archeval.evaluation.sensitivity import SweepPoint

KINDS = ["lookup", "multihop", "compare", "aggregate"]
BASELINE = "single-large"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"


def _money(x: float | None) -> str:
    return "n/a" if x is None else f"${x * 1000:.2f}"


def decision_table(sums: list[Summary]) -> list[dict[str, Any]]:
    rows = []
    for kind in ["all", *KINDS]:
        base = pick(sums, BASELINE, kind)
        for s in (x for x in sums if x.kind == kind):
            be = None if s.architecture == BASELINE else break_even(base, s)
            rows.append({"kind": kind, "architecture": s.architecture,
                         "success_rate": s.success_rate,
                         "cost_per_task": s.cost_per_task,
                         "cost_vs_baseline": round(s.cost_per_task / base.cost_per_task, 3),
                         "success_delta_pp": round((s.success_rate - base.success_rate) * 100, 1),
                         "break_even_value_usd": None if be is None else round(be, 4),
                         "verdict": verdict(base, s)})
    return rows


def markdown(run: BenchmarkRun, sums: list[Summary], sweep: list[SweepPoint]) -> str:
    lines = ["# Benchmark report: single agent vs orchestrator-worker", "",
             f"Provider: `{run.settings['provider']}`. Runs: {len(run.records)} "
             f"({run.settings['repeats']} seed(s) x 60 tasks x "
             f"{len({r.result.architecture for r in run.records})} architectures). "
             f"Cost unit: USD per 1,000 tasks shown as $ per task x 1000 (\"m$\").", "",
             "## Overall", "",
             "| Architecture | Success (95% CI) | Grounded | Fact recall | Cost/task (m$) | "
             "Cost/success (m$) | Tokens/task | Calls/task | Latency p50 (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in (x for x in sums if x.kind == "all"):
        lines.append(f"| {s.architecture} | {s.success_rate:.1%} ({s.ci_low:.0%} to "
                     f"{s.ci_high:.0%}) | {s.grounded_rate:.1%} | {s.fact_recall:.1%} | "
                     f"{s.cost_per_task * 1000:.2f} | {_money(s.cost_per_success)[1:]} | "
                     f"{s.tokens_per_task:,.0f} | {s.calls_per_task:.1f} | "
                     f"{s.latency_p50_s:.1f} |")
    lines += ["", "## By task type", "",
              "| Task type | Architecture | Success | Cost/task (m$) | Cost vs single-large | "
              "Success vs single-large (pp) | Beats single-large on expected value |",
              "|---|---|---|---|---|---|---|"]
    for row in decision_table(sums):
        if row["kind"] == "all":
            continue
        be_txt = row["verdict"]
        lines.append(f"| {row['kind']} | {row['architecture']} | {row['success_rate']:.0%} | "
                     f"{row['cost_per_task'] * 1000:.2f} | {row['cost_vs_baseline']:.2f}x | "
                     f"{row['success_delta_pp']:+.1f} | {be_txt} |")
    if sweep:
        lines += ["", "## Sensitivity (compare and aggregate tasks only)", "",
                  "| Factor | Value | single-large success | multi-mixed success | Gap (pp) | "
                  "single-large cost (m$) | multi-mixed cost (m$) | single-large p50 (s) | "
                  "multi-mixed p50 (s) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for p in sweep:
            lines.append(f"| {p.factor} | {p.value:g} | {p.single_success:.0%} | "
                         f"{p.multi_success:.0%} | {p.gap * 100:+.1f} | "
                         f"{p.single_cost * 1000:.2f} | {p.multi_cost * 1000:.2f} | "
                         f"{p.single_latency_s:.1f} | {p.multi_latency_s:.1f} |")
    lines += ["", "## How to read this", "",
              "* Success means correct **and** grounded (every citation was retrieved).",
              "* Break-even value: if a correct answer is worth more than this, the architecture "
              "beats single-large on expected value (value x success minus cost).",
              "* Results from the simulator depend on its stated assumptions; see "
              "`docs/methodology.md`. Re-run with `ARCHEVAL_LLM_PROVIDER=azure` to measure a "
              "real model.", ""]
    return "\n".join(lines)


def charts(sums: list[Summary], sweep: list[SweepPoint], out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    paths = []

    # 1. The frontier: success against cost, one point per architecture, labelled directly.
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    overall = [s for s in sums if s.kind == "all"]
    for s in overall:
        colour = SERIES_2 if s.architecture == "routed" else SERIES_1
        ax.scatter(s.cost_per_task * 1000, s.success_rate * 100, s=70, color=colour,
                   edgecolors=SURFACE, linewidths=2, zorder=3)
        # When two points sit side by side, label the cheaper one on its left.
        span = max(o.cost_per_task for o in overall)
        crowded = any(o is not s and abs(o.success_rate - s.success_rate) < 0.01
                      and 0 < o.cost_per_task - s.cost_per_task < 0.25 * span for o in overall)
        ax.annotate(s.architecture, (s.cost_per_task * 1000, s.success_rate * 100),
                    xytext=(-7, -3) if crowded else (7, -3), textcoords="offset points",
                    ha="right" if crowded else "left", fontsize=9, color=INK)
    ax.set_xlabel("Cost per task (USD x 1000, illustrative prices)")
    ax.set_ylabel("Task success rate (%)")
    ax.set_title("Success vs cost, all 60 tasks", loc="left", fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, max(s.cost_per_task for s in overall) * 1000 * 1.3)
    fig.tight_layout()
    paths.append(out / "frontier.png")
    fig.savefig(paths[-1], facecolor=SURFACE)
    plt.close(fig)

    # 2. Success and cost by task type: small multiples, two separate rows (no dual axis).
    archs = [s.architecture for s in overall]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), dpi=160, sharey="row")
    for col, kind in enumerate(KINDS):
        rows = {s.architecture: s for s in sums if s.kind == kind}
        for row, (metric, label) in enumerate([("success_rate", "Success (%)"),
                                               ("cost_per_task", "Cost/task (m$)")]):
            ax = axes[row][col]
            vals = [getattr(rows[a], metric) * (100 if row == 0 else 1000) for a in archs]
            ax.barh(archs, vals, color=SERIES_1, height=0.6)
            for y, v in enumerate(vals):
                ax.text(v, y, f" {v:.0f}" if row == 0 else f" {v:.1f}", va="center",
                        fontsize=8, color=MUTED)
            ax.invert_yaxis() if col == 0 else None
            ax.grid(True, axis="x", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(kind, loc="left", fontsize=10)
                ax.set_xlim(0, 115)
            else:
                ax.set_xlim(0, max(getattr(r, metric) for r in rows.values()) * 1000 * 1.35)
            if col == 0:
                ax.set_ylabel(label)
    fig.suptitle("By task type: success (top) and cost per task (bottom)", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    paths.append(out / "by_task_type.png")
    fig.savefig(paths[-1], facecolor=SURFACE)
    plt.close(fig)

    # 3. Step budget sweep: success and cost as two panels (two measures, two axes).
    budget = [p for p in sweep if p.factor == "single_step_budget"]
    if budget:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=160)
        xs = [p.value for p in budget]
        panels = [("Success on breadth tasks (%)", 100, "single_success", "multi_success"),
                  ("Cost per task (m$)", 1000, "single_cost", "multi_cost")]
        for ax, (label, scale, s_key, m_key) in zip(axes, panels, strict=True):
            for key, colour, name in ((s_key, SERIES_1, "single-large"),
                                      (m_key, SERIES_2, "multi-mixed (fixed budget)")):
                ys = [getattr(p, key) * scale for p in budget]
                ax.plot(xs, ys, color=colour, linewidth=2, marker="o", markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=2, label=name)
                if scale == 1000:  # the success lines converge, so label the cost panel only
                    ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=8)
            ax.set_xlabel("Single agent step budget (max tool-loop iterations)")
            ax.set_ylabel(label)
            ax.set_xlim(min(xs) - 1, max(xs) * (1.6 if scale == 1000 else 1.08))
            ax.set_ylim(0, None if scale == 1000 else 105)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")
        fig.suptitle("A bigger step budget closes the quality gap, at a higher cost per task",
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        paths.append(out / "step_budget.png")
        fig.savefig(paths[-1], facecolor=SURFACE)
        plt.close(fig)
    return paths


def write(run: BenchmarkRun, out: Path, sweep: list[SweepPoint] | None = None,
          with_charts: bool = True) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    sums = summarize(run.records)
    sweep = sweep or []
    with (out / "runs.jsonl").open("w") as f:
        for r in run.records:
            f.write(json.dumps(r.to_dict()) + "\n")
    summary = {"settings": run.settings, "seconds": run.seconds,
               "summaries": [s.to_dict() for s in sums], "decision": decision_table(sums),
               "sensitivity": [p.to_dict() for p in sweep]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "report.md").write_text(markdown(run, sums, sweep))
    if with_charts:
        summary["charts"] = [str(p) for p in charts(sums, sweep, out)]
    return summary
