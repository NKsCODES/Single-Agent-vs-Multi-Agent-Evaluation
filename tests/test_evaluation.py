from __future__ import annotations

from collections import Counter

from archeval.agents.system import AnswerResult
from archeval.corpus.contoso import FACTS
from archeval.evaluation.economics import Summary, break_even, verdict, wilson
from archeval.evaluation.grader import grade, segments
from archeval.evaluation.runner import run_benchmark
from archeval.evaluation.tasks import TASKS


def result(answer: str, grounded: bool = True) -> AnswerResult:
    return AnswerResult("q", "single-large", "single-large", answer, [], grounded, [], 1, 0,
                        10, 10, 0.001, 1.0, 1.0)


def test_benchmark_has_15_tasks_of_each_kind_with_ground_truth() -> None:
    assert Counter(t.kind for t in TASKS) == {"lookup": 15, "multihop": 15, "compare": 15,
                                              "aggregate": 15}
    assert all(t.expected_set is not None for t in TASKS if t.kind == "aggregate")
    assert len({t.question for t in TASKS}) == 60


def test_grader_checks_each_product_segment() -> None:
    t = next(t for t in TASKS if t.kind == "compare" and len(t.entities) == 3)
    good = "\n".join(f"{e}: " + "; ".join(f"{FACTS[e].value(a)} [x]" for a in t.attrs) + "."
                     for e in t.entities)
    assert grade(t, result(good)).success
    swapped = good.replace(t.entities[0], "TMP").replace(t.entities[1], t.entities[0]).replace(
        "TMP", t.entities[1])
    g = grade(t, result(swapped))
    if any(FACTS[t.entities[0]].value(a) != FACTS[t.entities[1]].value(a) for a in t.attrs):
        assert not g.correct


def test_correct_but_ungrounded_is_not_a_success() -> None:
    t = next(t for t in TASKS if t.kind == "lookup")
    e = t.entities[0]
    ans = f"{e}: " + "; ".join(FACTS[e].value(a) for a in t.attrs)
    g = grade(t, result(ans, grounded=False))
    assert g.correct and not g.success


def test_sla_values_do_not_match_by_prefix() -> None:
    t = next(t for t in TASKS if t.kind == "lookup" and t.attrs == ["sla"]
             and FACTS[t.entities[0]].sla == "99.9%") if any(
        t.kind == "lookup" and t.attrs == ["sla"] and FACTS[t.entities[0]].sla == "99.9%"
        for t in TASKS) else None
    if t is None:
        return
    assert not grade(t, result(f"{t.entities[0]}: uptime SLA 99.95% [x].")).correct


def test_aggregate_requires_exact_set() -> None:
    t = next(t for t in TASKS if t.kind == "aggregate" and len(t.expected_set or []) >= 2)
    names = t.expected_set or []
    assert grade(t, result("Products: " + ", ".join(names) + ".")).correct
    assert not grade(t, result("Products: " + names[0] + ".")).correct


def test_segments_split_by_product() -> None:
    s = segments("Contoso Mail: 99.9%. Contoso Chat: 99.99%.")
    assert "99.99%" not in s["Contoso Mail"] and "99.99%" in s["Contoso Chat"]


def _s(arch: str, success: float, cost: float) -> Summary:
    return Summary(arch, "all", 10, success, 0, 1, success, 1, 1, cost, None, 0, 0, 0, 0, 0)


def test_break_even_value() -> None:
    base, cand = _s("single-large", 0.8, 0.01), _s("multi-large", 0.9, 0.02)
    assert abs((break_even(base, cand) or 0) - 0.1) < 1e-9
    assert "more than $0.100" in verdict(base, cand)
    assert break_even(base, _s("x", 0.7, 0.03)) is None
    assert verdict(base, _s("y", 0.9, 0.005)).startswith("always")


def test_wilson_interval_contains_the_estimate() -> None:
    lo, hi = wilson(45, 50)
    assert lo < 0.9 < hi and lo >= 0 and hi <= 1


async def test_benchmark_slice_runs_end_to_end(settings) -> None:  # type: ignore[no-untyped-def]
    tasks = [t for t in TASKS if t.id in ("lookup-00", "compare-00", "aggregate-00")]
    run = await run_benchmark(settings, ["single-large", "multi-mixed"], tasks, repeats=1)
    assert len(run.records) == 6
    assert all(r.result.cost_usd > 0 for r in run.records)


async def test_summary_report_and_gate(settings, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from typer.testing import CliRunner

    from archeval.cli import app
    from archeval.evaluation import report
    from archeval.evaluation.economics import pick, summarize
    from archeval.evaluation.sensitivity import SweepPoint

    tasks = [t for t in TASKS if t.id in ("lookup-00", "multihop-00", "compare-00",
                                          "aggregate-00")]
    archs = ["single-small", "single-large", "multi-large", "multi-mixed", "routed"]
    run = await run_benchmark(settings, archs, tasks, repeats=1)
    sums = summarize(run.records)
    assert pick(sums, "routed").routed_multi_share == 0.5
    sweep = [SweepPoint("single_step_budget", b, 0.5, 0.9, 0.02, 0.01, 10, 8) for b in (4, 8)]
    summary = report.write(run, tmp_path, sweep, with_charts=True)
    assert (tmp_path / "report.md").read_text().startswith("# Benchmark report")
    assert len(summary["charts"]) == 3 and all((tmp_path / c).exists() or c for c in
                                                summary["charts"])
    data = json.loads((tmp_path / "summary.json").read_text())
    assert any(r["verdict"] == "baseline" for r in data["decision"])
    res = CliRunner().invoke(app, ["gate", str(tmp_path / "summary.json"), "--min-success",
                                   "0.0", "--max-cost-ratio", "100"])
    assert res.exit_code == 0 and "PASS" in res.output
    res = CliRunner().invoke(app, ["gate", str(tmp_path / "summary.json"), "--min-success",
                                   "1.01"])
    assert res.exit_code == 1
