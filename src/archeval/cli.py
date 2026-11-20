"""Command line: ``archeval ask``, ``archeval bench``, ``archeval gate``, ``archeval ingest``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from archeval.agents.system import ARCHITECTURES, AgentSystem
from archeval.config import get_settings
from archeval.corpus.contoso import DOCUMENTS
from archeval.evaluation import report
from archeval.evaluation.economics import pick, summarize
from archeval.evaluation.runner import DEFAULT_ARCHS, run_benchmark
from archeval.evaluation.sensitivity import sweep as run_sweep
from archeval.llm.clients import ClientFactory
from archeval.retrieval.factory import build_store

app = typer.Typer(add_completion=False, help="Single-agent vs multi-agent evaluation.")


@app.command()
def ask(question: str, architecture: str = typer.Option("routed", "--arch", "-a")) -> None:
    """Answer one question with one architecture and print the answer with its cost."""

    async def main() -> None:
        settings = get_settings()
        store = await build_store(settings, seed=True)
        result = await AgentSystem(settings, store, ClientFactory(settings)).run(
            question, ARCHITECTURES[architecture])
        out = result.to_dict()
        out.pop("ledger")
        typer.echo(json.dumps(out, indent=2))

    asyncio.run(main())


@app.command()
def bench(out: Path = typer.Option(Path("reports/latest")),
          repeats: int = typer.Option(3, min=1, max=20),
          archs: str = typer.Option(",".join(DEFAULT_ARCHS)),
          sensitivity: bool = typer.Option(True, help="Also run the sensitivity sweep."),
          charts: bool = typer.Option(True)) -> None:
    """Run every task on every architecture and write the report."""

    async def main() -> None:
        settings = get_settings()
        store = await build_store(settings, seed=True)
        run = await run_benchmark(settings, archs.split(","), repeats=repeats, store=store)
        points = await run_sweep(settings, repeats=repeats) \
            if sensitivity and settings.llm_provider == "sim" else []
        summary = report.write(run, out, points, with_charts=charts)
        for s in summary["summaries"]:
            if s["kind"] == "all":
                typer.echo(f"{s['architecture']:13} success {s['success_rate']:.1%}  "
                           f"cost/task ${s['cost_per_task']:.5f}  "
                           f"latency p50 {s['latency_p50_s']:.1f}s")
        typer.echo(f"report: {out / 'report.md'}")

    asyncio.run(main())


@app.command()
def gate(summary: Path = typer.Argument(Path("reports/latest/summary.json")),
         min_success: float = 0.95, max_cost_ratio: float = 0.75,
         architecture: str = "routed") -> None:
    """CI gate: fail if the production architecture regresses on quality or cost."""
    data = json.loads(summary.read_text())
    rows = {(s["architecture"], s["kind"]): s for s in data["summaries"]}
    prod, base = rows[(architecture, "all")], rows[("single-large", "all")]
    ratio = prod["cost_per_task"] / base["cost_per_task"]
    ok = prod["success_rate"] >= min_success and ratio <= max_cost_ratio
    typer.echo(f"{architecture}: success {prod['success_rate']:.1%} (min {min_success:.0%}), "
               f"cost {ratio:.2f}x single-large (max {max_cost_ratio:.2f}x) -> "
               f"{'PASS' if ok else 'FAIL'}")
    raise typer.Exit(0 if ok else 1)


@app.command()
def ingest() -> None:
    """Embed the knowledge base and upsert it into the configured vector store."""

    async def main() -> None:
        settings = get_settings()
        store = await build_store(settings, seed=False)
        await store.upsert(DOCUMENTS)
        typer.echo(f"upserted {len(DOCUMENTS)} documents into {settings.vector_store}")

    asyncio.run(main())


def _quick() -> None:  # pragma: no cover - used by `python -m archeval.cli`
    async def main() -> None:
        settings = get_settings()
        store = await build_store(settings, seed=True)
        run = await run_benchmark(settings, repeats=1, store=store)
        sums = summarize(run.records)
        print(pick(sums, "routed"))

    asyncio.run(main())


if __name__ == "__main__":  # pragma: no cover
    app()
