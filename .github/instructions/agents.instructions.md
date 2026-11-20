---
applyTo: "src/archeval/agents/**/*.py"
---
- Keep single and multi-agent prompts on the shared GROUNDING block.
- Workers return `FINDING | product | key | value | source` lines; the lead must cite sources.
- Budgets come from `settings.limits`; do not hard-code iteration counts.
