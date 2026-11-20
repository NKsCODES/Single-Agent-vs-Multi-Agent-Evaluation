# Benchmark report: single agent vs orchestrator-worker

Provider: `sim`. Runs: 1500 (5 seed(s) x 60 tasks x 5 architectures). Cost unit: USD per 1,000 tasks shown as $ per task x 1000 ("m$").

## Overall

| Architecture | Success (95% CI) | Grounded | Fact recall | Cost/task (m$) | Cost/success (m$) | Tokens/task | Calls/task | Latency p50 (s) |
|---|---|---|---|---|---|---|---|---|
| multi-large | 99.7% (98% to 100%) | 100.0% | 100.0% | 19.48 | 19.54 | 7,490 | 9.1 | 10.3 |
| multi-mixed | 99.7% (98% to 100%) | 100.0% | 100.0% | 8.39 | 8.42 | 8,343 | 9.6 | 7.8 |
| routed | 99.7% (98% to 100%) | 100.0% | 100.0% | 7.01 | 7.04 | 7,747 | 8.6 | 5.5 |
| single-large | 86.7% (82% to 90%) | 99.3% | 92.7% | 15.09 | 17.41 | 6,519 | 4.2 | 6.6 |
| single-small | 75.7% (71% to 80%) | 98.7% | 87.2% | 3.19 | 4.22 | 6,930 | 4.5 | 3.2 |

## By task type

| Task type | Architecture | Success | Cost/task (m$) | Cost vs single-large | Success vs single-large (pp) | Beats single-large on expected value |
|---|---|---|---|---|---|---|
| lookup | multi-large | 100% | 5.54 | 2.18x | +0.0 | never (no better, costs more) |
| lookup | multi-mixed | 100% | 3.21 | 1.26x | +0.0 | never (no better, costs more) |
| lookup | routed | 100% | 0.57 | 0.23x | +0.0 | always (as good and cheaper) |
| lookup | single-large | 100% | 2.54 | 1.00x | +0.0 | baseline |
| lookup | single-small | 100% | 0.57 | 0.23x | +0.0 | always (as good and cheaper) |
| multihop | multi-large | 100% | 8.11 | 1.66x | +0.0 | never (no better, costs more) |
| multihop | multi-mixed | 100% | 3.97 | 0.81x | +0.0 | always (as good and cheaper) |
| multihop | routed | 100% | 1.08 | 0.22x | +0.0 | always (as good and cheaper) |
| multihop | single-large | 100% | 4.90 | 1.00x | +0.0 | baseline |
| multihop | single-small | 100% | 1.08 | 0.22x | +0.0 | always (as good and cheaper) |
| compare | multi-large | 99% | 22.44 | 1.15x | +10.7 | if a correct answer is worth more than $0.027 |
| compare | multi-mixed | 99% | 9.84 | 0.50x | +10.7 | always (better and cheaper) |
| compare | routed | 99% | 9.84 | 0.50x | +10.7 | always (better and cheaper) |
| compare | single-large | 88% | 19.52 | 1.00x | +0.0 | baseline |
| compare | single-small | 72% | 4.64 | 0.24x | -16.0 | only if a correct answer is worth less than $0.093 |
| aggregate | multi-large | 100% | 41.82 | 1.25x | +41.3 | if a correct answer is worth more than $0.020 |
| aggregate | multi-mixed | 100% | 16.55 | 0.50x | +41.3 | always (better and cheaper) |
| aggregate | routed | 100% | 16.55 | 0.50x | +41.3 | always (better and cheaper) |
| aggregate | single-large | 59% | 33.38 | 1.00x | +0.0 | baseline |
| aggregate | single-small | 31% | 6.49 | 0.19x | -28.0 | only if a correct answer is worth less than $0.096 |

## Sensitivity (compare and aggregate tasks only)

| Factor | Value | single-large success | multi-mixed success | Gap (pp) | single-large cost (m$) | multi-mixed cost (m$) | single-large p50 (s) | multi-mixed p50 (s) |
|---|---|---|---|---|---|---|---|---|
| single_step_budget | 4 | 49% | 99% | +50.7 | 15.78 | 13.20 | 10.3 | 12.3 |
| single_step_budget | 6 | 73% | 99% | +26.0 | 26.45 | 13.20 | 15.1 | 12.3 |
| single_step_budget | 10 | 84% | 99% | +15.3 | 49.05 | 13.20 | 20.9 | 12.3 |
| single_step_budget | 16 | 100% | 99% | -0.7 | 59.99 | 13.20 | 20.9 | 12.3 |
| single_step_budget | 24 | 100% | 99% | -0.7 | 60.11 | 13.20 | 20.9 | 12.3 |
| degradation_multiplier | 0 | 84% | 99% | +15.3 | 26.27 | 13.20 | 15.2 | 12.3 |
| degradation_multiplier | 0.5 | 78% | 99% | +21.3 | 26.38 | 13.20 | 15.1 | 12.3 |
| degradation_multiplier | 1 | 73% | 99% | +26.0 | 26.45 | 13.20 | 15.1 | 12.3 |
| degradation_multiplier | 2 | 65% | 99% | +34.7 | 26.52 | 13.20 | 15.0 | 12.3 |
| finding_miss_factor | 0.2 | 73% | 99% | +26.0 | 26.45 | 13.20 | 15.1 | 12.3 |
| finding_miss_factor | 0.5 | 73% | 97% | +23.3 | 26.45 | 13.76 | 15.1 | 12.8 |
| finding_miss_factor | 1 | 73% | 90% | +16.7 | 26.45 | 14.39 | 15.1 | 14.1 |

## How to read this

* Success means correct **and** grounded (every citation was retrieved).
* Break-even value: if a correct answer is worth more than this, the architecture beats single-large on expected value (value x success minus cost).
* Results from the simulator depend on its stated assumptions; see `docs/methodology.md`. Re-run with `ARCHEVAL_LLM_PROVIDER=azure` to measure a real model.
