# Methodology

## Question

Is an orchestrator-worker multi-agent system worth its extra cost, and for which requests?
Anthropic reported that a multi-agent research system (Claude Opus 4 lead, Sonnet 4 subagents)
beat a single Opus 4 agent by 90.2% on its internal research eval, while multi-agent systems
used about 15 times the tokens of a chat interaction (single agents about 4 times). The gain
was largest on breadth-first queries. This project measures the same trade-off on a task suite
where the answer is known, so success can be graded without a judge model.

## Setup

* **Knowledge base.** 12 fictional Contoso products, each with an overview, a security page, a
  support policy and release notes, plus a catalog and distractor pages: 52 documents. Facts are
  generated from a seed and kept as ground truth.
* **Retrieval.** One `search_knowledge_base` tool, top 3 passages, hybrid vector plus BM25 fused
  with Reciprocal Rank Fusion. In Azure this is a single Cosmos DB for NoSQL query
  (`ORDER BY RANK RRF(VectorDistance(...), FullTextScore(...))`); offline it is an in-process
  index with the same fusion.
* **Architectures** (all on Microsoft Agent Framework, same tool, prompts' grounding rules and
  budgets):

| Name | Lead or single agent | Workers |
|---|---|---|
| single-small | small model, up to 6 tool-loop iterations | none |
| single-large | large model, up to 6 iterations | none |
| multi-large | large lead, up to 4 iterations, up to 4 parallel subtasks per turn | large, 4 iterations each |
| multi-mixed | large lead | small workers |
| routed | heuristic router: single-small for narrow questions, multi-mixed for breadth | |

* **Tasks.** 60 questions, 15 of each type: lookup (one product, one or two facts), multihop
  (find which product introduced a feature, then its facts), compare (3 to 8 products, 2 or 3
  facts each), aggregate (scan all 12 products and filter on two facts).
* **Grading.** Deterministic. A task succeeds when the answer is correct (every requested
  fact for every product, the right product for multihop, the exact set for aggregate) **and**
  grounded (every citation is a passage retrieved in that run, and no fact-bearing sentence is
  uncited). Correct but uncited answers fail: a lucky guess is not a production success.
* **Cost.** Tokens are counted from the API's `usage` field by an httpx hook on every model
  call (lead, workers and single agents alike). Prices are placeholders in USD per 1M tokens
  (large 2.00 in / 8.00 out, small 0.40 / 1.60). Set your own; ratios matter more than dollars.
* **Latency.** Critical path: lead and single-agent calls add up; workers in the same wave run
  in parallel, so a wave costs as long as its slowest worker.
* **Seeds.** 5 seeds per task and architecture: 1,500 graded runs. Success rates come with 95%
  Wilson intervals.

## The offline model simulator, and what it assumes

To make the benchmark free, fast and reproducible in CI, the default provider is an
OpenAI-compatible simulator served through `httpx.MockTransport`. The production code path is
unchanged (Agent Framework, OpenAI SDK, HTTP); only the transport differs. The simulator never
sees the ground truth. It reads what is in the request and behaves according to explicit
parameters:

| Assumption | Parameter | Default | Swept? |
|---|---|---|---|
| A model notices a fact in its context with probability p, falling past a knee as the prompt grows (context rot) | base recall, slope per 1k tokens, knee | large 0.97, -0.025/1k; small 0.92, -0.045/1k; knee 2,000 tokens | yes (0x to 2x slope) |
| Facts handed over as one-line structured findings are easier to read than inside passages | finding_miss_factor | 0.2 (miss chance x0.2) | yes (0.2 to 1.0) |
| When a fact is missing at answer time, a model sometimes guesses | hallucination rate | large 5%, small 15% | no |
| A model issues up to N searches per turn and rephrases a failed query up to 3 times | parallel_searches, max_query_attempts | 2, 3 | no |
| The lead re-delegates missing facts once | lead_followup_waves | 1 | no |
| Prompts over the window fail | context_window | 16,000 tokens | no |

Iteration budgets are not simulated: Agent Framework enforces them
(`function_invocation_configuration["max_iterations"]`) exactly as it would against Azure, and
`ContextWindowCompactionStrategy` compacts long single-agent histories.

## What the sweep says about these assumptions

* **Step budget is the biggest lever.** With 16 or more tool-loop iterations the single agent
  matches the multi-agent system on breadth tasks, but its context is re-read on every turn, so
  it costs about 4.5 times more per task and has a higher median latency. The multi-agent design
  buys the same quality with fresh, short contexts and parallel workers.
* **Context degradation matters but does not explain everything.** With no degradation at all
  (0x), the single agent still trails by 15 points at the default budget.
* **The structured hand-off assumption is not what carries the result.** If findings were no
  easier to read than raw passages (factor 1.0), multi-mixed drops from 99% to 90% on breadth
  tasks and still leads by 17 points.

## Limits

* Simulator results are a model of model behaviour, not a measurement of GPT or Claude. Absolute
  numbers will differ on real models; run `live-eval` (Azure OpenAI) for that. The harness,
  grading and cost accounting are the same either way.
* The simulator writes short answers and no reasoning text, so token ratios are lower than
  real systems (multi-large used 1.15 times the tokens of single-large here; Anthropic reports
  multi-agent around 15 times chat versus 4 times for single agents).
* The tasks are synthetic and structured. Open-ended research, where Anthropic saw the largest
  gains, is harder to grade without a judge model.
