# rag-lab

[![CI](https://github.com/rikitikitav1/rag-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rikitikitav1/rag-lab/actions/workflows/ci.yml)

A RAG system over a personal knowledge base and external IT repositories. It answers technical questions and returns the sources it retrieved. Retrieval is built **from primitives** (no framework); the agent was built from primitives too, then ported to **LangGraph** and the port was measured against the original before the original was retired (see [the entry](docs/experiments/2026-08-26_the-same-agent-written-four-ways.md)). Inference is **local** (Ollama on a single GPU).

This is a **showcase lab**: a bench for practicing LLM/RAG engineering approaches by hand (retrieval, LLM-as-judge eval, model lifecycle, reranking, async queues) and showing results as numbers. Not a production service, a playground for approaches.

> In Russian: [docs/README_ru.md](docs/README_ru.md) · Hands-on scenarios and commands: [docs/use_cases.md](docs/use_cases.md) · Experiments log: [docs/experiments.md](docs/experiments.md)

## What it is

- **Corpus variants**: the same corpus can hold several chunkings side by side, each with its own partial vector index and its own policy in the run snapshot. Re-indexing stops being a one-way border and becomes a swept parameter (see [the entry](docs/experiments/2026-08-26_a-corpus-you-can-keep-two-of.md)).
- **Hybrid retrieval**: vector search (pgvector) + full-text (Postgres FTS with per-language stemming), fused via **RRF** (Reciprocal Rank Fusion). Filter by hierarchical categories (ltree), a distance threshold with an honest refusal on out-of-corpus questions.
- **Local models** through Ollama, keyed by role: generation, embeddings (`bge-m3`), LLM judge. A role is decoupled from a concrete model: which model serves a role lives in the DB and can be switched at runtime.
- **Multi-source**, one taxonomy: personal notes (ru) + Devinterview-io interview repos (en, ~170 repos). Per-source ingestion strategies.
- **Eval bench, 5 quality axes**: retrieval (hit@k / MRR), faithfulness (is the answer grounded in the context), relevance (does it answer to the point), completeness (does it cover the reference answer), refusal accuracy, all via **LLM-as-judge** with structured output. Scores are reported per pool: answers grounded in the corpus, answers grounded in an external tool (`remote_grounding` measures grounding in what the tool returned, not whether the tool was right), and refusals, alongside the answer rate and the share of answers with no support at all. A run also records an outcome per question (`answered`, `refused`, `unsupported_answer`, `narrated_call`, `error`), because counting an unsupported answer as a refusal flatters exactly the policy under test.
- **Async job queue**: heavy operations (model pull/delete, corpus indexing, embedding the question bank) go to a Postgres-backed queue processed by a worker. The service depends only on Postgres: Ollama may be unavailable at startup, jobs defer/retry, the app does not crash.
- **Reranking (opt-in)**: a cross-encoder (`bge-reranker-v2-m3`) on top of hybrid retrieval (retrieve-wide → rerank → narrow), toggled by a flag (per-request / per-run); A/B tested on a cross-lingual set.
- **Agent on LangGraph**: a tool-calling graph where the model decides when to search the corpus, may refine the query and multi-hop, then answers. It started as a hand-rolled loop; the loop and the graph were run against each other on the same questions, agreed within the bench's own noise, and the loop was dropped. The policies around the loop (coverage gate, topic axis, tool admission) stayed ours and are measured, not assumed. Selectable as an eval pipeline (`pipeline: agent`) and benchmarked head-to-head against single-shot RAG (see the experiments log).
- **Route-driven eval platform**: generating non-circular eval sets (LLM paraphrase of interview questions + translation to ru), importing questions from a file, runs and judging, all bulk through the queue; observability via the request log (`question-log`) and jobs with `elapsed`.
- **Layering**: transport-neutral `use_cases` → thin adapters (CLI / FastAPI REST).

## Stack

Python · PostgreSQL + pgvector · SQLAlchemy 2.0 (sync psycopg + async asyncpg) · Ollama (GPU, OpenAI-compatible API) · FastAPI · dbmate (migrations) · uv/pyproject · Docker Compose.

## Models and prompts architecture

- **Model / ModelRole**: `Model` (name + status: available/loading/ready), `ModelRole` (role as PK → one model per role by construction, FK `ON DELETE RESTRICT` so the DB refuses to delete an assigned model). The resolver `llm.resolve_name(role)` reads the name from the DB, inference params come from `config.yaml`.
- **Prompt**: versioned in the DB (`purpose` + `version`, exactly one `active` per purpose). Prompt sources are files in `prompts/` (format `<purpose>.v<N>.txt`), the seed loads them into the DB, the freshest version becomes active.
- **Bootstrap on startup** (idempotent): ensure Model rows from config → seed roles → reconcile with Ollama (not pulled → status `loading` + a pull job) → enqueue indexing only if the database holds no corpus variant at all → enqueue embedding of questions without a vector, unless such a job is already queued. A named but empty variant is logged, not indexed: indexing a variant is a deliberate, measured step.

## Configuration

Everything tunable lives in **`config.yaml`** (mounted into the container):

- `llm.roles` - model + `options` per role (`generation` / `embedding` / `judging` / `paraphrasing`); `llm.candidates` - models to pull but not assign.
- `service.retrieval` - `distance_threshold`, `results_limit`, `rrf_k`, candidate limits, and the keyword-leg switches `keyword_query` (`and` joins every lexeme, `or` fires on any), `keyword_rank`, `keyword_norm`, `query_lang` (`langdetect` or `cyrillic_ratio`). Their values were picked by measurement and are recorded in every run's snapshot.
- `service.rerank` - `enabled`, `model`, `candidates`, `top`.
- `service.agent` - `max_hops` (agent hop cap), `fallback_policy` (`corpus_first` / `corpus_first_weak` / `agent_choice`), `gate_signal` (what calls retrieval weak: `distance`, `cross_encoder` or `either`) with its thresholds `weak_distance` and `weak_threshold`, `gate_candidates` (how many hits the cross-encoder scores), `topic_threshold` (the topic axis: distance to the nearest chunk above which the run refuses instead of reaching out, 0.50 by default, `0` in a run switches it off).
- `service.ingestion` - `chunk_max_size`, `batch_size`, `commit_size`.
- `service.corpus` - `description`, `variant` (which corpus variant everything reads by default) and `variants` (the chunking policy of each, recorded in run snapshots).
- `service.sources` - sources (interview repos and their base_url).
- `postgres` - DB connection.

Source-specific category trees stay in code.

## Quickstart

```bash
docker compose up -d
curl localhost:8000/readiness            # pg required, ollama soft
curl -X POST localhost:8000/v1/chat/question \
  -H 'Content-Type: application/json' -d '{"text": "What is a hash table?"}'
# Swagger: http://localhost:8000/docs
```

No authentication by design (REST, `/mcp`, `/mcp-ops` are all open): this is a local lab bound to 127.0.0.1. Do not expose it to a network as is.

The first `up` pulls ~16 GB of models and builds the index (~5-10 min, watch `docker compose logs -f worker`). The server starts **without waiting** for models and indexing, so the first requests may refuse until the corpus fills up.

Full hands-on scenarios (mini-eval to numbers, reranking A/B, importing your own questions, browsing logs) and the complete command reference: **[docs/use_cases.md](docs/use_cases.md)**.

## Architecture

![Architecture](docs/diagrams/architecture.svg)

Diagrams are D2 sources in `docs/diagrams/`, rendered by `scripts/render_diagrams.sh`; CI fails if a committed SVG drifts from its source.

## Compose services

| Service | Role |
|---------|------|
| `postgresql` | Postgres + pgvector, the only hard dependency |
| `dbmate` | applies migrations, runs to completion before the rest |
| `seed` | loads prompts and the question bank, runs once after migrations |
| `rag-lab` | FastAPI server (uvicorn) + bootstrap on startup |
| `worker` | processes the job queue (pull/delete/index/embed/paraphrase/eval/judge) |
| `ollama` | local inference on GPU |

### Environment knobs

Everything tunable about the pipeline lives in `config.yaml`; the environment only carries what depends on the machine or must stay out of the repo. Copy `.env.example` to `.env`: compose picks it up, and every value has a working default.

| Variable | Default | What it does |
|----------|---------|--------------|
| `RERANK_DEVICE` | `cuda` (worker), `cpu` (API) | Where the cross-encoder runs. The asymmetry is deliberate: eval runs rerank in a phase that owns the card, while interactive answers share it with ollama, and a reranker resident there would evict the generator on every question. `auto` picks cuda when a card is visible; CUDA OOM falls back to CPU with a warning. |
| `LLM_TIMEOUT` | `120` | Seconds per completion. A 70b model on CPU needs minutes; the default kills such runs mid-flight. |
| `WORKER_RERANK_DEVICE` | `cpu` | Where the worker runs the cross-encoder. The corpus-first gate scores a handful of pairs per question, and on this card a resident reranker fights ollama for VRAM during agent runs; the phased eval path sets `cuda` explicitly when it owns the GPU. |
| `OLLAMA_CONTEXT_LENGTH` | `8192` | Context window the server loads models with. Ollama defaults to 4096 and drops whole messages to fit, reporting the trimmed count, so an over-long prompt never appears as a number above the window. The longest single hop ever logged here is 4075 tokens, so 8192 covers it twice over. Raising it costs VRAM for the KV cache and the embedder shares the card: at 14336 the two evict each other on every switch (112 model loads and 16s per question against 12 loads and 4s). After a change, check `llm.model_spilled_to_cpu` in the log, and note that the run snapshot records the window the server reports rather than this value. |
| `WORKER_QUEUES` | `default,io` | Queue lanes the worker serves, one thread each. Network and disk jobs (model pulls, MCP health) live on `io` so they never wait behind GPU work. |
| `HF_TOKEN`, `CONTEXT7_API_KEY` | empty | Secrets for external MCP integrations. Only variables allowlisted in `config.yaml` (`mcp_integrations.secret_env`) are ever read. |

## REST API

Full interactive reference in Swagger at `/docs`.

List endpoints (`/v1/model`, `/v1/prompt`, `/v1/job`, `/v1/question-log`) share pagination: `limit` (default 100, max 1000), `offset`, `sort_by`, `sort_order` (`asc`/`desc`, default `desc`).

Health:
- `GET /liveness`, `GET /readiness`

Chat and search:
- `POST /v1/chat/question` (full RAG answer; optional `rerank` flag; optional `language` override `ru`/`en`)
- `POST /v1/chat/fast_question` (retrieval only, no generation)
- `POST /v1/agent/question` (agent answer; optional `max_hops`, `language`, `fallback_policy`, and `debug` for the full message trace)
- `GET /v1/categories` (category tree with chunk counts)

![Single-shot flow: hybrid retrieval, threshold, optional rerank](docs/diagrams/single_shot_flow.svg)

Model lifecycle:
- `GET /v1/model`, `GET /v1/model/{id}`, `POST /v1/model` (create enqueues a pull), `DELETE /v1/model/{id}` (409 if assigned to a role)
- `GET /v1/role`, `PUT /v1/role/{role}` (assign a model to a role)
- `GET /v1/source`, `PUT /v1/source/{id}` (enable/disable a corpus source; disabled sources are excluded from retrieval at runtime, no re-index - ablation / source-of-truth scoping)

Prompts:
- `GET /v1/prompt`, `GET /v1/prompt/{id}`, `POST /v1/prompt`, `POST /v1/prompt/{id}/activate`, `DELETE /v1/prompt/{id}`

Eval platform:
- `POST /v1/eval/paraphrase` (generate a paraphrase set), `POST /v1/eval/run` (run a set → judge; `pipeline: single_shot|agent`, per-run `rerank`, `k` retrieval-width, `max_hops`, `fallback_policy`, `gate_signal`, `weak_distance`, `topic_threshold` and `model` (generator) overrides; config only sets the defaults)
- `POST /v1/eval/experiment` (batch a parameter series: `param` (`k`, `max_hops`, `model`, `fallback_policy`, `gate_signal`, `weak_distance` or `topic_threshold`) swept over `values`, one auto-named run per value, each judged; set/pipeline/language stay fixed for a clean single-variable comparison; a `model` value absent from the registry is created and pulled, the run waits for it)
- `GET /v1/eval/misses?run_name=X` (retrieval misses for a run: in-corpus questions where the expected source was not retrieved, with expected vs retrieved)
- `GET /v1/eval/compare?runs=A&runs=B` (arms side by side split by pool: in-corpus, out-of-corpus, off-domain; per arm the judged axes, how often the answer came from a remote tool against the corpus, how often the coverage gate fired, latency avg/p50 and the outcome histogram; per pair of arms a paired Wilcoxon plus a bootstrap interval over the same questions, so a difference is reported with its size and its uncertainty instead of two averages)
- `POST /v1/questions/import` (upload a questions file, ≤5 MB; optional chained run)

![Eval pipeline](docs/diagrams/eval_pipeline.svg)

A single run does not loop per question: it goes through phases so each stage owns the GPU alone, which is what makes reranking affordable in bulk. Per question the loop needed the embedder, then the reranker, then the generator, and the three do not fit in 8 GB together, so ollama evicted and reloaded a model on every single question. Phases cost two model swaps per run instead of two per question, and the reranker finally gets a real batch: 44 s for 100 questions on the card against about 16 min on CPU.

![Phases inside one eval run](docs/diagrams/phased_run.svg)

Measured on 100 questions with reranking on: **2092s → 652s (3.2x)** while `hit@5` and `MRR` stayed identical to the third decimal. Most of the win did not come from batching, it came from noticing that the card was never actually released between phases, so ollama had been loading the generator as 26 layers of 33. The full story, including what the batch alone did *not* buy, is in [the journal entry](docs/experiments/2026-08-24_phased-eval-runs-and-the-empty-cache.md).

Experiments (first-class entity over the raw sweep route):
- `POST /v1/experiment` (creates the experiment - dataset + deterministic seed-based sample / procedure snapshot / varied param - and enqueues the run series), `GET /v1/experiment` (filtered list), `GET /v1/experiment/{id}`, `PUT /v1/experiment/{id}/conclusion`
- state machine `draft → running → aggregated → concluded` (+ `failed`); when the last judge job of the series finishes, the aggregator computes per-value metrics and an RRF composite over the three generation axes and stores them in `results` (retrieval hit@k/MRR reported per value but kept out of the fusion: hit@k is monotonic in `k`, it would confound the composite)
- results carry **paired significance statistics**, not just point estimates: for the winner vs every other value, per axis - mean paired delta (same question in both runs), bootstrap 95% CI and a Wilcoxon signed-rank p-value, plus Bonferroni-corrected significance flags over the whole test family, so the JSON itself says what survives multiple-comparison correction

One call asks the bench a question; generation, judging and aggregation happen in the background. Questions you can ask this way:

```bash
# "How many chunks should I feed the generator?" - retrieval width sweep
curl -sX POST localhost:8000/v1/experiment -H 'Content-Type: application/json' -d '{
  "name": "k_sweep", "dataset": "paraphrased_ru", "sample_size": 100,
  "pipeline": "agent", "language": "ru", "param": "k", "param_values": [1, 3, 5, 7, 10]}'

# "Does a bigger generator earn its cost on my corpus?" - model A/B
# (a model missing from the registry is pulled automatically, the run waits)
curl -sX POST localhost:8000/v1/experiment -H 'Content-Type: application/json' -d '{
  "name": "model_ab", "dataset": "paraphrased_ru", "sample_size": 100,
  "param": "model", "param_values": ["llama3.1:8b", "gemma2:9b"]}'

# "Do extra agent hops pay off?" - hop-cap sweep on a cheap 10-question sample
curl -sX POST localhost:8000/v1/experiment -H 'Content-Type: application/json' -d '{
  "name": "hops", "dataset": "paraphrased_ru", "sample_size": 10, "sample_seed": 42,
  "pipeline": "agent", "param": "max_hops", "param_values": [2, 4, 6]}'
```

When the series is judged, `GET /v1/experiment/{id}` returns per-value metrics and a composite verdict:

```json
{
  "status": "aggregated",
  "results": {
    "per_value": {"5": {"faithfulness": 7.18, "relevance": 8.9, "completeness": 6.16, "hit_at_k": 0.9, "mrr": 0.757}, "...": "..."},
    "composite": {
      "method": "rrf", "winner": "5",
      "ranking": [{"value": "5", "rrf": 0.0487}, {"value": "10", "rrf": 0.0484}],
      "pairwise": {
        "comparisons": {"5_vs_10": {"faithfulness": {"mean_delta": 0.19, "ci95": [-0.17, 0.57], "p": 0.3669, "n": 100, "significant_raw": false, "significant_bonferroni": false}, "...": "..."}},
        "method": "bonferroni", "alpha": 0.05, "tests": 15, "threshold": 0.00333
      }
    }
  }
}
```

The pairwise block is what keeps conclusions honest: an early version of this bench "concluded" k=5 beats k=10 from a composite-score gap in the third decimal - the paired test shows that comparison is a coin flip (p=0.37), and the corrected flags mark which of the 15 grid tests survive at all. See [docs/experiments.md](docs/experiments.md) for the cases where this reversed our own verdicts.

Before a grid starts, `scripts/preflight_grid.py` refuses the ways a run silently stops meaning
anything: a worker older than the newest source file (it holds its code in memory, unlike the API),
a dirty tree (the commit recorded in the snapshot would name code that never ran), a context window
the server disagrees with, models that are loaded but not on the GPU, a queue that is not empty.
`--verify` checks a finished run instead: one row per question, the expected orchestrator, and every
setting the arms have to share, from the corpus fingerprint to the prompt versions. Most of those
checks exist because the corresponding trap had already cost a grid.

Record the takeaway with `PUT /v1/experiment/{id}/conclusion` and the experiment becomes a self-contained artifact: what was varied, on what data, the numbers, the verdict.

Observability:
- `GET /v1/question-log`, `GET /v1/question-log/{id}` (answer logs; filters incl. `pipeline`, `faithfulness`/`relevance`/`completeness`, `run_name`; and over the recorded snapshot: `rerank`, `rerank_device`, `phased`, `empty_retrieval`, `max_distance` - so "show me every answer where the corpus returned nothing" is one request; detail with context)
- `GET /v1/job`, `GET /v1/job/{id}` (jobs + elapsed), `POST /v1/job/{id}/cancel` (cancels the job and its dependent judge, cooperative stop for a running eval)
- `POST /v1/job/cancel` (cancel a whole run or job type at once: cancelling id by id through a paginated listing is how a supposedly stopped eval quietly kept running)

## MCP

An MCP (Model Context Protocol) server is mounted at `/mcp` (streamable HTTP), exposing the corpus to any MCP client (Claude Desktop, Cursor, IDE agents). Built on standalone `fastmcp` and reusing the same retrieval primitives as the REST/agent paths. Tools:
- `search_corpus(query, category?)` - hybrid retrieval, returns chunks with `[source]` markers; optional category subtree filter.
- `answer_question(text, pipeline?, category?, language?)` - full RAG answer, returns `{answer, retrieved, sources}` (`agent` or `single_shot`; `category` only with `single_shot`).
- `list_categories(category?, only_top?)` - category paths with chunk counts, for discovering valid filter values before searching.

Connect: `claude mcp add --transport http rag-lab http://127.0.0.1:8000/mcp/`, or point the MCP Inspector at the same URL.

A second, separate ops server is mounted at `/mcp-ops` - an eval control plane kept off the product surface (an external client gets search/answer tools, not admin verbs):
- `run_metrics(run_name)` - aggregated eval metrics for one run (generation axes + retrieval hit@k/MRR).
- `compare_runs(run_names)` - side-by-side metrics with an RRF composite ranking (generation axes only).
- `compare_pools(run_names)` - the same runs split by pool (in-corpus / out-of-corpus / off-domain) with gate firings, latency, outcome histogram and a paired Wilcoxon per pair of runs.
- `list_jobs(status?, type?, run_name?)` / `cancel_job(id)` - job queue control, cancel takes the dependent judge down with the run.

### MCP client: the agent consumes external servers

The lab is both sides of the protocol: its own MCP server above, and an MCP *client* below. External hosted MCP servers are registered as `McpIntegration` rows and their tools join the agent's toolbox next to `search_corpus`, namespaced `integration__tool` (e.g. `deepwiki__ask_question`). The agent decides per hop whether to look outside the corpus; a successful remote call is recorded as an `mcp:` source (provenance), a failed one degrades to an error string the agent can route around.

![Agent flow with remote fallback](docs/diagrams/agent_flow.svg)

This is the policy flow, the same for every implementation of the loop. What executes it is the graph, and its picture is generated from the compiled graph itself: see [the implementations](#three-ways-to-run-the-same-agent-and-the-fourth-that-was-retired).

Registry lifecycle via `/v1/mcp_integration`:
- CRUD with filters; new integrations start `disabled`, a state machine (`disabled/active/unreachable`) separates operator intent from observed health (probes flip `active <-> unreachable`, never touch `disabled`).
- `POST /{id}/discover` - fetch the server's tool list, cache name/description/schema snapshots in the DB. The agent builds tools from this frozen cache (no network on run start, and a later description swap on the server side does not silently reach the LLM prompt - discover again to refresh).
- `POST /{id}/probe` - live ping writing `last_checked_at/last_error`; `GET /{id}/health` - cheap read of the stored state. A `check_mcp_health` job fires on every create/update (io queue lane, so it never waits behind GPU jobs).
- `allowed_tools` is an explicit allowlist: discovery shows the catalog, a human picks what the 8B model actually sees. Tool descriptions are truncated on cache; results are truncated to `max_result_chars`.

![McpIntegration state machine](docs/diagrams/mcp_state.svg)

Auth per integration is declared as `{"type": "bearer", "token_env": "HF_TOKEN"}` or `{"type": "header", "header": "...", "value_env": "..."}` - the DB stores only environment variable *names*; values come from the environment and only for variables allowlisted in `config.yaml` (`mcp_integrations.secret_env`).

Note on trust boundaries: no auth by design; anyone with API access can register an integration pointing anywhere, and allowlisted secrets will be sent to that URL. Do not expose the service.

Seeded integrations (all disabled until you enable them): DeepWiki (no auth), Hugging Face (`HF_TOKEN`), Context7 (`CONTEXT7_API_KEY`).

### Corpus-first: when the agent is allowed to look outside

Handing an 8B model a toolbox and hoping it prefers the local corpus is not a policy. `fallback_policy` makes the order explicit and, more importantly, measurable:

- `corpus_first` (default) - the run starts with `search_corpus` alone. External tools are not in the schema and `dispatch` refuses them by name, so a hallucinated tool call cannot leak out. The moment a corpus search comes back empty, the remote schemas join the toolbox from the next hop and the log records `fallback_reason: empty`.
- `corpus_first_weak` - same, plus a coverage gate over the hits that did come back. `agent.gate_signal` decides what counts as weak: `distance` (default) compares the vector distance of the best hit against `agent.weak_distance`, `cross_encoder` scores the top `agent.gate_candidates` hits and compares the best against `agent.weak_threshold`, `either` opens when one of the two says weak. Below the bar the retrieval counts as a miss (`fallback_reason: weak`) and the weak chunks are dropped from the conversation instead of being answered from, but only when an external tool exists to take over. The cross-encoder path does not need reranking to be on: with `rerank: false` it scores five pairs and leaves the ordering alone; with reranking on it reuses the scores already computed.
- `agent_choice` - everything visible from hop one (the pre-policy behaviour, kept as the A/B baseline).

What is forced is the *fact* of asking the corpus, not the wording: rephrasing and the decision to go outside stay with the model, which is where an 8B is actually decent (cross-language retrieval included).

Before any of that, a tool has to earn its place in the toolbox. Under the corpus-first policies each external tool is checked once per run against a single question: does the question already state the values its required arguments need (for DeepWiki, a repository in `owner/repo` form)? A tool that would force the model to invent an argument is not offered at all. This started as a measurement: with the gate on but no admissibility check, in-corpus questions like a Russian one about cooking carbonara were sent to a GitHub-repo tool with `repoName: "carbonara-recipe"`, and faithfulness on those questions fell to zero. Two cheaper routers were tried first and both failed: the cross-encoder scores a (question, tool description) pair at zero for everything (it is trained on question-passage relevance, not on capability blurbs), and bi-encoder cosine puts repo questions at 0.40-0.43 against 0.42 for an unrelated interview question. A rejected tool costs nothing: with no tool to hand off to, the weak chunks are kept and the run degrades to plain corpus behaviour.

When no source answers at all, the run no longer ends in silence: the final turn runs without tools, carrying a versioned instruction (`agent.no_evidence`) to say plainly that the available sources do not cover the question and not to answer from memory. Those runs come back as refusals in the language of the question instead of empty results.

When the toolbox opens, the model is told so: a versioned notice (`agent.fallback` prompt) is appended to the tool result, listing the external tools and their required arguments. It rides in the tool result on purpose. A `system` message injected mid-conversation breaks the llama3.1 chat template badly enough that the model starts printing tool calls as prose, role header included. Two more guards come from the same failure family: `dispatch` answers a call with missing required arguments by naming them (an 8B tends to reuse the argument shape of the previous tool), and a turn that narrates a call instead of issuing one gets exactly one nudge to do it for real.

Every remote failure is classified on the call path (`timeout`, `connect`, `auth`, `client`, `server`, `tool`) and lands in `metrics.tool_errors`, because the kinds carry different policies: `auth` means the key is dead and retrying is pointless, `tool` means our arguments are wrong, the rest are transient. Two traps found by probing real failures: fastmcp hides a dead host under `RuntimeError` with the real `httpx.ConnectError` on `__cause__` inside a TaskGroup `ExceptionGroup`, and an unknown tool name comes back as a server-side `ToolError` rather than any transport error.

Both the policy and the reason ride in the log snapshot, so runs before and after are comparable, and `/v1/question-log?fallback_reason=empty` pulls exactly the questions the corpus could not serve.

Measured on three pools of a hundred questions each (in-corpus, repository internals outside the corpus, and questions no available source answers). The empty rule fires **zero** times, because hybrid search always returns *something* above the distance threshold: `corpus_first` went outside 0 times out of 100. The cross-encoder gate turns that into 60 and lifts grounding on those questions from 2.31 to 5.08 (49 questions better, 17 worse, p<0.001) while the in-corpus half does not move (p=0.84) and `hit@5` stays identical across arms. Against a toolbox open from hop one the gated policy is indistinguishable everywhere (p from 0.25 to 0.87), so corpus priority is free in quality and not in time: the gate adds about two seconds per question and an external hop costs twelve more. Inside the corpus the gate fires on 23 of 100 questions, and they stay inside only because the admission check refuses to offer a repository tool for them.

Which signal calls retrieval weak got its own A/B: three arms of 200 questions (100 in-corpus, 100 outside), one variable, everything else pinned. No axis separates `distance` from the cross-encoder, and the bootstrap intervals bound how strong that claim may be: within about ±0.8 of a point in-corpus, ±1.5 outside. It opens the gate as often (90 against 92 of 100 outside) and costs 2.3s less per in-corpus question and five seconds less on the median external one, so the cross-encoder left the agent runtime and `distance` is the default. `either` opens more often (97 of 100), sends eight more questions outside and cuts the share answered from the corpus without ever leaving from 0.44 to 0.34, but what it buys is relevance (+0.99, interval [+0.21, +1.78]) and not grounding (+0.46, interval [-0.42, +1.35]), and relevance is the axis this very measurement shows to be blind to hallucination. It stays a measured option with a price tag (about 1.6s in-corpus, 3.4s outside) rather than the default. Numbers and caveats: [the journal entry](docs/experiments/2026-08-25_a-cheaper-gate-signal-and-a.md).

The same run says something less comfortable: **no policy ever refuses.** On the hundred questions nothing can answer, all three arms answer, with grounding between 0.53 and 0.77 out of ten and relevance between 8.4 and 8.8. The answers are on topic, fluent and supported by nothing, and the citation attached to a carbonara recipe is whatever chunk the hybrid returned. Deciding whether the corpus covers a question is not the same as deciding whether anyone does. Numbers and caveats: [the journal entry](docs/experiments/2026-08-25_the-gate-that-fires-and-the-refusal-that.md).

### The topic axis: refusing instead of reaching out

Coverage and topic are different questions, and the run above answers only the first. `agent.topic_threshold` adds the second: the distance from the question to the nearest chunk in the corpus, computed before any tool is offered. Above 0.50 the question is not ours, no external tool is admitted, whatever retrieval returned is dropped, and the run answers with a refusal instead. Measured against the same pools: refusals on the off-domain hundred go from 11 to 50 (paired, p<0.001), false refusals on in-corpus paraphrases stay at 0 of 100, and external questions turning into refusals instead of tool calls move from 7 to 9 of 100 (p=0.77). Both limits were written down before the run. The price shows up on hand-written questions rather than paraphrases: 4 answers of 30 are lost there, two refused and two replaced by a bare "no documents found".

The half that does not work is the more interesting one. The catch is 11 → 49 of 83 on distant topics (cooking, chemistry, law) and 0 → 1 of 17 on legacy stacks and post-cutoff technology: to a distance metric, FoxPro and Postgres are the same topic. Those questions are not off-topic, they are inside the topic and outside the corpus, which is coverage plus recency and needs a different mechanism. Numbers, the veto rules and the threshold that first measured nothing at all: [the journal entry](docs/experiments/2026-08-25_a-refusal-at-last-and-the.md).

### Three ways to run the same agent (and the fourth that was retired)

The flow above started as a hand-rolled loop: our own hop counter, our own dispatch, the coverage gate stitched between the turns. It now runs on a graph, and `orchestrator` selects which implementation executes the same policies. Every one of them fills the same `AgentResult`, so logging, the judge and the metrics cannot tell them apart.

| `orchestrator` | implementation | what it applies |
|---|---|---|
| `langgraph_ported` (default) | `StateGraph`, `app/orchestrators/graph.py` | everything |
| `langgraph_middleware` | `create_agent` plus our policies as middleware hooks, `app/orchestrators/middleware.py` | everything, expressed the way the framework wants it |
| `langgraph_idiomatic` | bare `create_agent`, `app/orchestrators/react.py` | tool admission and the topic axis (they run before the branch point), no coverage gate, no context drop, no fallback notice, no nudge, no final turn without tools |

The loop itself is gone. It was kept only until the port was measured against it: same questions, same settings, and agreement within the bench's own noise on every key the pipeline reads. Keeping a second implementation of behaviour that is already pinned by tests buys nothing and costs a branch in every future change. Its runs are still in the log under `orchestrator=agent`, and the numbers are in [the entry](docs/experiments/2026-08-26_the-same-agent-written-four-ways.md).

![The ported agent graph, generated from the compiled graph](docs/diagrams/agent_graph.svg)

The graph picture is generated from the compiled graph by `scripts/graph_to_d2.py`, and CI fails if the committed drawing no longer matches the code.

`/v1/question-log?orchestrator=langgraph_ported` slices the logs by implementation, the same way `fallback_policy` and `fallback_reason` do. The value `agent` is readable there and nowhere else: runs cannot ask for the retired loop.

The policies themselves live in `app/use_cases/agent_policy.py` as plain functions, and both arms that carry policies call the same ones. What differs is the harness, which is the point: it makes "what does the standard cost" a measurable question rather than an opinion. Two things do not survive the move to the standard tool contract, and both are recorded rather than hidden: error kinds (`timeout`, `auth`, `tool`, ...) collapse into success-or-error, and the bare arm has no final turn, so a question that wants a fifth hop ends without an answer instead of with a refusal.

Two implementation notes worth stealing. Under `corpus_first` the withheld external tools have to be refused at dispatch, not merely hidden from the model: the standard tool node is built once from the full tool list, so narrowing what the model sees does not stop a call the model invents. And the fallback notice rides in the tool result rather than in a system message, for the template reason described above.

## How it is built

- `app/config.py` - `config.yaml` loader.
- `app/orm/` - SQLAlchemy: `base` (declarative), `sync_db` (psycopg), `async_db` (asyncpg).
- `app/models/` - ORM models: `registry` (Model/ModelRole/Prompt), `eval` (Question/QuestionLog), `jobs` (Job), `corpus` (DataSource/DataChunk), `experiment` (Experiment + state machine).
- `app/llm.py` - Ollama client via the OpenAI SDK (generation / embeddings / structured output) + role→model resolver.
- `app/rerank.py` - cross-encoder reranker (sentence-transformers, CPU, lazy-loaded).
- `app/job_queue.py`, `app/worker.py`, `app/job_handlers/` - Postgres queue (FOR UPDATE SKIP LOCKED) and worker with retries/defer; handlers split by theme.
- `app/bootstrap.py` - idempotent startup init.
- `app/sources/` - per-source ingestion (reader pattern: `Base` ABC + sources).
- `app/db.py` - hybrid search (raw SQL: pgvector `<=>`, FTS, ltree, RRF).
- `app/use_cases/` - `chat` (retrieve/answer), `agent` (policies, logging and the snapshot around the agent run), `agent_policy` (the policies themselves as plain functions), `index` (corpus build), `judge` (answer scoring), `experiment` (series aggregator + RRF composite).
- `app/orchestrators/` - adapters to the framework: `graph` (StateGraph), `middleware` (our policies as hooks), `react` (bare `create_agent`). No langchain import reaches `use_cases`.
- `app/agent_tools.py` - tool registry + `dispatch` + the `search_corpus` tool over hybrid retrieval.
- `app/mcp_server.py` - FastMCP server (mounted at `/mcp`): `search_corpus` / `answer_question` / `list_categories` tools reusing the retrieval primitives.
- `app/mcp_ops.py` - ops MCP server (mounted at `/mcp-ops`): `run_metrics` / `compare_runs` / `compare_pools` / `list_jobs` / `cancel_job` over the eval platform.
- `app/evals/pools.py`, `app/evals/compare.py` - one place that decides which pool a question belongs to and what the run's outcome was, shared by the metrics, the comparison report and both MCP tools.
- `app/api/` - REST adapters (health + v1: chat / agent / categories / model / role / source / prompt / eval / experiment / questions / question-log / job).
- `app/seed.py`, `app/console.py` - prompt/question-bank seed; REPL console.
- `app/evals/` - eval bench (runner + retrieval and generation metrics via the judge).
- `tests/` - unit tests (pure logic, no DB/Ollama): `docker compose exec rag-lab pytest -q`.

## Status

A learning project: the goal is to master RAG/LLM engineering by hand, from primitives, and where the industry has a standard, to move onto it and measure what the move costs. RAG from primitives (hybrid, ltree, FTS) + FastAPI server + 5-axis LLM-judge eval + production layer (uv packaging, central config, SQLAlchemy ORM sync+async, OpenAI-compatible client, role-keyed model lifecycle, prompt versioning, async job queue, question bank, reranking, route-driven eval platform, MCP server).
