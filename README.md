# rag-lab

[![CI](https://github.com/rikitikitav1/rag-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rikitikitav1/rag-lab/actions/workflows/ci.yml)

A RAG system over a personal knowledge base and external IT repositories. It answers technical questions and returns the sources it retrieved. Built **from primitives** (no LangChain/LlamaIndex) on **local inference** (Ollama on a single GPU, via the OpenAI-compatible protocol).

This is a **showcase lab**: a bench for practicing LLM/RAG engineering approaches by hand (retrieval, LLM-as-judge eval, model lifecycle, reranking, async queues) and showing results as numbers. Not a production service, a playground for approaches.

> Русская версия: [docs/README_ru.md](docs/README_ru.md) · Hands-on scenarios and commands: [docs/use_cases.md](docs/use_cases.md) · Experiments log: [docs/experiments.md](docs/experiments.md)

## What it is

- **Hybrid retrieval**: vector search (pgvector) + full-text (Postgres FTS with per-language stemming), fused via **RRF** (Reciprocal Rank Fusion). Filter by hierarchical categories (ltree), a distance threshold with an honest refusal on out-of-corpus questions.
- **Local models** through Ollama, keyed by role: generation, embeddings (`bge-m3`), LLM judge. A role is decoupled from a concrete model: which model serves a role lives in the DB and can be switched at runtime.
- **Multi-source**, one taxonomy: personal notes (ru) + Devinterview-io interview repos (en, ~170 repos). Per-source ingestion strategies.
- **Eval bench, 5 quality axes**: retrieval (hit@k / MRR), faithfulness (is the answer grounded in the context), relevance (does it answer to the point), completeness (does it cover the reference answer), refusal accuracy, all via **LLM-as-judge** with structured output.
- **Async job queue**: heavy operations (model pull/delete, corpus indexing, embedding the question bank) go to a Postgres-backed queue processed by a worker. The service depends only on Postgres: Ollama may be unavailable at startup, jobs defer/retry, the app does not crash.
- **Reranking (opt-in)**: a cross-encoder (`bge-reranker-v2-m3`) on top of hybrid retrieval (retrieve-wide → rerank → narrow), toggled by a flag (per-request / per-run); A/B tested on a cross-lingual set.
- **Agent (ReAct, from primitives)**: a hand-rolled tool-calling loop where the model decides when to search the corpus, may refine the query and multi-hop, then answers. Selectable as an eval pipeline (`pipeline: agent`) and benchmarked head-to-head against single-shot RAG (see the experiments log).
- **Route-driven eval platform**: generating non-circular eval sets (LLM paraphrase of interview questions + translation to ru), importing questions from a file, runs and judging, all bulk through the queue; observability via the request log (`question-log`) and jobs with `elapsed`.
- **Layering**: transport-neutral `use_cases` → thin adapters (CLI / FastAPI REST).

## Stack

Python · PostgreSQL + pgvector · SQLAlchemy 2.0 (sync psycopg + async asyncpg) · Ollama (GPU, OpenAI-compatible API) · FastAPI · dbmate (migrations) · uv/pyproject · Docker Compose.

## Models and prompts architecture

- **Model / ModelRole**: `Model` (name + status: available/loading/ready), `ModelRole` (role as PK → one model per role by construction, FK `ON DELETE RESTRICT` so the DB refuses to delete an assigned model). The resolver `llm.resolve_name(role)` reads the name from the DB, inference params come from `config.yaml`.
- **Prompt**: versioned in the DB (`purpose` + `version`, exactly one `active` per purpose). Prompt sources are files in `prompts/` (format `<purpose>.v<N>.txt`), the seed loads them into the DB, the freshest version becomes active.
- **Bootstrap on startup** (idempotent): ensure Model rows from config → seed roles → reconcile with Ollama (not pulled → status `loading` + a pull job) → enqueue indexing if the corpus is empty → enqueue embedding of questions without a vector.

## Configuration

Everything tunable lives in **`config.yaml`** (mounted into the container):

- `llm.roles` - model + `options` per role (`generation` / `embedding` / `judging` / `paraphrasing`); `llm.candidates` - models to pull but not assign.
- `service.retrieval` - `distance_threshold`, `results_limit`, `rrf_k`, candidate limits.
- `service.rerank` - `enabled`, `model`, `candidates`, `top`.
- `service.agent` - `max_hops` (ReAct hop cap).
- `service.ingestion` - `chunk_max_size`, `batch_size`, `commit_size`.
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

Everything tunable about the pipeline lives in `config.yaml`; the environment only carries what depends on the machine or must stay out of the repo. Copy `.env.example` to `.env` — compose picks it up, and every value has a working default.

| Variable | Default | What it does |
|----------|---------|--------------|
| `RERANK_DEVICE` | `cuda` (worker), `cpu` (API) | Where the cross-encoder runs. The asymmetry is deliberate: eval runs rerank in a phase that owns the card, while interactive answers share it with ollama, and a reranker resident there would evict the generator on every question. `auto` picks cuda when a card is visible; CUDA OOM falls back to CPU with a warning. |
| `LLM_TIMEOUT` | `120` | Seconds per completion. A 70b model on CPU needs minutes; the default kills such runs mid-flight. |
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
- `POST /v1/agent/question` (ReAct agent answer; optional `max_hops`, `language`, and `debug` for the full message trace)
- `GET /v1/categories` (category tree with chunk counts)

![Single-shot flow: hybrid retrieval, threshold, optional rerank](docs/diagrams/single_shot_flow.svg)

Model lifecycle:
- `GET /v1/model`, `GET /v1/model/{id}`, `POST /v1/model` (create enqueues a pull), `DELETE /v1/model/{id}` (409 if assigned to a role)
- `GET /v1/role`, `PUT /v1/role/{role}` (assign a model to a role)
- `GET /v1/source`, `PUT /v1/source/{id}` (enable/disable a corpus source; disabled sources are excluded from retrieval at runtime, no re-index - ablation / source-of-truth scoping)

Prompts:
- `GET /v1/prompt`, `GET /v1/prompt/{id}`, `POST /v1/prompt`, `POST /v1/prompt/{id}/activate`, `DELETE /v1/prompt/{id}`

Eval platform:
- `POST /v1/eval/paraphrase` (generate a paraphrase set), `POST /v1/eval/run` (run a set → judge; `pipeline: single_shot|agent`, per-run `rerank`, `k` retrieval-width, `max_hops` and `model` (generator) overrides; config only sets the defaults)
- `POST /v1/eval/experiment` (batch a parameter series: `param` (`k`, `max_hops` or `model`) swept over `values`, one auto-named run per value, each judged; set/pipeline/language stay fixed for a clean single-variable comparison; a `model` value absent from the registry is created and pulled, the run waits for it)
- `GET /v1/eval/misses?run_name=X` (retrieval misses for a run: in-corpus questions where the expected source was not retrieved, with expected vs retrieved)
- `POST /v1/questions/import` (upload a questions file, ≤5 MB; optional chained run)

![Eval pipeline](docs/diagrams/eval_pipeline.svg)

A single run does not loop per question: it goes through phases so each stage owns the GPU alone, which is what makes reranking affordable in bulk.

![Phases inside one eval run](docs/diagrams/phased_run.svg)

Measured on 100 questions with reranking on: **2092s → 652s (3.2x)** while `hit@5` and `MRR` stayed identical to the third decimal. Most of the win did not come from batching — it came from noticing that the card was never actually released between phases, so ollama had been loading the generator as 26 layers of 33. The full story, including what the batch alone did *not* buy, is in [the journal entry](docs/experiments/2026-08-24_phased-eval-runs-and-the-empty-cache.md).

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

Record the takeaway with `PUT /v1/experiment/{id}/conclusion` and the experiment becomes a self-contained artifact: what was varied, on what data, the numbers, the verdict.

Observability:
- `GET /v1/question-log`, `GET /v1/question-log/{id}` (answer logs; filters incl. `pipeline`, `faithfulness`/`relevance`/`completeness`, `run_name`; detail with context)
- `GET /v1/job`, `GET /v1/job/{id}` (jobs + elapsed), `POST /v1/job/{id}/cancel` (cancels the job and its dependent judge, cooperative stop for a running eval)

## MCP

An MCP (Model Context Protocol) server is mounted at `/mcp` (streamable HTTP), exposing the corpus to any MCP client (Claude Desktop, Cursor, IDE agents). Built on standalone `fastmcp` and reusing the same retrieval primitives as the REST/agent paths. Tools:
- `search_corpus(query, category?)` - hybrid retrieval, returns chunks with `[source]` markers; optional category subtree filter.
- `answer_question(text, pipeline?, category?, language?)` - full RAG answer, returns `{answer, retrieved, sources}` (`agent` or `single_shot`; `category` only with `single_shot`).
- `list_categories(category?, only_top?)` - category paths with chunk counts, for discovering valid filter values before searching.

Connect: `claude mcp add --transport http rag-lab http://127.0.0.1:8000/mcp/`, or point the MCP Inspector at the same URL.

A second, separate ops server is mounted at `/mcp-ops` - an eval control plane kept off the product surface (an external client gets search/answer tools, not admin verbs):
- `run_metrics(run_name)` - aggregated eval metrics for one run (generation axes + retrieval hit@k/MRR).
- `compare_runs(run_names)` - side-by-side metrics with an RRF composite ranking (generation axes only).
- `list_jobs(status?, type?, run_name?)` / `cancel_job(id)` - job queue control, cancel takes the dependent judge down with the run.

### MCP client: the agent consumes external servers

The lab is both sides of the protocol: its own MCP server above, and an MCP *client* below. External hosted MCP servers are registered as `McpIntegration` rows and their tools join the agent's toolbox next to `search_corpus`, namespaced `integration__tool` (e.g. `deepwiki__ask_question`). The agent decides per hop whether to look outside the corpus; a successful remote call is recorded as an `mcp:` source (provenance), a failed one degrades to an error string the agent can route around.

![Agent flow with remote fallback](docs/diagrams/agent_flow.svg)

Registry lifecycle via `/v1/mcp_integration`:
- CRUD with filters; new integrations start `disabled`, a state machine (`disabled/active/unreachable`) separates operator intent from observed health (probes flip `active <-> unreachable`, never touch `disabled`).
- `POST /{id}/discover` - fetch the server's tool list, cache name/description/schema snapshots in the DB. The agent builds tools from this frozen cache (no network on run start, and a later description swap on the server side does not silently reach the LLM prompt - discover again to refresh).
- `POST /{id}/probe` - live ping writing `last_checked_at/last_error`; `GET /{id}/health` - cheap read of the stored state. A `check_mcp_health` job fires on every create/update (io queue lane, so it never waits behind GPU jobs).
- `allowed_tools` is an explicit allowlist: discovery shows the catalog, a human picks what the 8B model actually sees. Tool descriptions are truncated on cache; results are truncated to `max_result_chars`.

![McpIntegration state machine](docs/diagrams/mcp_state.svg)

Auth per integration is declared as `{"type": "bearer", "token_env": "HF_TOKEN"}` or `{"type": "header", "header": "...", "value_env": "..."}` - the DB stores only environment variable *names*; values come from the environment and only for variables allowlisted in `config.yaml` (`mcp_integrations.secret_env`).

Note on trust boundaries: no auth by design; anyone with API access can register an integration pointing anywhere, and allowlisted secrets will be sent to that URL. Do not expose the service.

Seeded integrations (all disabled until you enable them): DeepWiki (no auth), Hugging Face (`HF_TOKEN`), Context7 (`CONTEXT7_API_KEY`).

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
- `app/use_cases/` - `chat` (retrieve/answer), `agent` (ReAct tool-calling loop), `index` (corpus build), `judge` (answer scoring), `experiment` (series aggregator + RRF composite).
- `app/agent_tools.py` - tool registry + `dispatch` + the `search_corpus` tool over hybrid retrieval.
- `app/mcp_server.py` - FastMCP server (mounted at `/mcp`): `search_corpus` / `answer_question` / `list_categories` tools reusing the retrieval primitives.
- `app/mcp_ops.py` - ops MCP server (mounted at `/mcp-ops`): `run_metrics` / `compare_runs` / `list_jobs` / `cancel_job` over the eval platform.
- `app/api/` - REST adapters (health + v1: chat / agent / categories / model / role / source / prompt / eval / experiment / questions / question-log / job).
- `app/seed.py`, `app/console.py` - prompt/question-bank seed; REPL console.
- `app/evals/` - eval bench (runner + retrieval and generation metrics via the judge).
- `tests/` - unit tests (pure logic, no DB/Ollama): `docker compose exec rag-lab pytest -q`.

## Status

A learning project: the goal is to master RAG/LLM engineering by hand, from primitives. RAG from primitives (hybrid, ltree, FTS) + FastAPI server + 5-axis LLM-judge eval + production layer (uv packaging, central config, SQLAlchemy ORM sync+async, OpenAI-compatible client, role-keyed model lifecycle, prompt versioning, async job queue, question bank, reranking, route-driven eval platform, MCP server).
