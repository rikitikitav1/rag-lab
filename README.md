# rag-lab

A RAG system over a personal knowledge base and external IT repositories. It answers technical questions and cites its sources. Built **from primitives** (no LangChain/LlamaIndex) on **local inference** (Ollama on a single GPU, via the OpenAI-compatible protocol).

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

The first `up` pulls ~16 GB of models and builds the index (~5-10 min, watch `docker compose logs -f worker`). The server starts **without waiting** for models and indexing, so the first requests may refuse until the corpus fills up.

Full hands-on scenarios (mini-eval to numbers, reranking A/B, importing your own questions, browsing logs) and the complete command reference: **[docs/use_cases.md](docs/use_cases.md)**.

## Compose services

| Service | Role |
|---------|------|
| `postgresql` | Postgres + pgvector, the only hard dependency |
| `dbmate` | applies migrations, runs to completion before the rest |
| `seed` | loads prompts and the question bank, runs once after migrations |
| `rag-lab` | FastAPI server (uvicorn) + bootstrap on startup |
| `worker` | processes the job queue (pull/delete/index/embed/paraphrase/eval/judge) |
| `ollama` | local inference on GPU |

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

Model lifecycle:
- `GET /v1/model`, `GET /v1/model/{id}`, `POST /v1/model` (create enqueues a pull), `DELETE /v1/model/{id}` (409 if assigned to a role)
- `GET /v1/role`, `PUT /v1/role/{role}` (assign a model to a role)
- `GET /v1/source`, `PUT /v1/source/{id}` (enable/disable a corpus source; disabled sources are excluded from retrieval at runtime, no re-index — ablation / source-of-truth scoping)

Prompts:
- `GET /v1/prompt`, `GET /v1/prompt/{id}`, `POST /v1/prompt`, `POST /v1/prompt/{id}/activate`, `DELETE /v1/prompt/{id}`

Eval platform:
- `POST /v1/eval/paraphrase` (generate a paraphrase set), `POST /v1/eval/run` (run a set → judge; `pipeline: single_shot|agent`, optional `rerank`; 400 if `rerank` is combined with `pipeline: agent`)
- `GET /v1/eval/misses?run_name=X` (retrieval misses for a run: in-corpus questions where the expected source was not retrieved, with expected vs retrieved)
- `POST /v1/questions/import` (upload a questions file, ≤5 MB; optional chained run)

Observability:
- `GET /v1/question-log`, `GET /v1/question-log/{id}` (answer logs; filters incl. `pipeline`, `faithfulness`/`relevance`/`completeness`, `run_name`; detail with context)
- `GET /v1/job`, `GET /v1/job/{id}` (jobs + elapsed)

## How it is built

- `app/config.py` - `config.yaml` loader.
- `app/orm/` - SQLAlchemy: `base` (declarative), `sync_db` (psycopg), `async_db` (asyncpg).
- `app/models/` - ORM models: `registry` (Model/ModelRole/Prompt), `eval` (Question/QuestionLog), `jobs` (Job), `corpus` (DataSource/DataChunk).
- `app/llm.py` - Ollama client via the OpenAI SDK (generation / embeddings / structured output) + role→model resolver.
- `app/rerank.py` - cross-encoder reranker (FlagEmbedding, CPU, lazy-loaded).
- `app/job_queue.py`, `app/worker.py`, `app/job_handlers/` - Postgres queue (FOR UPDATE SKIP LOCKED) and worker with retries/defer; handlers split by theme.
- `app/bootstrap.py` - idempotent startup init.
- `app/sources/` - per-source ingestion (reader pattern: `Base` ABC + sources).
- `app/db.py` - hybrid search (raw SQL: pgvector `<=>`, FTS, ltree, RRF).
- `app/use_cases/` - `chat` (retrieve/answer), `agent` (ReAct tool-calling loop), `index` (corpus build), `judge` (answer scoring).
- `app/agent_tools.py` - tool registry + `dispatch` + the `search_corpus` tool over hybrid retrieval.
- `app/api/` - REST adapters (health + v1: chat / agent / categories / model / role / source / prompt / eval / questions / question-log / job).
- `app/seed.py`, `app/console.py` - prompt/question-bank seed; REPL console.
- `app/evals/` - eval bench (runner + retrieval and generation metrics via the judge).
- `tests/` - unit tests (pure logic, no DB/Ollama): `docker compose exec rag-lab pytest -q`.

## Status

A learning project: the goal is to master RAG/LLM engineering by hand, from primitives. RAG from primitives (hybrid, ltree, FTS) + FastAPI server + 4-axis LLM-judge eval + production layer (uv packaging, central config, SQLAlchemy ORM sync+async, OpenAI-compatible client, role-keyed model lifecycle, prompt versioning, async job queue, question bank, reranking, route-driven eval platform).
