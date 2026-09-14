# Code map

Where each part of the code lives.

- `app/config.py` - `config.yaml` loader.
- `app/orm/` - SQLAlchemy: `base` (declarative), `sync_db` (psycopg), `async_db` (asyncpg).
- `app/models/` - ORM models: `registry` (Model/ModelRole/Prompt), `eval` (Question/QuestionLog), `jobs` (Job), `corpus` (DataSource/DataChunk), `experiment` (Experiment + state machine), `mcp_integration` (the remote-tool registry).
- `app/llm.py` - one OpenAI-SDK client per engine (generation / embeddings / structured output / pair scoring) + role→model resolver; in the worker every call takes the card for its engine first.
- `app/engines/` - the engine layer: rows and lookups, one driver per kind (`drivers.py`: ollama, vLLM), the card owner (`card.py`: who holds the GPU, read from the servers, and the only road that hands it over), vLLM weights in the host's HF cache.
- `app/rerank.py` - reranking over the `reranking` role: pairs scored by a vLLM pooling server (`vllm-rerank`).
- `app/job_queue.py`, `app/worker.py`, `app/job_handlers/` - Postgres queue (FOR UPDATE SKIP LOCKED) and worker with retries/defer; handlers split by theme.
- `app/job_specs.py` - what each job type accepts, one model per type. Checked when a job is queued, by whichever door or script queues it, and again when the worker takes it; the queue lane belongs to the type rather than to the caller.
- `app/bootstrap.py` - idempotent startup init.
- `app/sources/` - per-source ingestion (reader pattern: `Base` ABC + sources), each source declaring its policy rather than being special-cased downstream.
- `app/ingest.py` - the cutter: headings located by `MarkdownHeaderTextSplitter`, text sliced from the file itself, sections cut by subheading before size, slivers merged, and what decided each boundary recorded on the chunk.
- `app/use_cases/ingest_quality.py` - the coverage report: metrics from the text alone, no embeddings and no labelled questions, gated per source and kept as history on `data_sources`.
- `app/db.py` - hybrid search (raw SQL: pgvector `<=>`, FTS, ltree, RRF).
- `app/use_cases/` - `chat` (retrieve/answer), `agent` (policies, logging and the snapshot around the agent run), `agent_policy` (the policies themselves as plain functions), `index` (corpus build), `judge` (answer scoring), `experiment` (series aggregator + RRF composite), `retrieval_compare` (the grid of arms, paired deltas and halves), `rejudge` (re-scoring answers that already exist), `search_depth` (resolving and caching the index depth), `mcp_integration` (remote tools).
- `app/orchestrators/` - adapters to the framework: `graph` (StateGraph), `react` (bare `create_agent`). No langchain import reaches `use_cases`.
- `app/agent_tools.py` - tool registry + `dispatch` + the `search_corpus` tool over hybrid retrieval.
- `app/mcp_server.py` - FastMCP server (mounted at `/mcp`): `search_corpus` / `answer_question` / `list_categories` tools reusing the retrieval primitives.
- `app/mcp_ops.py` - ops MCP server (mounted at `/mcp-ops`): `run_metrics` / `compare_runs` / `compare_pools` / `judge_correlation` / `question_sets` / `questions` / `experiment_results` / `list_jobs` / `cancel_job` / `holm_over` / `engines` / `language_cost` / `broker_balances` over the eval platform.
- `app/evals/pools.py`, `app/evals/compare.py` - one place that decides which pool a question belongs to and what the run's outcome was, shared by the metrics, the comparison report and both MCP tools; a population is named once in `pools.py` and called by name, never restated where it is used.
- `app/api/` - REST adapters (health: liveness, readiness and the stand; v1: chat / agent / categories / engine / model / role / source / prompt / eval / experiment / questions / question-log / job / mcp_integration; `card_door` turns a wait for the card into a 503 or a 409 at the answering doors).
- `app/seed.py`, `app/console.py` - prompt/question-bank seed; REPL console.
- `app/evals/` - eval bench: the runner, retrieval and generation metrics, the guest axes (`guest_axes`, `guest_llm`, `guest_probes`), the judge-against-judge report (`judge_correlation`), the language probe, the replay (`replay`), what a run still owes (`run_debts`), what a question set holds (`question_sets`), the blind pairs the owner ranks (`human_anchor`), what the axes charge for answering in the language asked (`language_cost`) and where a job leaves its number (`measurements`).
- `tests/` - unit tests (pure logic, no DB/Ollama): `docker compose exec rag-lab pytest -q`.
