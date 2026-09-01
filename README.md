# rag-lab

[![CI](https://github.com/rikitikitav1/rag-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rikitikitav1/rag-lab/actions/workflows/ci.yml)

A RAG system over a personal knowledge base and external IT repositories. It answers technical questions and returns the sources it retrieved. Retrieval is built **from primitives**: the hybrid returns the rank each leg gave, and the weak-retrieval gate is built on those ranks, which a library retriever hands back as a list of documents. Where the standard is simply better it is used rather than reimplemented: the agent runs on **LangGraph**, and markdown is parsed by `MarkdownHeaderTextSplitter`, which knows a heading from a comment inside a fenced block (ours did not). The agent started hand-rolled and the port was measured against the original before the original was retired (see [the entry](docs/experiments/2026-08-26_the-same-agent-written-four-ways.md)). Inference is **local** (Ollama on a single GPU).

This is a **showcase lab**: a bench for practicing LLM/RAG engineering approaches by hand (retrieval, LLM-as-judge eval, model lifecycle, reranking, async queues) and showing results as numbers. Not a production service, a playground for approaches.

> In Russian: [docs/README_ru.md](docs/README_ru.md) · Hands-on scenarios: [docs/use_cases.md](docs/use_cases.md) · Experiments log: [docs/experiments.md](docs/experiments.md) · What the preflight refuses: [docs/preflight.md](docs/preflight.md) · What each role requires of a model: [docs/model_requirements.md](docs/model_requirements.md)

## What it is

- **Corpus variants**: the same corpus can hold several chunkings side by side, each with its own partial vector index and its own policy in the run snapshot. Re-indexing stops being a one-way border and becomes a swept parameter (see [the entry](docs/experiments/2026-08-26_a-corpus-you-can-keep-two-of.md)).
- **Hybrid retrieval**: vector search (pgvector) + full-text (Postgres FTS with per-language stemming), fused via **RRF** (Reciprocal Rank Fusion). Filter by hierarchical categories (ltree), a distance threshold with an honest refusal on out-of-corpus questions.
- **Local models** through Ollama, keyed by role: generation, embeddings (`bge-m3`), LLM judge. A role is decoupled from a concrete model: which model serves a role lives in the DB and can be switched at runtime.
- **Multi-source**, one taxonomy: personal notes (ru), 173 Devinterview-io interview repos (en), and three documentation sources with rules of their own: `system-design-primer`, `redis-doc` (frontmatter titles) and `cheatsheets` (frontmatter plus per-file junk rules). Per-source ingestion strategies, declared as a policy rather than special-cased in the cutter.
- **Eval bench, 5 quality axes**: retrieval (hit@k / MRR), faithfulness, relevance, completeness, refusal accuracy, all via **LLM-as-judge** with structured output. Scores are reported per pool (grounded in the corpus, grounded in an external tool, refusals) rather than as one average, and a run records an outcome per question (`answered`, `refused`, `unsupported_answer`, `narrated_call`, `exhausted`, `error`), because counting an unsupported answer as a refusal flatters exactly the policy under test.
- **Async job queue**: heavy operations (model pull/delete, corpus indexing, embedding the question bank) go to a Postgres-backed queue processed by a worker. The service depends only on Postgres: Ollama may be unavailable at startup, jobs defer/retry, the app does not crash.
- **Reranking, off by default since 30.08 and measured both ways**: a cross-encoder (`bge-reranker-v2-m3`) on top of hybrid retrieval (retrieve-wide → rerank → narrow). It was the default until the agent path turned out to need a generator with tool calling: the 8b that has it takes the card room the reranker held, so a request or a run now turns reranking **on**. Its measured worth is unchanged; the default is a card decision, not a verdict on the method. At the level of the ranking the evidence disagrees with itself: on 823 cross-lingual questions it is worth **+0.0369 section MRR** [0.0155, 0.0596], and on the 820 same-language ones **−0.0245** [−0.0460, −0.0021], both clear of zero. At the level of the answer it costs nothing for the generator we serve: `gemma3:4b` with reranking against without gives relevance **+0.324** [0.111, 0.530] and completeness **+0.277** [0.111, 0.433] on the Russian set and three intervals through zero on the English one. Cost on the card: 168 ms a question in a phased run, against 2.76 s on cpu.
- **Agent on LangGraph**: a tool-calling graph where the model decides when to search the corpus, may refine the query and multi-hop, then answers. It started as a hand-rolled loop, which was retired only after the port agreed with it within the bench's own noise. The policies around it (coverage gate, topic axis, tool admission) stayed ours and are measured, not assumed. Selectable as an eval pipeline (`pipeline: agent`) and benchmarked head-to-head against single-shot RAG.
- **Route-driven eval platform**: generating non-circular eval sets (LLM paraphrase of interview questions + translation to ru), importing questions from a file, runs and judging, all bulk through the queue; observability via the request log (`question-log`) and jobs with `elapsed`.
- **Layering**: transport-neutral `use_cases` → thin adapters (CLI / FastAPI REST).

## Stack

Python · PostgreSQL + pgvector · SQLAlchemy 2.0 (sync psycopg + async asyncpg) · Ollama (GPU, OpenAI-compatible API) · FastAPI · dbmate (migrations) · uv/pyproject · Docker Compose.

From the LangChain family, five packages: **langgraph** runs the agent as a `StateGraph`, **langchain** provides `create_agent` for the arm that answers "what does the idiomatic version cost", **langchain-text-splitters** decides what counts as a markdown heading, and **langchain-core** with **langchain-ollama** are what the idiomatic arm imports directly. Retrieval, the queue, the eval bench and the corpus policies are ours; the version of the splitter travels in the ingest report, because the cut is external code now and a lock refresh would otherwise move it silently.

## Models and prompts architecture

- **Model / ModelRole**: `Model` (name + status: available/loading/ready), `ModelRole` (role as PK → one model per role by construction, FK `ON DELETE RESTRICT` so the DB refuses to delete an assigned model). `Role` is a closed set of four: `generation`, `embedding`, `judging`, `paraphrasing`. The resolver `llm.resolve_name(role)` reads the name from the DB and inference params from `config.yaml`, so the file declares a role's model and the database serves it. `llm.candidates` names models to pull without assigning them.
- **A model name is checked the same way at both doors**: `MODEL_NAME_RE` for the shape, 128 characters for the length, and a three-segment name may only point at `hf.co` or `registry.ollama.ai`. The HTTP door and the job that registers a model by name call one function, because a name reaching the pull through the second door used to skip the first door's rules.
- **Purpose**: eleven of them, and a prompt belongs to exactly one. One for answering (`generate.answer`), three for judging, three for building question sets (`paraphrase.question`, `translate.question`, `question.from_heading`), four for the agent (`agent.system`, `agent.fallback`, `agent.tool_match`, `agent.no_evidence`).
- **Pipeline**: `single_shot` or `agent`, recorded per answer in `question_logs.pipeline`. The two paths ask different things of the same generation role, which is why a model can be fine for one and unusable for the other.
- **Prompt**: versioned in the DB (`purpose` + `version`, exactly one `active` per purpose). Sources are files in `prompts/`, named by the enum member rather than the purpose string (`agent_fallback.v1.txt` carries the purpose `agent.fallback`). The seed loads them in, and a new version becomes active only when the purpose has no active prompt yet; otherwise it lands inactive with a warning, and `POST /v1/prompt/{id}/activate` is the deliberate step that switches it.
- **Bench**: what one judging pass actually used, as an object rather than an assumption: a model and a pinned prompt version per axis. The live path uses the active versions; a rejudge arm carries its own, which is how two arms can score the same answers differently on purpose.
- **What a run records about its own instruments**: `question_logs.models` and `question_logs.prompts`, filled by the answering path and extended by the judge with its model and `judge_*` versions. The arm says what was asked for, these two columns say what ran, and on a rejudge that difference is the whole measurement.
- **The reranker is not among the roles**: the cross-encoder is not in ollama and not in `llm.roles`, it is loaded in-process from `service.rerank`, and residency checks that ask ollama cannot see it.
- **What a role requires of a model** (tool calling for the agent's generator, no reasoning-only judge, the card arithmetic, and why the role's name is served from the database rather than the file): [docs/model_requirements.md](docs/model_requirements.md).
- **Bootstrap** (a one-shot compose service, idempotent): ensure Model rows from config → seed roles → reconcile with Ollama (not pulled → status `loading` + a pull job) → enqueue indexing only if the database holds no corpus variant at all → build a partial vector index for every variant that has rows and none → repair the served variant's index if it is missing → enqueue embedding of questions without a vector, unless such a job is already queued. A named but empty variant is logged, not indexed: indexing a variant is a deliberate, measured step.

## Configuration

Everything tunable lives in **`config.yaml`** (mounted into the container):

- `llm.roles` - model + `options` per role (`generation` / `embedding` / `judging` / `paraphrasing`); the name here is what bootstrap assigns on an empty database, and the database serves it afterwards. `llm.candidates` - models to pull but not assign. `llm.context_length` - what we expect `OLLAMA_CONTEXT_LENGTH` to be, which the preflight compares against the server. `llm.base_url` - where ollama listens.
- `service.retrieval` - `distance_threshold`, `results_limit`, `rrf_k`, candidate limits, and the keyword-leg switches `keyword_query` (`and` joins every lexeme, `or` fires on any), `keyword_rank`, `keyword_norm`, `query_lang` (`function_words` by default, also `langdetect` and `cyrillic_ratio`), and the depth of the vector index: `ef_search` (pinned to 100, see below) plus the ladder it is chosen from and the gates it is judged by (`ef_ladder`, `recall_gate`, `max_mrr_loss`, `max_questions_lost`) and the preflight smoke test that only says the index is alive (`index_alive_recall`, `index_alive_questions`). `criterion_sets` and `veto_sets` name the sets a verdict is read on and the set that can only veto. Their values were picked by measurement and are recorded in every run's snapshot.
- `service.rerank` - `enabled` (off since 30.08, see above; a request or a run can turn it on), `model`, `candidates`.
- `service.agent` - `max_hops` (agent hop cap), `fallback_policy` (`corpus_first` / `corpus_first_weak` / `agent_choice`), `gate_signal` (what calls retrieval weak: `distance`, `cross_encoder` or `either`) with its thresholds `weak_distance` and `weak_threshold`, `gate_candidates` (how many hits the cross-encoder scores), `topic_threshold` (the topic axis: distance to the nearest chunk above which the run refuses instead of reaching out; one number or one per language, `0` in a run switches it off).
- `service.ingestion` - `batch_size`, `commit_size`, and `chunk_max_size` as the fallback ceiling for a source with no variant policy (the cut reads the variant's `max_chunk_size`).
- `service.fts` - the text-search config per language and the fallback. This is index-time: changing it re-indexes the corpus rather than re-ranking a query.
- `repos_dir`, `prompts_dir` - where cloned sources and prompt files are read from.
- `service.ingest_quality` - what makes a source broken rather than merely dirty: `hard_gates` and `soft_gates` in shares of a source's chunks, `weights` for the 0-100 score, how much history to keep per variant, and `score_formula`, which is written into every report so a score from another formula is not read as this one. Thresholds are set from what the corpus measures, and each one carries what it reads today beside it.
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

The first `up` pulls ~16 GB of models and builds the index (~5-10 min, watch `docker compose logs -f worker`). The server waits for `bootstrap` to finish and for ollama to report healthy, which is most of that time; it does **not** wait for the pulls and the indexing those steps queue, so the first requests may refuse until the corpus fills up.

Hands-on scenarios (mini-eval to numbers, reranking A/B, importing your own questions, browsing logs): **[docs/use_cases.md](docs/use_cases.md)**. It is a walkthrough of nine scenarios, not a route index; the complete reference is Swagger at `/docs`, which is generated from the code and cannot fall behind it.

## Architecture

<details>
<summary>Diagram: Architecture</summary>

![Architecture](docs/diagrams/architecture.svg)

</details>

Diagrams are D2 sources in `docs/diagrams/`, rendered by `scripts/render_diagrams.sh`; CI fails if a committed SVG drifts from its source.

## Compose services

| Service | Role |
|---------|------|
| `postgresql` | Postgres + pgvector, the only hard dependency |
| `dbmate` | applies migrations, runs to completion before the rest |
| `seed` | loads prompts and the question bank, runs once after migrations |
| `bootstrap` | prepares models, roles and indexing jobs, runs to completion before the rest |
| `rag-lab` | FastAPI server (uvicorn) |
| `worker` | processes the job queue, twelve types: pull/delete a model, index the corpus, build a vector index, analyze a source, embed questions, paraphrase questions, build the veto set, eval run, judge answers, compare retrieval, mcp health |
| `ollama` | local inference on GPU |

### Environment knobs

Everything tunable about the pipeline lives in `config.yaml`; the environment only carries what depends on the machine or must stay out of the repo. Copy `.env.example` to `.env`: compose picks it up, and every value has a working default.

| Variable | Default | What it does |
|----------|---------|--------------|
| `RERANK_DEVICE` | `cuda` | Where the cross-encoder runs for the API. Compose sets it there and gives the worker `WORKER_RERANK_DEVICE` instead, so setting this one in `.env` changes nothing; it is listed because the code reads it. The card is why the default moved: `bge-m3` at 851 MiB, `bge-reranker-v2-m3` at 1728 MiB measured and `gemma3:4b` at 4248 MiB fit on an 8188 MiB card, but `llama3.1:8b` in place of the 4b does not, and the agent path needs it because gemma3 has no tool calling at all. So reranking is off by default and asked for per run. `auto` picks cuda when a card is visible; CUDA OOM falls back to CPU with a warning. |
| `LLM_TIMEOUT` | `120` | Seconds per completion. A 70b model on CPU needs minutes; the default kills such runs mid-flight. |
| `WORKER_RERANK_DEVICE` | `cuda` | Where the worker runs the cross-encoder. A phased eval run reranks in a batch that owns the card, and the corpus-first gate scores a handful of pairs per question. Set it to `cpu` only when something else needs the whole card. |
| `OLLAMA_CONTEXT_LENGTH` | `8192` | Context window the server loads models with. Ollama defaults to 4096 and silently drops whole messages to fit, so an over-long prompt never shows up as a number above the window; the longest hop logged here is 4075 tokens. Raising it costs VRAM the embedder shares, and at 14336 the two evict each other on every switch. After a change, check `llm.model_spilled_to_cpu`; the run snapshot records the window the server reports, not this value. |
| `WORKER_QUEUES` | `default,io` | Queue lanes the worker serves, one thread each. Network and disk jobs (model pulls, MCP health) live on `io` so they never wait behind GPU work. |
| `JUDGE_WIDTH` | `1` | Rows the judge scores in flight. Over 1 it is a different instrument, not a faster one: the same rows come back with different scores, so a measured arm stays at 1 and only smokes are widened. Capped by the slots the server has and by the connection pool. |
| `OLLAMA_NUM_PARALLEL` | `1` | Sequences ollama batches at once. Pinned rather than left unset: unset, the server picks slots from free VRAM, so two arms run on different instruments. |
| `HF_TOKEN`, `CONTEXT7_API_KEY` | empty | Secrets for external MCP integrations. Only variables allowlisted in `config.yaml` (`mcp_integrations.secret_env`) are ever read. |

## REST API

Full interactive reference in Swagger at `/docs`.

List endpoints (`/v1/model`, `/v1/prompt`, `/v1/job`, `/v1/question-log`) share pagination: `limit` (default 100, max 1000), `offset`, `sort_by`, `sort_order` (`asc`/`desc`, default `desc`).

Health:
- `GET /liveness`, `GET /readiness`
- `GET /v1/health/stand` (what the stand is right now: the card, which models are resident and how much VRAM each holds, the window the server actually serves against the declared one, the live queue, role drift between config and database, the corpus variant and the search depth per variant. Readable while a run competes with it, so a run that answers slowly can be diagnosed without stopping it)

Chat and search:
- `POST /v1/chat/question` (full RAG answer; optional `rerank` flag; optional `language` override `ru`/`en`)
- `POST /v1/chat/fast_question` (retrieval only, no generation)
- `POST /v1/agent/question` (agent answer; optional `max_hops`, `language`, `fallback_policy`, and `debug` for the full message trace)
- `GET /v1/categories` (category tree with chunk counts)

<details>
<summary>Diagram: Single-shot flow: hybrid retrieval, threshold, optional rerank</summary>

![Single-shot flow: hybrid retrieval, threshold, optional rerank](docs/diagrams/single_shot_flow.svg)

</details>

Model lifecycle:
- `GET /v1/model`, `GET /v1/model/{id}`, `POST /v1/model` (create enqueues a pull), `POST /v1/model/{id}/load` (make it resident before a timed run), `DELETE /v1/model/{id}` (409 if assigned to a role)
- `GET /v1/role`, `PUT /v1/role/{role}` (assign a model to a role; the model is asked whether it can do the job first, and a 400 says what it lacks. `anyway: true` insists, which is how a model the server describes wrongly is still seated)
- `GET /v1/source`, `PUT /v1/source/{id}` (enable/disable a corpus source; disabled sources are excluded from retrieval at runtime, no re-index - ablation / source-of-truth scoping)

Prompts:
- `GET /v1/prompt`, `GET /v1/prompt/{id}`, `POST /v1/prompt`, `POST /v1/prompt/{id}/activate`, `DELETE /v1/prompt/{id}`

Eval platform:
- `POST /v1/eval/paraphrase` (generate a paraphrase set), `POST /v1/eval/run` (run a set → judge; `pipeline: single_shot|agent`, per-run `rerank`, `k` retrieval-width, `max_hops`, `fallback_policy`, `gate_signal`, `weak_distance`, `topic_threshold`, `orchestrator`, `variant` (which cut of the corpus the run reads) and `model` (generator) overrides, plus `allow_cpu` for a run that means to measure the processor; config only sets the defaults)
- `POST /v1/eval/experiment` (batch a parameter series: `param` (`k`, `max_hops`, `model`, `variant`, `orchestrator`, `fallback_policy`, `gate_signal`, `weak_distance` or `topic_threshold`) swept over `values`, one auto-named run per value, each judged; set/pipeline/language stay fixed for a clean single-variable comparison; a `model` value absent from the registry is created and pulled, the run waits for it)
- `GET /v1/eval/misses?run_name=X` (retrieval misses for a run: in-corpus questions where the expected source was not retrieved, with expected vs retrieved)
- `GET /v1/eval/compare?runs=A&runs=B` (arms side by side split by pool: in-corpus, out-of-corpus, off-domain, rejected; per arm the judged axes, how often the answer came from a remote tool against the corpus, how often the coverage gate fired, latency avg/p50 and the outcome histogram; per pair of arms a paired Wilcoxon plus a bootstrap interval over the same questions, so a difference is reported with its size and its uncertainty instead of two averages)
- `POST /v1/questions/import` (upload a questions file, ≤5 MB; optional chained run)

<details>
<summary>Diagram: Eval pipeline</summary>

![Eval pipeline](docs/diagrams/eval_pipeline.svg)

</details>

A single run does not loop per question: it goes through phases so each stage owns the GPU alone, which is what makes reranking affordable in bulk. Per question the loop needed the embedder, then the reranker, then the generator, and the three do not fit in 8 GB together, so ollama evicted and reloaded a model on every single question. Phases cost a few model swaps per run instead of two per question, and the run gives the card back when it ends, so a queue of runs on different generators does not end up holding two of them at once. The reranker finally gets a real batch: 31 s for 100 questions on the card against about 16 min on CPU. That 310 ms a question is the whole phase, model load and retrieval included; the reranking itself is the 86 ms measured in `datasets/measurements/rerank_latency.json`.

<details>
<summary>Diagram: Phases inside one eval run</summary>

![Phases inside one eval run](docs/diagrams/phased_run.svg)

</details>

Measured on 100 questions with reranking on: **2092s → 652s (3.2x)** while `hit@5` and `MRR` stayed identical to the third decimal. Most of the win did not come from batching, it came from noticing that the card was never actually released between phases, so ollama had been loading the generator as 26 layers of 33. The full story, including what the batch alone did *not* buy, is in [the journal entry](docs/experiments/2026-08-24_phased-eval-runs-and-the-empty-cache.md).

Experiments (first-class entity over the raw sweep route):
- `POST /v1/experiment` (creates the experiment - dataset + deterministic seed-based sample / procedure snapshot / varied param - and enqueues the run series), `GET /v1/experiment` (filtered list), `GET /v1/experiment/{id}`, `PUT /v1/experiment/{id}/conclusion`, `POST /v1/experiment/{id}/arms` (a rejudge only: copies more arms of the same answers and re-judges them, from `aggregated` and back to `running`; the arms are named one by one rather than as a grid, the row cap counts what the experiment already holds, and an arm added later is built on the sample, the control size and the seed the experiment was created with)
- state machine `draft → running → aggregated → concluded` (+ `failed`); when the last judge job of the series finishes, the aggregator computes per-value metrics and an RRF composite over five axes (the three judged ones, the off-domain refusal rate and the supported rate) and stores them in `results` (retrieval hit@k/MRR reported per value but kept out of the fusion: hit@k is monotonic in `k`, it would confound the composite)
- results carry **paired significance statistics**, not just point estimates: for the winner vs every other value, per axis - mean paired delta (same question in both runs), bootstrap 95% CI and a Wilcoxon signed-rank p-value, plus Holm step-down flags over the test family the record itself names, so the JSON says what survives multiple-comparison correction and against which family it was corrected

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

# "Is the judge reading the answer, or itself?" - a rejudge, where the answers are held
# still and only the judge moves. Arms are copies of one recorded run, so the corpus, the
# retrieval and the generator cannot differ between them; the only axes are the judge model
# and the versions of its three prompts, plus `repeat`, which changes nothing and names a
# repetition. The delta of two `repeat` arms is the judge's own noise
curl -sX POST localhost:8000/v1/experiment -H 'Content-Type: application/json' -d '{
  "kind": "rejudge", "name": "judge_noise", "source_run": "grid_gemma3_4b_ru_plain",
  "param": "repeat", "axes": {"repeat": [1, 2]}}'
```

`control_sample: N` judges the axes the arm does not move on N rows only, drawn by question
with the experiment's own seed, so a repeat costs a third of the card time and the axes that
carry the measurement still read every row. The rows outside the sample are marked skipped
rather than left owed, which is one-way: only a hand-built `log_ids` job reaches them again.

A rejudge answers a question the other kinds cannot: how much of a difference between two
runs was the judge rather than the answer. Nothing is generated, so an arm costs a judge
pass and no card time for the generator. The report drops the RRF composite (retrieval
metrics are identical across arms by construction) and gives per-arm means, paired deltas
over eight bootstrap seeds for every pair while the grid holds six arms or fewer and
first-against-the-rest once it does not, the A/B halves, and an `answers_digest`
per arm beside the source's, so "the arms judged the same answers" is a fact of the record
rather than a claim in its description.

To re-judge one run without making an experiment of it: `POST /v1/eval/rejudge` with
`{"source": "<run>", "run_name": "<copy>"}` copies the answers under a new name with the
verdicts cleared and queues the judge over them.

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
        "comparisons": {"5_vs_10": {"faithfulness": {"mean_delta": 0.19, "ci95": [-0.17, 0.57], "p": 0.3669, "n": 100, "holm_threshold": null, "significant_raw": false, "significant_holm": false}, "...": "..."}},
        "method": "holm", "alpha": 0.05, "tests": 15, "family": "every pair of the grid on every axis"
      }
    }
  }
}
```

The pairwise block is what keeps conclusions honest: an early version of this bench "concluded" k=5 beats k=10 from a composite-score gap in the third decimal - the paired test shows that comparison is a coin flip (p=0.37), and the corrected flags mark which of the 15 grid tests survive at all. See [docs/experiments.md](docs/experiments.md) for the cases where this reversed our own verdicts.

Before a grid starts, `scripts/preflight_grid.py` refuses the ways a run silently stops meaning
anything: an edited tree, a worker running yesterday's code, a model that spilled to the CPU, a
corpus that no longer cuts into the rows it holds, a search depth the planner has quietly stopped
walking the index at. `--verify` checks a finished run instead. Every one of those failures produces
a completed run with plausible numbers and no error anywhere, which is why the check exists rather
than a test. What each of the sixteen refuses and which incident put it there:
[docs/preflight.md](docs/preflight.md).

Record the takeaway with `PUT /v1/experiment/{id}/conclusion` and the experiment becomes a self-contained artifact: what was varied, on what data, the numbers, the verdict.

Observability:
- `GET /v1/question-log`, `GET /v1/question-log/{id}` (the row carries what it was asked and what it read: `question_text`, `reference_answer` and `contexts`, the chunks as elements rather than the string they were joined into, because that join cannot be undone. Filters incl. `pipeline`, `faithfulness`/`relevance`/`completeness`, `run_name`; and over the recorded snapshot: `rerank`, `rerank_device`, `phased`, `empty_retrieval`, `max_distance`, `answered_via_remote` - so "show me every answer where the corpus returned nothing" is one request; detail with context)
- `GET /v1/job`, `GET /v1/job/{id}` (jobs + elapsed), `POST /v1/job/{id}/cancel` (cancels the job and its dependent judge, cooperative stop for a running eval)
- `POST /v1/job/cancel` (cancel a whole run or job type at once: cancelling id by id through a paginated listing is how a supposedly stopped eval quietly kept running). A `type` with no `run_name` is refused with 400 unless the call also passes `every: true` and means it. Either door takes a run's judge down with the run

## MCP

An MCP (Model Context Protocol) server is mounted at `/mcp` (streamable HTTP), exposing the corpus to any MCP client (Claude Desktop, Cursor, IDE agents). Built on standalone `fastmcp` and reusing the same retrieval primitives as the REST/agent paths. Tools:
- `search_corpus(query, category?)` - hybrid retrieval, returns chunks with `[source]` markers; optional category subtree filter.
- `answer_question(text, pipeline?, category?, language?)` - full RAG answer, returns `{answer, retrieved, sources}` (`agent` or `single_shot`; `category` only with `single_shot`).
- `list_categories(category?, only_top?)` - category paths with chunk counts, for discovering valid filter values before searching.

Connect: `claude mcp add --transport http rag-lab http://127.0.0.1:8000/mcp/`, or point the MCP Inspector at the same URL.

A second, separate ops server is mounted at `/mcp-ops` - an eval control plane kept off the product surface (an external client gets search/answer tools, not admin verbs):
- `run_metrics(run_name)` - aggregated eval metrics for one run (generation axes + retrieval hit@k/MRR).
- `compare_runs(run_names)` - side-by-side metrics with an RRF composite ranking over the five judged-and-behavioural axes, retrieval excluded.
- `compare_pools(run_names)` - the same runs split by pool (in-corpus / out-of-corpus / off-domain) with gate firings, latency, outcome histogram and a paired Wilcoxon per pair of runs.
- `experiment_results(id, pair?)` - one experiment's report, whatever its kind: the arms with their n, the paired deltas per axis with interval and p, and whether each survives the correction over the family the record names.
- `list_jobs(status?, type?, run_name?)` / `cancel_job(id)` - job queue control, cancel takes the dependent judge down with the run.

### MCP client: the agent consumes external servers

The lab is both sides of the protocol: its own MCP server above, and an MCP *client* below. External hosted MCP servers are registered as `McpIntegration` rows and their tools join the agent's toolbox next to `search_corpus`, namespaced `integration__tool` (e.g. `deepwiki__ask_question`). The agent decides per hop whether to look outside the corpus; a successful remote call is recorded as an `mcp:` source (provenance), a failed one degrades to an error string the agent can route around.

<details>
<summary>Diagram: Agent flow with remote fallback</summary>

![Agent flow with remote fallback](docs/diagrams/agent_flow.svg)

</details>

This is the policy flow, the same for every implementation of the loop. What executes it is the graph, and its picture is generated from the compiled graph itself: see [the implementations](#two-ways-to-run-the-same-agent-and-the-two-that-were-retired).

Registry lifecycle via `/v1/mcp_integration`:
- CRUD with filters; new integrations start `disabled`, a state machine (`disabled/active/unreachable`) separates operator intent from observed health (probes flip `active <-> unreachable`, never touch `disabled`).
- `POST /{id}/discover` - fetch the server's tool list, cache name/description/schema snapshots in the DB. The agent builds tools from this frozen cache (no network on run start, and a later description swap on the server side does not silently reach the LLM prompt - discover again to refresh).
- `POST /{id}/probe` - live ping writing `last_checked_at/last_error`; `GET /{id}/health` - cheap read of the stored state. A `check_mcp_health` job fires on every create/update (io queue lane, so it never waits behind GPU jobs).
- `allowed_tools` is an explicit allowlist: discovery shows the catalog, a human picks what the 8B model actually sees. Tool descriptions are truncated on cache; results are truncated to `max_result_chars`.

<details>
<summary>Diagram: McpIntegration state machine</summary>

![McpIntegration state machine](docs/diagrams/mcp_state.svg)

</details>

Auth per integration is declared as `{"type": "bearer", "token_env": "HF_TOKEN"}` or `{"type": "header", "header": "...", "value_env": "..."}` - the DB stores only environment variable *names*; values come from the environment and only for variables allowlisted in `config.yaml` (`mcp_integrations.secret_env`).

Note on trust boundaries: anyone with API access can register an integration pointing anywhere, and allowlisted secrets will be sent to that URL (see the Quickstart note on authentication).

Seeded integrations (all disabled until you enable them): DeepWiki (no auth), Hugging Face (`HF_TOKEN`), Context7 (`CONTEXT7_API_KEY`).

### Corpus-first: when the agent is allowed to look outside

Handing an 8B model a toolbox and hoping it prefers the local corpus is not a policy. `fallback_policy` makes the order explicit and, more importantly, measurable:

- `corpus_first` (default) - the run starts with `search_corpus` alone. External tools are not in the schema and `dispatch` refuses them by name, so a hallucinated tool call cannot leak out. The moment a corpus search comes back empty, the remote schemas join the toolbox from the next hop and the log records `fallback_reason: empty`.
- `corpus_first_weak` - same, plus a coverage gate over the hits that did come back. `agent.gate_signal` decides what counts as weak: `distance` (default) compares the vector distance of the best hit against `agent.weak_distance`, `cross_encoder` scores the top `agent.gate_candidates` hits and compares the best against `agent.weak_threshold`, `either` opens when one of the two says weak. Below the bar the retrieval counts as a miss (`fallback_reason: weak`) and the weak chunks are dropped from the conversation instead of being answered from, but only when an external tool exists to take over. The cross-encoder path does not need reranking to be on: with `rerank: false` it scores five pairs and leaves the ordering alone; with reranking on it reuses the scores already computed.
- `agent_choice` - everything visible from hop one (the pre-policy behaviour, kept as the A/B baseline).

What is forced is the *fact* of asking the corpus, not the wording: rephrasing and the decision to go outside stay with the model, which is where an 8B is actually decent (cross-language retrieval included).

Before any of that, a tool has to earn its place in the toolbox: each external tool is checked once per run against the question, and one that would force the model to invent a required argument is not offered at all. Without that check an in-corpus question about cooking was sent to a repository tool with an invented repo name and faithfulness on such questions fell to zero. Two cheaper routers were measured first and both failed. A rejected tool costs nothing: with nothing to hand off to, the run degrades to plain corpus behaviour.

When no source answers at all, the run no longer ends in silence: the final turn runs without tools, carrying a versioned instruction (`agent.no_evidence`) to say plainly that the available sources do not cover the question and not to answer from memory. Those runs come back as refusals in the language of the question instead of empty results.

When the toolbox opens, the model is told so: a versioned notice (`agent.fallback` prompt) is appended to the tool result, listing the external tools and their required arguments. It rides in the tool result on purpose. A `system` message injected mid-conversation breaks the llama3.1 chat template badly enough that the model starts printing tool calls as prose, role header included. Two more guards come from the same failure family: `dispatch` answers a call with missing required arguments by naming them (an 8B tends to reuse the argument shape of the previous tool), and a turn that narrates a call instead of issuing one gets exactly one nudge to do it for real.

Every remote failure is classified on the call path (`timeout`, `connect`, `auth`, `client`, `server`, `tool`) and lands in `metrics.tool_errors`, because the kinds carry different policies: `auth` means the key is dead and retrying is pointless, `tool` means our arguments are wrong, the rest are transient. Two traps found by probing real failures: fastmcp hides a dead host under `RuntimeError` with the real `httpx.ConnectError` on `__cause__` inside a TaskGroup `ExceptionGroup`, and an unknown tool name comes back as a server-side `ToolError` rather than any transport error.

Both the policy and the reason ride in the log snapshot, so runs before and after are comparable, and `/v1/question-log?fallback_reason=empty` pulls exactly the questions the corpus could not serve.

Measured on three pools of a hundred questions each. The empty rule fires **zero** times, because hybrid search always returns something above the threshold; the coverage gate turns that into 60 questions sent outside and lifts grounding on them from 2.31 to 5.08 (p<0.001) while the in-corpus half does not move. Corpus priority is free in quality and not in time: about two seconds per question for the gate, twelve more for an external hop. Numbers and caveats: [the journal entry](docs/experiments/2026-08-25_the-gate-that-fires-and-the-refusal-that.md).

Which signal calls retrieval weak got its own A/B, three arms of 200 questions with one variable. No axis separates `distance` from the cross-encoder, and the intervals bound how strong that claim may be (±0.8 of a point in-corpus, ±1.5 outside), while `distance` is seconds cheaper, so it is the default. `either` buys relevance and not grounding, and relevance is the axis the same measurement shows to be blind to hallucination, so it stays a measured option rather than the default.

The same run says something less comfortable: **no policy ever refuses.** On the hundred questions nothing can answer, all three arms answer, fluently and grounded in nothing. Deciding whether the corpus covers a question is not the same as deciding whether anyone does. Numbers: [the journal entry](docs/experiments/2026-08-25_the-gate-that-fires-and-the-refusal-that.md).

### The topic axis: refusing instead of reaching out

Coverage and topic are different questions, and the run above answers only the first. `agent.topic_threshold` adds the second: the distance from the question to the nearest chunk, computed before any tool is offered. Above it the question is not ours, nothing external is admitted and the run refuses. Refusals on the off-domain hundred go from 11 to 50 (p<0.001) with false refusals on in-corpus paraphrases staying at 0 of 100; both limits were written down before the run.

The threshold is declared per language, because the axis is a distance to *this* corpus and how far an off-topic question lands depends on whether it shares the corpus's vocabulary. The corpus is almost entirely English, so an off-domain English question sits at a median 0.4547 from it against 0.5116 for a Russian one, and one number cannot be right for both.

Re-derived on 1643 in-corpus and 231 off-domain questions in August 2026, against a 1% budget for false refusals. The budget sits there because of the shape of the curve rather than taste: the first percent buys 33 and 39 catches for 6 and 7 wrongly refused questions, about five and a half each, while the second buys English eight catches for nine refusals and Russian nothing at all. Two distributions overlap in a thin band; inside it the threshold cuts the body of the off-domain distribution and only the extreme tail of the in-corpus one, and past it each step costs more real questions than it saves foreign ones.

The knee is estimated from six or seven questions, so the values are fitted to the sample they are read on. Choosing on half A and reporting on half B says how much of that survives: the catch does, 84% Russian and 73% English, and at this budget so does the cost, 1.42% and 0.47% of false refusals on the unseen halves. At a 2% budget the cost did not survive: the Russian half came back at 2.61%. A budget is a target with about a point of slop, and buying that slop back is part of what the first percent is worth.

A language nobody measured gets the most permissive of the declared thresholds: the gate refuses only where refusing was shown not to cost a real question. The row records both the threshold applied to it and the policy that produced it, and a comparison pins the policy.

The half that does not work is the more interesting one. The catch is on distant topics (cooking, chemistry, law) and almost nothing on legacy stacks and post-cutoff technology: to a distance metric, FoxPro and Postgres are the same topic. Those questions are inside the topic and outside the corpus, which is coverage plus recency and needs a different mechanism. Numbers and the veto rules: [the journal entry](docs/experiments/2026-08-25_a-refusal-at-last-and-the.md).

### Corpus hygiene: the cut is a parameter, not a fact

A corpus variant is a named cut of the same sources living beside the others, with its own partial vector index and its own policy in every run snapshot. Two are declared in `config.yaml`:

| variant | chunker | what it is for |
|---|---|---|
| `baseline` | `legacy` | frozen: the cut the corpus was first measured on, never re-indexed |
| `clean_1024` | `rooted` | the same size cut, but the heading path comes from a declared root, frontmatter is parsed and the junk is gone. Isolates source hygiene from the splitter |

Variants that were measured and lost are gone from the file and from the table: `prefix_1024` (subheadings before size), `cap_2048` and `noboiler_1024`. Their policies ride in the journal entries that measured them, so one job rebuilds any of them, and a variant nobody serves stops paying into every other variant's query plan.

The policy carries the whole rule, so a variant gets the cut it asks for and nothing else: `chunker` (`legacy`, `rooted` or `structured`), `max_chunk_size`, `ceiling_on` (whether the ceiling counts the body alone or the prefix with it) and `drop_boilerplate` (off by default: a block repeated verbatim across half a source's files is dropped unless it is the only carrier of its section, and since that changes the cut it is a variant of its own rather than a switch under an existing one), typed with `extra: forbid` so a key nobody reads fails the start rather than reading as a switch. `header_prefix` is derived from `chunker` rather than declared, because two keys deciding one thing is how they came to disagree. Nothing is dropped, parsed or prefixed unless a policy says so, and the preflight re-cuts every indexed variant and compares the text of each chunk against what the table holds, because a source that changed can keep its row count exactly.

`baseline → clean_1024` on 823 questions is **+0.0435 section MRR** over the whole set with all six intervals clear of zero; on the pre-registered reporting half it is **+0.0494**. `clean_1024 → prefix_1024` on the same questions is **−0.0109 [−0.0253, +0.0032]**: no winner on the half where the winner is chosen, so the third heading level bought nothing and was removed.

### How deep the index is walked, and why that is not a number

`ef_search` is pinned to 100, and `auto` is the alternative it was pinned against: the deepest rung of `ef_ladder` whose plan still contains an `Index Scan`, asked of the planner rather than remembered. Where the planner abandons the walk moves with what is indexed: adding a variant moved that point from about 197 to about 265, deleting one moved it back not at all (a `DELETE` leaves the pages where they were), and a variant with fewer rows gets a deeper answer than its neighbours in the same table. A depth that moves with what else happens to be indexed is not reproducible across days, which is why the number is in the file.

Pinning is safe rather than blind because the audit checks the pinned number, not only a resolved one: the preflight asks the planner whether this depth still walks the index on **every indexed variant**, and refuses when any of them sorts, so a moved crossover turns the preflight red instead of quietly measuring exact search while the record says hnsw. That happened on the day the depth was first pinned. It was pinned at 200 with three variants indexed; dropping the losing variant and rewriting the table moved the crossover from about 248 to about 123, 200 stopped walking the index, the preflight said so, and the pin moved to 100. The rewrite also rebuilt the hnsw graphs, so the shallower rung came out cheaper than the deeper one had been: at 100 the paired section MRR against exact search is now +0.0000 [0.0000, 0.0000] with nothing lost and recall@20 of 1.0 (823 questions, `clean_1024`), where the same rung read +0.0011 with one question lost before the rewrite. Under `auto` the resolved answer is cached against the two statistics the planner itself reads (`relpages`, `reltuples`) and re-asked when either moves. `ef_search` on a request, on a run, or as a comparison axis overrides the file either way; what a run used is a number in its snapshot.

Comparing cuts is an experiment of its own kind rather than a second entity. `POST /v1/experiment` with `kind: retrieval` takes `axes` instead of one swept parameter (`{"variant": ["baseline", "clean_1024"], "rerank_top": [0, 20]}`), measures every point of the grid on the same fixed questions and reports, for each arm, a paired delta of MRR against the arm that differs from it in the axis of record alone, with a bootstrap interval. It costs minutes and neither the card nor a judge, because it reads where the right chunk landed rather than what a model said about it. The procedure travels in the result, per arm and in the same fields the standalone report writes: variant and set, the search and its depth, the candidate pools, the threshold, the keyword settings, the questions and their hash, the cut policy and the corpus fingerprint. "These two are not comparable" is a field of the record instead of a line in somebody's terminal.

Whether a source was cut well is asked without embeddings and without labelled questions: `POST /v1/source/{id}/analyze` runs the coverage report over the text alone (share of chunks under a real heading, prefix outweighing its body, duplicates inside a file and inside a source, boilerplate standing in most files, slivers, soup, code with no prose, and how often the counter rather than the author's structure decided a boundary), gates it, scores it 0-100 and keeps the history per variant on `data_sources`. `GET /v1/source/{id}/report` reads that history back and `GET /v1/source/compare` puts two cuts of every source beside each other and counts the ones whose verdict moved, which is how a re-cut is judged before any question is asked of it. A metric with nothing to measure abstains rather than returning zero, because a zero passes a gate and earns its weight.

<details>
<summary>Diagram: Ingestion</summary>

![Ingestion](docs/diagrams/ingestion.svg)

</details>

### Two ways to run the same agent, and the two that were retired

The flow above started as a hand-rolled loop: our own hop counter, our own dispatch, the coverage gate stitched between the turns. It now runs on a graph, and `orchestrator` selects which implementation executes the same policies. Both fill the same `AgentResult`, so logging, the judge and the metrics cannot tell them apart.

| `orchestrator` | implementation | what it applies |
|---|---|---|
| `langgraph_ported` (default) | `StateGraph`, `app/orchestrators/graph.py` | everything |
| `langgraph_idiomatic` | bare `create_agent`, `app/orchestrators/react.py` | tool admission and the topic axis (they run before the branch point), no coverage gate, no context drop, no fallback notice, no nudge, no final turn without tools |

Two implementations were retired, each after it had been measured against the one that stayed: the hand-rolled loop, and an arm that expressed our policies as framework middleware hooks. Both agreed with the graph within the bench's own noise on every key the pipeline reads, and a second implementation of behaviour already pinned by tests buys nothing while costing a branch in every future change. Their runs stay readable in the log under `orchestrator=agent` and `orchestrator=langgraph_middleware`; neither value can be asked for by a new run. Numbers: [the entry](docs/experiments/2026-08-26_the-same-agent-written-four-ways.md).

<details>
<summary>Diagram: The ported agent graph, generated from the compiled graph</summary>

![The ported agent graph, generated from the compiled graph](docs/diagrams/agent_graph.svg)

</details>

The graph picture is generated from the compiled graph by `scripts/graph_to_d2.py`, and CI fails if the committed drawing no longer matches the code.

`/v1/question-log?orchestrator=langgraph_ported` slices the logs by implementation, the same way `fallback_policy` and `fallback_reason` do. The value `agent` is readable there and nowhere else: runs cannot ask for the retired loop.

The policies themselves live in `app/use_cases/agent_policy.py` as plain functions, and both arms that carry policies call the same ones. What differs is the harness, which is the point: it makes "what does the standard cost" a measurable question rather than an opinion. Two things do not survive the move to the standard tool contract, and both are recorded rather than hidden: error kinds (`timeout`, `auth`, `tool`, ...) collapse into success-or-error, and the bare arm has no final turn, so a question that wants a fifth hop ends without an answer instead of with a refusal.

One implementation note worth stealing: under `corpus_first` the withheld external tools have to be refused at dispatch, not merely hidden from the model, because the standard tool node is built once from the full tool list and narrowing what the model sees does not stop a call the model invents.

## How it is built

- `app/config.py` - `config.yaml` loader.
- `app/orm/` - SQLAlchemy: `base` (declarative), `sync_db` (psycopg), `async_db` (asyncpg).
- `app/models/` - ORM models: `registry` (Model/ModelRole/Prompt), `eval` (Question/QuestionLog), `jobs` (Job), `corpus` (DataSource/DataChunk), `experiment` (Experiment + state machine), `mcp_integration` (the remote-tool registry).
- `app/llm.py` - Ollama client via the OpenAI SDK (generation / embeddings / structured output) + role→model resolver.
- `app/rerank.py` - cross-encoder reranker (sentence-transformers, lazy-loaded, on the card by default and refusing a run that finds it on the CPU).
- `app/job_queue.py`, `app/worker.py`, `app/job_handlers/` - Postgres queue (FOR UPDATE SKIP LOCKED) and worker with retries/defer; handlers split by theme.
- `app/bootstrap.py` - idempotent startup init.
- `app/sources/` - per-source ingestion (reader pattern: `Base` ABC + sources), each source declaring its policy rather than being special-cased downstream.
- `app/ingest.py` - the cutter: headings located by `MarkdownHeaderTextSplitter`, text sliced from the file itself, sections cut by subheading before size, slivers merged, and what decided each boundary recorded on the chunk.
- `app/use_cases/ingest_quality.py` - the coverage report: metrics from the text alone, no embeddings and no labelled questions, gated per source and kept as history on `data_sources`.
- `app/db.py` - hybrid search (raw SQL: pgvector `<=>`, FTS, ltree, RRF).
- `app/use_cases/` - `chat` (retrieve/answer), `agent` (policies, logging and the snapshot around the agent run), `agent_policy` (the policies themselves as plain functions), `index` (corpus build), `judge` (answer scoring), `experiment` (series aggregator + RRF composite), `retrieval_compare` (the grid of arms, paired deltas and halves), `rejudge` (re-scoring answers that already exist), `search_depth` (resolving and caching the index depth), `mcp_integration` (remote tools).
- `app/orchestrators/` - adapters to the framework: `graph` (StateGraph), `react` (bare `create_agent`). No langchain import reaches `use_cases`.
- `app/agent_tools.py` - tool registry + `dispatch` + the `search_corpus` tool over hybrid retrieval.
- `app/mcp_server.py` - FastMCP server (mounted at `/mcp`): `search_corpus` / `answer_question` / `list_categories` tools reusing the retrieval primitives.
- `app/mcp_ops.py` - ops MCP server (mounted at `/mcp-ops`): `run_metrics` / `compare_runs` / `compare_pools` / `experiment_results` / `list_jobs` / `cancel_job` over the eval platform.
- `app/evals/pools.py`, `app/evals/compare.py` - one place that decides which pool a question belongs to and what the run's outcome was, shared by the metrics, the comparison report and both MCP tools.
- `app/api/` - REST adapters (health + v1: chat / agent / categories / model / role / source / prompt / eval / experiment / questions / question-log / job).
- `app/seed.py`, `app/console.py` - prompt/question-bank seed; REPL console.
- `app/evals/` - eval bench (runner + retrieval and generation metrics via the judge).
- `tests/` - unit tests (pure logic, no DB/Ollama): `docker compose exec rag-lab pytest -q`.

## Status

A learning project: the goal is to master RAG/LLM engineering by hand, from primitives, and where the industry has a standard, to move onto it and measure what the move costs. RAG from primitives (hybrid, ltree, FTS) + FastAPI server + 5-axis LLM-judge eval + production layer (uv packaging, central config, SQLAlchemy ORM sync+async, OpenAI-compatible client, role-keyed model lifecycle, prompt versioning, async job queue, question bank, reranking, route-driven eval platform, MCP server).
