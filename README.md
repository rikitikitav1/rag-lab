# rag-lab

[![CI](https://github.com/rikitikitav1/rag-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rikitikitav1/rag-lab/actions/workflows/ci.yml)

rag-lab is a bench for measuring how different factors affect the answers of a RAG system: the model and the engine it runs on, the way the corpus is cut, retrieval settings, an agent instead of a single pass, the judge. It shows what the quality of the answers really depends on and what each change costs.

Every such comparison is an experiment: several variants of the system that differ in one parameter (the arms on the diagram) answer the same set of questions, and a judge scores each answer on its own. A difference between variants is weighed against the spread between two runs of the same variant: anything below that spread is treated as noise. If the variants differ in anything besides what is being compared, the bench records it and says so.

<details>
<summary>Diagram: what an experiment can compare</summary>

![What an experiment can compare](docs/diagrams/what_we_compare.svg)

</details>

> In Russian: [docs/README_ru.md](docs/README_ru.md) · Hands-on scenarios: [docs/use_cases.md](docs/use_cases.md) · Experiments log: [docs/experiments.md](docs/experiments.md) · What the preflight refuses: [docs/preflight.md](docs/preflight.md) · What each role requires of a model: [docs/model_requirements.md](docs/model_requirements.md)

## Why a bench

Anyone can put a RAG together, and it works until the first change: another model, another quantization, another engine, a new judge prompt, a new cut of the corpus. Every such change changes the ruler itself. Without a bench a team compares the mean before and after, reads 7.3 against 7.1 as a result, and ships it. With one, "did it get better, or is that noise" becomes a forty-minute question with a numeric answer and a known noise floor.

It is also the only way to see what other people's defaults do to your numbers. A server applies a repetition penalty the request never asked for; a prefix cache answers the first call differently from the rest; an OpenAI-compatible endpoint drops a parameter without a word; a JSON grammar flag moves a share of the scores while the means stand still. None of that shows on a dashboard. It shows only where the bench records what was sent, what the server was running and what came back, and says so before it compares variants that differ in any of it.

Who does not need this: a product with one model that nobody plans to replace. Ten questions and a careful look will do. Anyone who swaps a model or an engine more than once a quarter needs it, and usually finds out after an "improvement" has already reached production.

## Questions it answers

- **Does a bigger model pay off?** A 70b generator answered only slightly better than the 8b one and ran 23 times slower (it had to run on the CPU). Reranking the 8b's context got almost the same gain. [Entry](docs/experiments/2026-07-28_generator-a-b-llama3-1-8b.md)
- **Is reranking worth its space on the GPU?** On the mixed question set it improves ranking. It is off by default: the agent needs a generator that can call tools, and such a model does not fit on the GPU (called the card below) together with the reranker. [Entry](docs/experiments/2026-08-29_generator-grid-4b-against-8b.md)
- **How should the corpus be cut?** Two cuts can live side by side, each with its own index, so a new cut is compared against the old one instead of replacing it blindly. Cleaning up the cut improved retrieval; a third heading level did not. [Two cuts](docs/experiments/2026-08-26_a-corpus-you-can-keep-two-of.md) · [cleaning](docs/experiments/2026-08-27_hygiene-that-moved-the-number.md) · [third level](docs/experiments/2026-08-28_a-third-heading-level-in-the-cut.md)
- **A hand-rolled agent or the standard one?** The agent ported to LangGraph behaved like the hand-rolled one, within the hand-rolled one's usual spread, and only then was the hand-rolled one retired. [Entry](docs/experiments/2026-08-26_the-same-agent-written-four-ways.md)
- **Does filtering the retrieved chunks with an LLM pay off?** Before the generator sees them, a second model call reads each retrieved chunk against the question and drops the ones it calls irrelevant. This is the document-grading step that Corrective RAG (Yan et al., 2024) and Self-RAG (Asai et al., 2023) both contain, in the form LangGraph's archived tutorials for the two papers share, and LangGraph files both papers under self-reflective RAG. On our English question set (820 questions) a prompted `llama3.1:8b` kept the right section on 97.6% of the rows where retrieval had found it and dropped, on a typical row, 44% of the chunks that came from unrelated files. Our bar asked for at least 95% and at least half, both read on the lower edge of the interval, and no setting of the grader met both. A hundred questions answered with the filter and without it, judged in one residency, moved groundedness by less than the generator scores against itself, so it is off by default and costs 4.4 seconds a question when it is on. A blind reading of a hundred of the chunks the bar counts as unrelated, corroborated by the cross-encoder and by heading overlap, found that 39 to 48 of them are on the subject of the question, so the floor was asking for a large share of what could be dropped at all. [Entry](docs/experiments/2026-09-19_a-grader-that-does-not-buy-it.md)
- **Does rewriting the question when the search looks weak pay off?** When the nearest retrieved chunk is far, or the five chunks the path serves lie within a hair of each other, a second model call rewrites the question and the search runs again, which is what Corrective RAG (Yan et al., 2024) does when it judges retrieval wrong. The trigger is the stand's own calibrated weak distance, and it fires on 136 of our 820 English questions with precision 0.42 against "the gold section is not in the top five". Replacing the first five chunks with the rewrite's five recovered the right section on 22 rows and lost it on 14: a net of +8 rows, +0.0098, with an interval of [-0.0049, +0.0244] that still covers zero, and exactly zero change in whether the right file was found at all. Telling an effect this size from nothing would need about 1850 questions. The node was then removed: a gain this small does not pay for a second model call and a second search on one row in six. The entry keeps the measurement and the trigger's numbers. [Entry](docs/experiments/2026-09-20_a-rewrite-that-does-not-clear-the-bar.md)
- **Does a reasoning schema lift a small model's tool calling?** Instead of calling a tool itself, `llama3.1:8b` filled a schema whose action field names the tool, so the choice became a validated field, the way Schema-Guided Reasoning proposes. On 200 questions outside the corpus the share of rows that described a tool call instead of making one, or spent every hop, went from 0.745 to 0.990: the schema removed the described calls and the rows ran out of hops instead. Answers without a source went from 4.5% to 11.5%, and on the corpus questions relevance fell from 9.1 to 5.6 of ten. The arm was removed. [Entry](docs/experiments/2026-09-23_a-schema-that-makes-the-loop-worse.md)
- **Can the judge be trusted?** Judges of different size agree on how complete an answer is and disagree on how well it is grounded, so scores compare only within one judge. Our judge and RAGAS (a standard evaluation library, run here as a second, guest judge) agree moderately (rank correlation 0.5 on questions from the corpus); RAGAS is the noisier one, and on refusals the two disagree by design. [Two judges](docs/experiments/2026-07-29_judge-vs-judge-qwen2-5-7b.md) · [ours against RAGAS](docs/experiments/2026-09-06_our-judge-against-the-standards.md)
- **Does the engine change the score?** The same answers scored by a judge on vLLM and on ollama got different scores in many cases, but that cannot be put down to the engine alone: the quantization of the model differs too. So the bench refuses to compare variants scored by judges on different engines. [Entry](docs/experiments/2026-09-09_the-same-rows-judged-by-two-engines.md)

## Why the numbers hold

- **Question by question, with an interval.** All variants answer the same questions, so every difference comes with a confidence interval and a correction for multiple comparisons. Before that, a "winner" sometimes won by less than the usual spread. [Entry](docs/experiments/2026-07-28_paired-significance-testing-lands-in-the.md)
- **A measured noise floor.** A variant is run twice and a judge scores twice, so the noise has a number; a difference below it is treated as noise, not as a result. The same check showed that vLLM's batch invariance flag (`VLLM_BATCH_INVARIANT`, a judging pass 8.7 times slower) buys nothing here. [Entry](docs/experiments/2026-09-09_what-batch-invariance-costs-on-an-awq-judge.md)
- **An outcome for every question, scored by group.** Each answer is marked: answered with sources, answered without any source, refused, ran out of steps. Questions from the corpus, outside it and off topic are scored separately. That is how an agent that never refused anything was caught. [Entry](docs/experiments/2026-08-25_the-gate-that-fires-and-the-refusal-that.md)
- **A run records its instruments.** Models, prompts, engine settings and the corpus variant are saved with every run. That is how a rewritten agent was checked: recorded runs were replayed through the new code without calling the model again, and every replayable row matched. [Entry](docs/experiments/2026-09-07_the-phases-split-and-the-replay-that-checked-it.md)
- **A run says what it promised before it starts.** A closing run names a preregistration, a record in the database of its population, arms, closing columns and guards, and the queue refuses a name it does not hold. The closing number is then computed from that record through the ops MCP server, so a column picked after seeing the numbers cannot be reported as the result. [Tools](docs/mcp.md)
- **What two arms must share.** Before a difference is read, the comparison checks that both arms were measured by one instrument, and names the field that differs when they were not. [The table](docs/api.md#what-two-arms-must-share)

How much each instrument moves on its own, measured by running it twice on the same material. A verdict is one judge score on one axis of one answer; a reload is the model unloaded from its server and loaded again.

| Instrument | Moved between two passes | Entry |
|---|---|---|
| judge Qwen2.5-7B on ollama, across a reload | 14% of scores, 58% of reason texts | [entry](docs/experiments/2026-09-07_what-the-standard-was-worth.md) |
| judge Qwen2.5-7B-AWQ on vLLM, across a reload | 0 of 388 verdicts | [entry](docs/experiments/2026-09-09_what-batch-invariance-costs-on-an-awq-judge.md) |
| the same judge in one load, JSON without free whitespace | 0 of 275 verdicts | [entry](docs/experiments/2026-09-13_the-judge-that-looped-on-whitespace.md) |
| DeepSeek-V4-Flash as a judge, passes hours apart | 154 of 573 verdicts, 27% | [entry](docs/experiments/2026-09-14_a-cloud-judge-on-the-same-answers.md) |
| generator Qwen2.5-7B-AWQ on vLLM, two runs | 78 of 285 verdicts, 27%; 8 of 100 answers equal | [entry](docs/experiments/2026-09-13_the-same-generator-on-two-engines.md) |
| generator qwen2.5:7b on ollama, two runs | 101 of 300 verdicts, 34%; 2 of 100 answers equal | [entry](docs/experiments/2026-09-13_the-same-generator-on-two-engines.md) |
| generator llama3.1:8b on ollama, two runs | about a point per row; 4% of answers equal | [entry](docs/experiments/2026-09-13_a-bigger-model-at-the-same-retrieval.md) |
| generator DeepSeek-V4-Flash, two runs | 0 of 50 answers equal; faithfulness moved on 18 of 44 | [entry](docs/experiments/2026-09-13_a-bigger-model-at-the-same-retrieval.md) |
| a human rater (the author), the same pairs a day later | the same choice in 5 of 7 | [entry](docs/experiments/2026-09-13_fifteen-pairs-the-owner-judged.md) |

A difference smaller than its instrument's own movement is read as noise.

## What's inside

Under the bench is an ordinary RAG system: it answers technical questions from a personal knowledge base and a set of public IT repositories, and shows the sources it used. Search is written by hand from basic parts, so that every step can be measured; where the industry has a standard that does the job better, the bench moves onto it and measures what the move costs (the agent runs on LangGraph, markdown is split by `MarkdownHeaderTextSplitter`). Models run locally on one GPU, which ollama and vLLM hand to each other through the job queue, or on the CPU; a cloud model can be plugged in as a remote engine.

- **Corpus variants.** The same corpus can be cut in several ways at once, each cut with its own index, so re-cutting it is something you compare rather than a step you cannot take back. [Entry](docs/experiments/2026-08-26_a-corpus-you-can-keep-two-of.md)
- **Hybrid search.** Vector search (pgvector) and full-text search (Postgres FTS with stemming per language) are merged by RRF. Results can be filtered by category, and a distance threshold lets the system say "the corpus has nothing on this" instead of guessing.
- **Models by role.** There are eight roles: six of our own (generation, embeddings, judging, paraphrasing, reranking, grading) and two for the RAGAS guest judge, its model and its embedder. Which model serves a role is stored in the database and can be changed without a restart. Each role sits on an engine: ollama or vLLM, on the card or on the CPU, or a cloud API.
- **Several sources, one category tree.** Personal notes in Russian, 173 interview-question repositories in English, and three documentation sources (`system-design-primer`, `redis-doc`, `cheatsheets`), each with its own rules for what counts as a heading and what is junk.
- **Five quality measures.** Retrieval (did the right file and the right section come back) and, scored by an LLM judge, faithfulness (is the answer supported by the context), relevance, completeness against a reference answer, and whether the system refuses when it should. Scores are kept separately for questions from the corpus, outside it and off topic, and every question gets an outcome. When the system refuses, or answers without any source, the judge leaves the row unscored; such rows are counted by their outcome instead.
- **A job queue.** Heavy work (pulling models, indexing, running and judging question sets) goes through a queue in Postgres, processed by a worker. The app itself depends only on Postgres: if a model server goes away, jobs wait and the app does not crash; the compose stand starts the API only after the model servers are up.
- **Reranking.** A cross-encoder (`bge-reranker-v2-m3`) can re-order search results before they reach the generator. It is off by default because it does not fit on the card next to the agent's generator; a run can turn it on once its server is up, while the chat cannot, since the reranker and the generator would be two engines on the card. What it does to ranking and answers is in the questions above.
- **An agent on LangGraph.** The model decides when to search, may rephrase the question and search again, then answers. The rules around it (when retrieval counts as weak, when a question is off topic, which tools it may call) are ours and measured. Any set of questions can be run through the agent or through a single retrieve-and-answer pass, and the two compared.
- **Everything through the API.** Building question sets (paraphrasing and translating interview questions), importing your own, running and judging them are API calls that go through the queue; every request and every job is logged with its timing.

## Stack

Python · PostgreSQL + pgvector · SQLAlchemy 2.0 (sync psycopg + async asyncpg) · Ollama and vLLM (OpenAI-compatible APIs, one GPU handed between them) · FastAPI · dbmate (migrations) · uv/pyproject · Docker Compose.

The standard's axes come from **ragas**, pinned at `0.4.3` in the `eval` dependency group and installed in the images of both services, so a guest pass runs inside the queue rather than beside it in a host shell. `langchain-community` is pinned below `0.4`, because that release dropped `chat_models.vertexai`, which ragas imports.

From the LangChain family, five packages: **langgraph** runs the agent as a `StateGraph`, **langchain** provides `create_agent` for a variant of the agent built on the stock `create_agent`, which measures what the standard version costs, **langchain-text-splitters** decides what counts as a markdown heading, and **langchain-core** with **langchain-ollama** are what that variant imports directly. Retrieval, the queue, the eval bench and the corpus policies are ours; the version of the splitter travels in the ingest report, because the cut is external code now and a lock refresh would otherwise move it silently.

## Models and prompts architecture

- Which model serves each of the seven roles is a row in the database, switched at runtime; a model is asked whether it can do its role's job before it is seated ([docs/model_requirements.md](docs/model_requirements.md)).
- Prompts are versioned in the database from files in `prompts/`, one active version per purpose, and a new version is switched on deliberately.
- Every answer records the models and prompt versions that produced and judged it, so two runs are compared on what actually ran rather than on what was asked for.

The data model, the reranker's server and what bootstrap does on start: [docs/design.md](docs/design.md#models-prompts-and-what-a-run-records).

## Configuration

Each kind of setting has one home:

- `config.yaml` (mounted into the container): the pipeline, that is the roles and their models, retrieval, reranking, the agent, ingestion, text search, corpus variants and sources. A value chosen by measurement carries its reason and its measurement file beside it, and every run records the values it used in its snapshot.
- `.env`: what depends on the machine or must stay out of the repo (timeouts, card shares, keys); [`.env.example`](.env.example) lists every variable with its default.
- the database: which model serves a role, prompt versions and engine rows, switched at runtime through the API.
- `datasets/`: the question banks and the corpus sources. A pass writes its measurement and its frozen candidate pool here too, and those stay out of git: a number reaches a reader as the table in its journal entry, with the file name and the job id as its address.
- the code: the category trees of the sources and the protocol limits.

## Quickstart

The host needs Docker Compose v2, an NVIDIA driver and the NVIDIA Container Toolkit with a CDI spec: `docker info` must list `nvidia.com/gpu=all` among the discovered CDI devices (tested on Docker Engine 29.0, Compose 2.40, toolkit 1.20). Without it, `sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml` writes the spec, and the toolkit's `nvidia-cdi-refresh` unit keeps it current after a driver update. The card goes to the containers through CDI rather than the runtime hook because on a cgroup v2 host with the systemd driver every `systemctl daemon-reload` (snapd and unattended upgrades run one on their own) took the card from the running containers.

```bash
scripts/up.sh                            # checks the card, then docker compose up -d
curl localhost:8000/readiness            # 503 only when Postgres is down; otherwise "ok", or "degraded" naming the roles whose engine is down
curl -X POST localhost:8000/v1/chat/question \
  -H 'Content-Type: application/json' -d '{"text": "What is a hash table?"}'
# Swagger: http://localhost:8000/docs
```

`scripts/up.sh` rather than a bare `docker compose up -d`: on a host without a card it says why the stand would not start instead of Docker's "unresolvable CDI devices", it unloads ollama's models before `vllm` starts, and it hands `vllm` the judge `config.yaml` names. `scripts/up.sh --cpu` brings the stand up without a card, every role on the processor ollama: [docs/stand_modes.md](docs/stand_modes.md), mode 8.

<details>
<summary>Diagram: how the stand comes up</summary>

![How the stand comes up: scripts/up.sh, the compose chain, bootstrap, and who waits for whom](docs/diagrams/stand_up.svg)

</details>

No authentication by design (REST, `/mcp`, `/mcp-ops` are all open): this is a local lab bound to 127.0.0.1. Do not expose it to a network as is.

The first `up` pulls the models of the roles (on ollama about 15 GiB: `llama3.1:8b`, `gemma2:9b`, `bge-m3` and `qwen2.5:7b` for the RAGAS guest; the judge on vLLM about 5.2 GiB) and the vLLM image (about 21.5 GB of disk), then builds the index (~5-10 min, watch `docker compose logs -f worker`). The server waits for `bootstrap`, which waits for `vllm` to report healthy (up to an hour on a first start, while it downloads the judge) and for ollama; it does **not** wait for the pulls and the indexing those steps queue, so the first requests may refuse until the corpus fills up.

The `notes` source reads `~/working_docs/notes` on the host, mounted read-only into the containers. On a machine without that directory Docker creates it empty, and the source indexes nothing.

Hands-on scenarios (mini-eval to numbers, reranking A/B, importing your own questions, browsing logs): **[docs/use_cases.md](docs/use_cases.md)**. It is a walkthrough of nine scenarios, not a route index; the complete reference is Swagger at `/docs`, which is generated from the code and cannot fall behind it.

Other layouts of the stand (reranking, an embedder or the generator on vLLM, the processor, the judge on ollama, a stuck card), step by step and with the way back: **[docs/stand_modes.md](docs/stand_modes.md)**.

Your first comparison: [scenario 2](docs/use_cases.md#scenario-2-mini-eval-from-scratch-to-numbers) takes a small set from scratch to numbers, and [scenario 9](docs/use_cases.md#scenario-9-parameter-series-measure-a-retrieval-lever) sweeps one lever as an experiment.

## Architecture

<details>
<summary>Diagram: Architecture</summary>

![Architecture](docs/diagrams/architecture.svg)

</details>

How the card moves between engines, and where a failed handover leads: [docs/stand_modes.md](docs/stand_modes.md).

Diagrams are D2 and PlantUML sources in `docs/diagrams/`, rendered by `scripts/render_diagrams.sh`; CI fails if a committed SVG drifts from its source. Structural and behavioural views are UML and C4; the generated agent graph and the explanatory drawings stay in D2.

## Compose services

| Service | Role |
|---------|------|
| `postgresql` | Postgres + pgvector, the only hard dependency |
| `dbmate` | applies migrations, runs to completion before the rest |
| `seed` | loads prompts and the question bank, runs once after migrations |
| `bootstrap` | prepares models, roles and indexing jobs, runs to completion before the rest; waits for `vllm` and puts it to sleep before ollama loads a role |
| `rag-lab` | FastAPI server (uvicorn) |
| `repos-owner` | hands the `repos_data` volume to the host's user before the worker starts, runs once |
| `worker` | processes the job queue as the host's user, sixteen types listed with what each one takes in [docs/api.md](docs/api.md#the-queue) |
| `ollama` | local inference on GPU: the generator, the embedder, the paraphraser and the RAGAS guest's model |
| `ollama-cpu` | a second ollama on the processor, for a role that should not take the card (the RAGAS guest's embedder by default); always up, since ollama loads a model on the first call and gives the memory back after its keep-alive |
| `vllm` | the judge (`Qwen/Qwen2.5-7B-Instruct-AWQ`); takes the card first at start and is put to sleep whenever another engine needs it; its port is not published on the host |
| `vllm-rerank`, `vllm-embed`, `vllm-cpu` | under the compose profiles `rerank`, `embed` and `cpu`: the reranking role, an embedder on vLLM, vLLM on the processor; started only with `--profile`, because a vLLM holds its share of the card, or its whole model in host memory (19 GB for a 7B on the processor), for as long as it runs; how to switch: [docs/stand_modes.md](docs/stand_modes.md) |

### Environment knobs

Copy [`.env.example`](.env.example) to `.env`, compose picks it up; what each variable does is written beside it. What changes from one stand mode to another: [docs/stand_modes.md](docs/stand_modes.md).

## REST API

The complete reference is Swagger at `http://localhost:8000/docs`. [docs/api.md](docs/api.md) maps the routes by group, and [docs/use_cases.md](docs/use_cases.md) walks through them as scenarios.

## MCP

Two MCP servers are mounted on the API: `/mcp` for the corpus (`search_corpus`, `answer_question`, `list_categories`) and `/mcp-ops` for the stand itself (runs, comparisons, jobs, engines). The agent is also an MCP client of external servers. Details: [docs/mcp.md](docs/mcp.md).

## Design notes

Why the agent and the corpus are built the way they are, each decision with its journal entry: [docs/design.md](docs/design.md). It covers corpus-first and the coverage gate, the topic axis, corpus hygiene, how deep the index is walked, the two implementations of the agent, and the data model of models, prompts and runs.

## How it is built

Where each part of the code lives: [docs/code_map.md](docs/code_map.md). The logic lives in transport-neutral `use_cases` behind thin adapters (CLI, FastAPI REST, MCP).
