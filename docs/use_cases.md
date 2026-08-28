# Use cases and commands

Hands-on scenarios for rag-lab. Each is copy-paste ready. Everything is also clickable in Swagger at `http://localhost:8000/docs`.

## Prerequisites

- Docker + an NVIDIA GPU (8 GB is enough).
- First `docker compose up -d` pulls ~16 GB of models and builds the index (~5-10 min). Wait until `curl localhost:8000/readiness` returns ok; watch progress with `docker compose logs -f worker`.
- The server answers before indexing finishes, so early requests may refuse until the corpus is populated.

## Scenario 1: ask a question (RAG live)

```bash
curl -sX POST localhost:8000/v1/chat/question -H 'Content-Type: application/json' \
  -d '{"text":"What is a hash table?"}' | python3 -m json.tool
```
Returns the answer, the retrieved sources (with vector/keyword ranks and score), and token/time metrics. Add `"rerank": true` to apply the cross-encoder for this single request (slower, ~10s on CPU).

## Scenario 2: mini-eval from scratch to numbers

Retrieval on the raw interview questions is trivially high (they are near-verbatim to their source), so it hides quality differences. This generates a **non-circular** set by paraphrasing questions (and translating to Russian), which forces meaning-based retrieval.

```bash
# 1. generate 20 paraphrased interview questions (+ ru translations) into set "demo"
curl -sX POST localhost:8000/v1/eval/paraphrase -H 'Content-Type: application/json' \
  -d '{"limit":20,"set_name":"demo"}'
# (add "source":"ruby" to slice by topic)

# 2. run the set: answers + auto-judge (one judge job per run, bulk)
curl -sX POST localhost:8000/v1/eval/run -H 'Content-Type: application/json' \
  -d '{"set_name":"demo","run_name":"demo_run"}'

# 3. follow the jobs (and their elapsed seconds)
curl -s "localhost:8000/v1/job?sort_by=id&sort_order=desc&limit=5" | python3 -m json.tool

# 4. once demo_run is judged, read the metrics
docker compose exec rag-lab python -m evals.retrieval_metrics demo_run    # hit@k / MRR
docker compose exec rag-lab python -m evals.generation_metrics demo_run   # faithfulness / relevance / completeness / refusal
```

## Scenario 3: reranking A/B

Run the same set with the cross-encoder on, then compare against the baseline.

```bash
curl -sX POST localhost:8000/v1/eval/run -H 'Content-Type: application/json' \
  -d '{"set_name":"demo","run_name":"demo_rerank","rerank":true}'

docker compose exec rag-lab python -m evals.retrieval_metrics demo_rerank
docker compose exec rag-lab python -m evals.generation_metrics demo_rerank
# compare hit@k / MRR / faithfulness vs demo_run
```
The `paraphrased_ru` set (cross-lingual: ru question over en corpus, where keyword FTS misses and retrieval is vector-only) is where reranking shows the most effect.

## Scenario 4: bring your own questions

```bash
# a plain text file: one question per line; optional "question | source1, source2" for retrieval ground truth
printf 'What is a deadlock?\nHow does TCP differ from UDP? | networking\n' > my.txt

# import (set_name required, language optional; run=true chains eval + judge immediately)
curl -sX POST localhost:8000/v1/questions/import \
  -F "file=@my.txt" -F "set_name=my_set" -F "language=en" -F "run=true"
```

## Scenario 5: inspect answers and jobs

```bash
# bad answers with the judge's reasons
curl -s "localhost:8000/v1/question-log?run_name=demo_run&faithfulness=unfaithful" | python3 -m json.tool
# filters: question_id, text (substring), set_name, run_name, answered, faithfulness, relevance,
#          created_from, created_to, limit, offset, sort_by, sort_order

# retrieval misses for a run: in-corpus questions where the expected source was not retrieved
curl -s "localhost:8000/v1/eval/misses?run_name=demo_run&limit=20" | python3 -m json.tool

# two arms side by side, split by pool, with a paired Wilcoxon over the same questions
curl -s "localhost:8000/v1/eval/compare?runs=arm_a&runs=arm_b" | python3 -m json.tool

# jobs by type/status with elapsed
curl -s "localhost:8000/v1/job?type=eval_run&sort_by=elapsed&sort_order=desc" | python3 -m json.tool
```

## Scenario 6: model lifecycle

```bash
# register a new model (enqueues an Ollama pull)
curl -sX POST localhost:8000/v1/model -H 'Content-Type: application/json' -d '{"name":"qwen2.5:14b"}'
# list models / roles
curl -s localhost:8000/v1/model | python3 -m json.tool
curl -s localhost:8000/v1/role  | python3 -m json.tool
# assign a model to a role (switches at runtime)
curl -sX PUT localhost:8000/v1/role/generation -H 'Content-Type: application/json' -d '{"model_id": 1}'
```

## Scenario 7: prompt versioning

Prompt sources live in `prompts/<purpose>.v<N>.txt` and are seeded into the DB. To ship a new version: add `prompts/generate_answer.v2.txt`, re-run the seed, then activate it.

```bash
docker compose run --rm seed                                  # loads new prompt versions (inactive)
curl -s localhost:8000/v1/prompt | python3 -m json.tool       # find the new version id
curl -sX POST localhost:8000/v1/prompt/<id>/activate          # switch active (deactivates siblings)
```

## Scenario 8: agentic answer and agent vs single-shot

The agent decides its own retrieval (a ReAct tool-calling loop): it can search, refine the query, and multi-hop before answering. `debug=true` returns the full message trace; `max_hops` and `language` are optional.

```bash
# one agentic answer with the trace
curl -sX POST localhost:8000/v1/agent/question -H 'Content-Type: application/json' \
  -d '{"text":"What is a Redis sorted set and when would you use it?","debug":true}' | python3 -m json.tool

# A/B: run the same set through the agent pipeline, then compare to a single_shot run
curl -sX POST localhost:8000/v1/eval/run -H 'Content-Type: application/json' \
  -d '{"set_name":"paraphrased_ru","run_name":"agent_ru","pipeline":"agent"}'

docker compose exec rag-lab python -m evals.retrieval_metrics agent_ru
docker compose exec rag-lab python -m evals.generation_metrics agent_ru
```
Caveat: retrieval hit@k/MRR are computed the same way for both pipelines, but for the agent the source list is a union across hops (recall-flavoured), so read it as a caveat, not a head-to-head with single-shot precision@k. See [experiments.md](experiments.md) for the measured result.

## Scenario 9: parameter series (measure a retrieval lever)

`POST /v1/eval/experiment` queues one run per value of a swept parameter, keeping set, pipeline and language fixed for a clean single-variable comparison. Swept params: `k` (retrieval width, chunks fed to the generator), `max_hops` (agent hop cap), `model` (generator model name; a model absent from the registry is created and pulled, the run waits for it), `variant` (which corpus variant the run reads, so a chunking can be swept like any other parameter), and, for the agent pipeline only, `fallback_policy`, `gate_signal`, `weak_distance` (the coverage gate's distance threshold) and `topic_threshold`. Runs are auto-named `<base>_<param>_<value>` and each enqueues its own judge pass; the worker drains them one at a time, so it is fire-and-forget.

A corpus can be pinned as well as swept. `variant` on `POST /v1/eval/run` and on either
experiment route names the cut every arm reads, and when `variant` is itself the swept
parameter the swept value wins. Without it a run reads the corpus named in the config, and
the snapshot then records that one rather than the one somebody meant.

Comparing cuts rather than answers is the same route with another kind:

```bash
curl -X POST localhost:8000/v1/experiment -H 'Content-Type: application/json' -d '{
  "kind": "retrieval", "dataset": "paraphrased_ru", "sample_size": 100,
  "param": "variant",
  "axes": {"variant": ["baseline", "clean_1024"], "rerank_top": [0, 20]}
}'
```

Four arms, one job, minutes rather than hours: no generation and no judge, only where the
right chunk landed. `param` names the axis the comparison is reported along and has to be
one of the axes. `GET /v1/experiment/{id}` returns each arm's hit@k and MRR plus the
paired delta of every other arm against the first point of that axis, with a bootstrap
interval and the counts of questions that moved either way.

Every delta carries two more fields. `not_comparable` lists what else differs between the
pair besides the axis of record, so an empty list is the record's own statement that the
comparison has one variable. `halves` repeats each delta on two halves of the question
set, split by a hash of the question id and fixed by a seed, so a rule that picks its
winner on half A and reports on half B can be checked against the record instead of
against somebody's memory.

The corpus itself has two routes of its own. `POST /v1/source/{id}/analyze` runs the coverage report over a source (`mode: dry` cuts it in memory and says what the cut would be, `mode: indexed` reads the rows that are actually served) and `GET /v1/source/{id}/report` reads the history back, per variant, oldest first.
`GET /v1/source/compare?variants=baseline&variants=clean_1024` puts the latest verdict of
each variant beside the other and counts the sources whose verdict moved. In the source
listing, `chunks` is every variant's rows and `chunks_in_variant` counts only the cut named
by `ingest_variant`, which is the one the verdict beside it is about. Neither needs embeddings or labelled questions: the metrics come from the text, so a source can be judged the moment it is added and long before anyone writes a question about it.

```bash
# sweep retrieval width k over the agent pipeline (5 runs, each judged)
curl -sX POST localhost:8000/v1/eval/experiment -H 'Content-Type: application/json' \
  -d '{"set_name":"paraphrased_ru","pipeline":"agent","language":"ru","param":"k","values":[1,3,5,7,10]}'

# watch the runs drain
curl -s "localhost:8000/v1/job?type=eval_run&sort_by=id&sort_order=desc&limit=6" | python3 -m json.tool

# per-run numbers once a run is judged
docker compose exec rag-lab python -m evals.generation_metrics paraphrased_ru_agent_<ts>_k05
```
Set temperature to 0 (config `llm.roles.generation`) so the swept parameter is the only variable. For the agent, `context_tokens` (peak per-hop prompt size) is logged in each answer's metrics, so a run also reveals how many answers approach the model's context window.

## Command reference

```bash
# Full reindex (reset docs + rebuild); on an empty DB the index_data job does this at startup
docker compose exec rag-lab python app/main.py --index

# Interactive console (REPL with chat/db/llm/session and all ORM entities auto-loaded, like rails console)
docker compose exec -it rag-lab python app/main.py --console

# CLI eval runner (alternative to the route): python -m evals.runner <set_name> [run_name]

# Unit tests (pure logic, no DB/Ollama)
docker compose exec rag-lab pytest -q

# Dependencies (uv)
uv sync                 # install from uv.lock
uv add <pkg>            # add a dependency

# Rebuild images (after editing Dockerfile / pyproject / uv add)
docker compose build    # rebuilds ALL app services (rag-lab, worker, seed)
# GOTCHA: compose keeps a separate image per build service. After `uv add` rebuild with no args,
#         otherwise worker/seed stay on the old image and crash on ModuleNotFoundError.

# Migrations (dbmate)
docker compose run --rm dbmate up       # up | new <name> | status | down

# Postgres / Ollama
docker compose exec postgresql psql -U postgres -c "\d data_chunks"
docker compose exec ollama ollama list

# Stop (WITHOUT -v! the -v flag drops volumes, including pulled Ollama models)
docker compose down

# Full reset (drop everything, including models): re-pulls and re-indexes itself
docker compose down -v && docker compose up -d
```
