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
Returns the answer, cited sources (with vector/keyword ranks and score), and token/time metrics. Add `"rerank": true` to apply the cross-encoder for this single request (slower, ~10s on CPU).

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
docker compose exec rag-lab python -m evals.generation_metrics demo_run   # faithfulness / relevance / refusal
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
