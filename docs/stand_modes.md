# Stand modes: how to switch the stand, and how to come back

This page holds one question: how to put the stand into a given layout of engines and roles, what
that costs, and how to leave it. What exists (services, profiles, variables) is in the README
tables; the scenarios in [use_cases.md](use_cases.md) run on a stand that is already up and point
here for the mode they need. Measured numbers live in the [experiment journal](experiments.md); this
page gives only the constants of the device and orders of magnitude.

**Where you are, read in this order, every time:** the ops MCP tool `engines` (who holds the card,
which vLLM is asleep, which engine answers), then `GET /v1/health/stand` (where each role's model
sits, which roles are down, the queue), then the preflight ([preflight.md](preflight.md)) before a
run that will be quoted.

```bash
curl -s localhost:8000/v1/health/stand | python3 -m json.tool
# ids for the commands below: one line per model, its engine and its role-relevant name
curl -s "localhost:8000/v1/model?limit=100" | python3 -c \
  'import json,sys; [print(m["id"], m["engine"], m["name"], m["status"]) for m in json.load(sys.stdin)]'
```

Two rules hold in every mode. The card is handed only through the queue (`hand_card`, or a job taking
it for its own call): a sleep or a wake sent to a server by hand is invisible to a pass in flight.
And a mode with a profile is left in the opposite order it was entered: seat the role back first,
then stop the service, or the role reads as down and `/readiness` turns `degraded`.

## 1. Default

```bash
docker compose up -d
```

The generator (`llama3.1:8b`), the embedder (`bge-m3`) and the paraphraser (`gemma2:9b`) sit on
`ollama` on the card; the judge (`Qwen/Qwen2.5-7B-Instruct-AWQ`) on `vllm`, which takes the card first
at start and is put to sleep by the bootstrap before ollama loads a role. A run and its judging pass
hand the card between the two: to vLLM in about a second, back to ollama in a few seconds with the
load. The reranking role is seated on `vllm-rerank` but its service is down until mode 2; that is
not counted as a fault while no run asks for reranking.

What the record says: the judge's axes carry `engine_name: vllm` and a residency read off the
server's process start (`vllm /metrics process start`). Numbers judged this way compare with each
other; numbers judged before 11.09 used another judge on another engine and compare only in mode 6.

## 2. With reranking

```bash
docker compose --profile rerank up -d vllm-rerank
```

The role is already seated by the bootstrap from `config.yaml`. A run asks for it with `"rerank":
true`, and the agent's gate uses it at `gate_signal: cross_encoder` or `either`. The server takes a
0.3 share of the card (`VLLM_RERANK_GPU_UTIL`); in a phased run the embedder answers first, the
reranker takes the card for one scoring pass, and the card goes back to ollama for generation.

What the record says: the run snapshot names the reranking role's engine among the engines per role.

Leave it: stop asking for reranking, then `docker compose stop vllm-rerank`.

## 3. The embedder on vLLM

```bash
docker compose --profile embed up -d vllm-embed
curl -sX POST localhost:8000/v1/engine -H 'Content-Type: application/json' \
  -d '{"name":"vllm-embed","kind":"vllm","env_prefix":"VLLM_EMBED","placement":"gpu"}'
curl -sX POST localhost:8000/v1/model -H 'Content-Type: application/json' \
  -d '{"name":"BAAI/bge-m3","engine":"vllm-embed"}'
curl -sX PUT localhost:8000/v1/role/embedding -H 'Content-Type: application/json' -d '{"model_id": <id>}'
```

A vector remembers the embedder that made it (`embedded_by`, `model@engine`), and a search over a
variant holding another embedder's vectors is refused rather than answered from two geometries. So
this mode needs its own corpus variant, declared in `config.yaml` under `corpus.variants` and
indexed after the switch, and the questions embedded again:

```bash
curl -sX POST localhost:8000/v1/job -H 'Content-Type: application/json' \
  -d '{"type":"index_data","options":{"variant":"<the new variant>"}}'
curl -sX POST localhost:8000/v1/job -H 'Content-Type: application/json' \
  -d '{"type":"embed_questions","options":{}}'
```

With the embedder and the generator on two engines of the card, the chat answers 409: it does not
hand the card between them within one answer. A run does, one handover per call of the other role.

What the record says: every chunk and question carries `BAAI/bge-m3@vllm-embed`, and retrieval
numbers compare only between runs over the same embedder's vectors.

Leave it: seat `embedding` back on `bge-m3` on `ollama`, then `docker compose stop vllm-embed`. The
variant indexed by vLLM stays refused under ollama's embedder until it is removed or indexed again.

## 4. On the processor

`ollama-cpu` is always up; vLLM on the processor comes with its profile:

```bash
free -g && docker stats --no-stream
docker compose --profile cpu up -d vllm-cpu
curl -sX POST localhost:8000/v1/engine -H 'Content-Type: application/json' \
  -d '{"name":"vllm-cpu","kind":"vllm","env_prefix":"VLLM_CPU","placement":"cpu"}'
```

A role on a processor engine never takes the card. The price is host memory, not the card:
`vllm-cpu` holds its whole model there (`Qwen/Qwen2.5-7B-Instruct` in fp16 is about 15 GB of
weights), and the sleeping judge keeps about 13 GB of host memory too. Before starting it, unload
the models of `ollama-cpu` and keep free at least twice the weights plus room for the desktop; a
start without that check once took the whole session down.

What the record says: the role's engine has `placement: cpu`, and its answers come from other
kernels than the card's, so they are a different instrument, not a slower copy of the same one.

Leave it: seat the roles back on their card engines, then `docker compose stop vllm-cpu`.

## 5. Generation on vLLM

```bash
curl -sX PUT localhost:8000/v1/role/generation -H 'Content-Type: application/json' \
  -d '{"model_id": <id of Qwen/Qwen2.5-7B-Instruct-AWQ on vllm>}'
```

The agent's generator must return tool calls, and `vllm` is started with a tool-call parser for it.
An asleep server has no probe to answer with, so the door answers 202 with a `hand_card` job that
wakes it, asks it once and seats the role; follow it with `GET /v1/job/{id}`. The generator and the
judge are then one process, and nothing is handed between answering and judging. The embedder stays
on ollama, so the chat answers 409 as in mode 3 and a run hands the card per call.

What the record says: the generator's engine is `vllm`, and its window is the server's
`max_model_len`, which refuses a longer prompt instead of cutting it.

Leave it: seat `generation` back on `llama3.1:8b` on `ollama`.

## 6. The judge on ollama: the ruler of arcs 1 to 4

```bash
curl -sX PUT localhost:8000/v1/role/judging -H 'Content-Type: application/json' \
  -d '{"model_id": <id of qwen2.5:7b on ollama>}'
```

Every number judged before 11.09 was judged by `qwen2.5:7b` on ollama, and this is the only mode
that reproduces them. The judge and the generator then share one ollama, so each run evicts the
judge and its residency starts again at every judging pass; its noise floor across reloads is in the
journal.

What the record says: `engine_name: ollama` and a residency read off `ollama /api/ps and the queue`.

Leave it: seat `judging` back on `Qwen/Qwen2.5-7B-Instruct-AWQ` on `vllm`.

## 7. The card is stuck

The one mode nobody enters on purpose. Signs: the chat answers 503 with `Retry-After` and keeps
answering it; a job ends with an error that names the card.

```bash
curl -s "localhost:8000/v1/job?type=hand_card&status=error&limit=5" | python3 -m json.tool
```

- `... did not let go of the card in 60s`: an engine kept a model on the card; `engines` names it.
- `... did not wake in 60s`: the vLLM could not take the card back, usually because something else
  still holds memory on it.
- `... is down; the card stays where it is`, `... is unknown; ...`: the target server is stopped or
  silent, and nothing was let go. `docker compose up -d <service>` for a stopped one; a silent one is
  restarted with `docker compose restart vllm`, and it comes back awake on the card.
- `... loaded on ollama, but not whole on the card`: the model landed half on the processor; the
  card is short of memory, and `engines` with `nvidia-smi` say who holds it.
- `nvidia-smi` inside a container answers `Failed to initialize NVML: Unknown Error`, and jobs end
  with `not whole on the card`: the container lost the card. With the card given through CDI a
  `systemctl daemon-reload` no longer does this; if it happens anyway, `docker compose restart
  <service>` gives the card back, and `docker info` should list the NVIDIA CDI devices (README,
  Quickstart).
- `vllm` exits at start with `Free memory on device ... is less than desired GPU memory
  utilization`, and the API and the worker stay `Created` behind it: the stack was recreated while
  ollama still held a model, since the ollama container is not recreated with it and keeps its
  models loaded. The API is down, so the queue cannot hand anything: unload ollama's models through
  its own port (`curl localhost:11434/api/generate -d '{"model":"<name>","keep_alive":0}'` for each
  name in `curl localhost:11434/api/ps`), then `docker compose up -d` again.

Back to the default: `POST /v1/model/{id of the judge}/load` hands the card to the judge through the
queue, and the next job that needs ollama takes it back the same way.

## After MR 3: a cloud engine

Written with the cloud engine.
