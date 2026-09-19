# REST API

Full interactive reference in Swagger at `/docs`.

`POST /v1/chat/question` and `POST /v1/job` refuse a field they do not know with a 422 rather than dropping it.

List endpoints (`/v1/model`, `/v1/prompt`, `/v1/job`, `/v1/question-log`) share pagination: `limit` (default 100, max 1000), `offset`, `sort_by`, `sort_order` (`asc`/`desc`, default `desc`).

Health:
- `GET /liveness`, `GET /readiness` (names each role whose engine does not answer)
- `GET /v1/health/stand` (what the stand is right now: the card, which models are resident and how much VRAM each holds, the window the server actually serves against the declared one, the live queue, role drift between config and database, the corpus variant and the search depth per variant. Readable while a run competes with it, so a run that answers slowly can be diagnosed without stopping it)

Chat and search:
- `POST /v1/chat/question` (full RAG answer; optional `rerank` flag; optional `language` override `ru`/`en`)
- `POST /v1/chat/fast_question` (retrieval only, no generation)
- `POST /v1/agent/question` (agent answer; optional `max_hops`, `language`, `fallback_policy`, and `debug` for the full message trace)
- The answering doors never hand the card themselves: while another engine holds it they queue the handover and answer 503 with `Retry-After`; an answer that would need two engines of the card is a 409 (the chat with `rerank: true` in the default layout, where the generator sits on ollama and the reranker on `vllm-rerank`).
- `GET /v1/categories` (category tree with chunk counts)

<details>
<summary>Diagram: Single-shot flow: hybrid retrieval, threshold, optional rerank</summary>

![Single-shot flow: hybrid retrieval, threshold, optional rerank](diagrams/single_shot_flow.svg)

</details>

Engines and models:
- `GET /v1/engine`, `POST /v1/engine` (name, `kind` (`ollama`/`vllm`/`openai_compatible`), `env_prefix`, `placement` (`gpu`/`cpu`/`gpu+cpu`/`remote`); a vLLM on the card without sleep routes is a 422), `PATCH /v1/engine/{id}` (`placement`, a 409 while the server runs where the row said, and a cloud's `balance_reader`), `GET /v1/engine/{id}/live` (asks the engine itself, in seconds), `DELETE /v1/engine/{id}` (409 while models point at it). The response shows the address resolved from the environment; a prefix with no address is a 400 before the row is written.
- `GET /v1/model`, `GET /v1/model/{id}`, `POST /v1/model` (`engine` by name or `engine_id`; required on the default stand, where the seed registers two ollama engines, `ollama` and `ollama-cpu`; an engine that pulls gets a pull job, one that does not is asked whether it already serves the name: `ready`, 422 if it serves another, 503 if it does not answer), `PATCH /v1/model/{id}` (`quant` and the hub key of the `weights` when the server cannot say them, the `answer_parser`, and `options` of its own over the role's), `POST /v1/model/{id}/load` (202 with the queued `hand_card` job, and a load already waiting answers a second ask with its job; 422 when a vLLM serves another model, 409 for a vLLM on the processor), `DELETE /v1/model/{id}` (501 on a remote engine; 409 if the model is assigned to a role, shares its weights with another row, or is served by a running vLLM). One name may live on two engines, so the list and the response name the engine, along with `quant`, `size_bytes` and the weights row, all read off the server on the pull rather than typed.
- `GET /v1/role`, `PUT /v1/role/{role}` (assign a model to a role; the model is asked whether it can do the job first, and a 400 says what it lacks. An asleep vLLM with no tool-call probe recorded answers 202 with a `hand_card` job that wakes it, probes it and then seats the role; an engine that does not answer is a 503. `anyway: true` insists, which is how a model the server describes wrongly is still seated)
- `GET /v1/source`, `PUT /v1/source/{id}` (enable/disable a corpus source; disabled sources are excluded from retrieval at runtime, no re-index - ablation / source-of-truth scoping)

Prompts:
- `GET /v1/prompt`, `GET /v1/prompt/{id}`, `POST /v1/prompt`, `POST /v1/prompt/{id}/activate`, `DELETE /v1/prompt/{id}`

Eval platform:
- `POST /v1/eval/paraphrase` (generate a paraphrase set), `POST /v1/eval/run` (run a set → judge; `pipeline: single_shot|agent`, per-run `rerank`, `k` retrieval-width, `max_hops`, `fallback_policy`, `gate_signal`, `weak_distance`, `topic_threshold`, `orchestrator`, `variant` (which cut of the corpus the run reads) and `model` (generator) overrides, plus `allow_cpu` for a run that means to measure the processor; config only sets the defaults)
- `POST /v1/eval/experiment` (batch a parameter series: `param` (`k`, `max_hops`, `model`, `variant`, `orchestrator`, `fallback_policy`, `gate_signal`, `weak_distance` or `topic_threshold`) swept over `values`, one auto-named run per value, each judged; set/pipeline/language stay fixed for a clean single-variable comparison; a `model` value absent from the registry is created and pulled, the run waits for it)
- `GET /v1/eval/misses?run_name=X` (retrieval misses for a run: in-corpus questions where the expected source was not retrieved, with expected vs retrieved)
- `GET /v1/eval/compare?runs=A&runs=B` (arms side by side split by pool: in-corpus, out-of-corpus, off-domain, rejected; per arm the judged axes, how often the answer came from a remote tool against the corpus, how often the coverage gate fired, latency avg/p50 and the outcome histogram; per pair of arms a paired Wilcoxon plus a bootstrap interval over the same questions, so a difference is reported with its size and its uncertainty instead of two averages)
- `POST /v1/questions/import` (upload a questions file, ≤5 MB; optional chained run)
- `GET /v1/questions?set_name=&language=&pool=&limit=&offset=` (the questions themselves, one row each: id, pool, text, reference and marked sources; where a run's `question_ids` come from)

<details>
<summary>Diagram: Eval pipeline</summary>

![Eval pipeline](diagrams/eval_pipeline.svg)

</details>

A single run does not loop per question: it goes through phases so each stage owns the GPU alone, which is what makes reranking affordable in bulk. Per question the loop needed the embedder, then the reranker, then the generator, and the three do not fit in 8 GB together, so ollama evicted and reloaded a model on every single question. Phases cost a few model swaps per run instead of two per question, and the run gives the card back when it ends, so a queue of runs on different generators does not end up holding two of them at once. The reranker finally gets a real batch: 31 s for 100 questions on the card against about 16 min on CPU. That 310 ms a question is the whole phase, model load and retrieval included; the reranking itself is the 86 ms measured in `datasets/measurements/rerank_latency.json`.

<details>
<summary>Diagram: Phases inside one eval run</summary>

![Phases inside one eval run](diagrams/phased_run.svg)

</details>

Measured on 100 questions with reranking on: **2092s → 652s (3.2x)** while `hit@5` and `MRR` stayed identical to the third decimal. Most of the win did not come from batching, it came from noticing that the card was never actually released between phases, so ollama had been loading the generator as 26 layers of 33. The full story, including what the batch alone did *not* buy, is in [the journal entry](experiments/2026-08-24_phased-eval-runs-and-the-empty-cache.md).

Experiments (first-class entity over the raw sweep route):
- `POST /v1/experiment` (creates the experiment - dataset + deterministic seed-based sample / procedure snapshot / varied param - and enqueues the run series), `GET /v1/experiment` (filtered list), `GET /v1/experiment/{id}`, `PUT /v1/experiment/{id}/conclusion`, `POST /v1/experiment/{id}/arms` (a rejudge only: copies more arms of the same answers and re-judges them, from `aggregated` and back to `running`; the arms are named one by one rather than as a grid, the row cap counts what the experiment already holds, and an arm added later is built on the sample, the control size and the seed the experiment was created with)
- state machine `draft → running → aggregated → concluded` (+ `failed`); when the last judge job of the series finishes, the aggregator computes per-value metrics and an RRF composite over five axes (the three judged ones, the off-domain refusal rate and the supported rate) and stores them in `results` (retrieval hit@k/MRR reported per value but kept out of the fusion: hit@k is monotonic in `k`, it would confound the composite)
- results carry **paired significance statistics**, not just point estimates: for the winner vs every other value, per axis - mean paired delta (same question in both runs), bootstrap 95% CI (10000 resamples, the one resampling the whole stand draws with) and a Wilcoxon signed-rank p-value, plus Holm step-down flags over the test family the record itself names, so the JSON says what survives multiple-comparison correction and against which family it was corrected

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

## The queue

Any job type can be queued through one door: `POST /v1/job` with `{"type": ..., "options": {...}}`. It and every door that queues a job of its own (`/v1/eval/*`, `/v1/source/{id}/analyze`) answer with the whole job row: `job_id`, `type`, `queue`, `status`, `options`, `apply_since`, `created_at`.
What each type accepts is a model per type in `app/job_specs.py`, checked when the job is queued,
whichever door or script queues it, and again when the worker takes it, so a row written straight
into the table meets the same refusal. A door of its own is for work done before the enqueue rather
than for checking: `/eval/rejudge` copies a run, `/eval/guest-axes` answers on a property of the
runtime. The lane belongs to the type, not to the caller.

Every type the queue knows, what it does and what it takes:

| type | what it does | the options it reads | roles it takes the card for | where the result lands |
|---|---|---|---|---|
| `pull_llm_model` | pulls weights into an engine | `name`, `engine_id` | none (io lane) | the model row goes `ready` |
| `delete_llm_model` | removes weights from an engine | `name`, `engine_id` | none (io lane) | the model row |
| `index_data` | cuts a corpus variant and embeds it | `variant`, `source` | embedding | `data_chunks` of that variant |
| `build_vector_index` | builds the hnsw index of a variant | `variant` | none | the index |
| `analyze_source` | reads one source and reports its ingest quality | `source`, `variant`, `mode` | none | `data_sources.ingest_quality` |
| `embed_questions` | embeds a question set | `set_name` | embedding | `questions.embedding` |
| `paraphrase_questions` | writes paraphrases of a set | `set_name`, `limit` | paraphrasing | new questions of the paraphrased set |
| `build_veto_set` | builds the veto set from a source set | `set_name`, `limit` | paraphrasing | the veto question set |
| `eval_run` | answers a set through a pipeline and records a row per question | the run's whole snapshot (`run_name`, `set_name`, `pipeline`, `k`, `grade_chunks`, …) | generation, embedding, reranking | `question_logs` of that run |
| `judge_answers` | scores our three axes over a run's rows | `run_name`, `axes`, `width` | judging | verdicts on the rows |
| `judge_guest_axes` | scores the standard's axes over a subsample | `run_name`, `sample`, `seed` | ragas, ragas_embedding | guest verdicts on the rows |
| `judge_language` | restates a row in two languages and scores both | `run_name`, `panel` | judging, generation | a measurement file |
| `compare_retrieval` | measures a grid of retrieval arms of one experiment | `experiment_id` | reranking | the experiment's `results` |
| `grade_candidates` | grades frozen candidates chunk by chunk, no generator and no judge | `candidates` (the frozen file), `form`, `top`, `limit`, `sample`, `seed`, `shuffle`, `prompt_version`, `name` | grading | a measurement file with a verdict and its probability per chunk, and the curve of both arms over the cuts |
| `check_mcp_health` | asks a remote integration whether it answers | `integration_id` | none (io lane) | the integration's health |
| `hand_card` | wakes an engine, probes it and seats a role | `engine_id`, `model`, `seat` | the seat it hands | the card and the role row |

To judge a run in place: `POST /v1/eval/judge` with `{"run_name": "<run>"}` queues our three axes
over the rows that still owe them, and refuses with 404 when the run holds no answered row or owes
nothing. `POST /v1/eval/guest-axes` queues the standard's axes over a run, and takes `sample` and
`seed`: the guests cost between six and eight times our three axes a row, so they calibrate on a drawn
subsample rather than riding every run. They ride the same judging pass and land on the same row,
so a correlation between the two is a join rather than a comparison of two copies, and they stay
out of the composite and out of our axes' Holm family: a ruler used to check a ruler is not a
fourth measure of quality. `POST /v1/eval/language-probe` restates a row's own answer
in two languages and scores both against the same context, which asks whether our faithfulness
prompt loses a point on Russian. To re-judge one run without making an experiment of it: `POST /v1/eval/rejudge` with
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
        "comparisons": {"5_vs_10": {"faithfulness": {"mean_delta": 0.19, "ci95": [-0.17, 0.57], "p": 0.36692741, "n": 100, "holm_threshold": null, "significant_raw": false, "significant_holm": false}, "...": "..."}},
        "method": "holm", "alpha": 0.05, "tests": 15, "family": "every pair of the grid on every axis",
        "population": "every row of both runs that pairs by question_id, all pools blended: ..."
      },
      "rows_by_population": {"in_corpus_and_answered_in_every_arm": 91, "by_run": {"...": "..."}}
    }
  }
}
```

The pairwise block is what separates a result from noise: an early version of this bench "concluded" k=5 beats k=10 from a composite-score gap in the third decimal - the paired test shows that comparison is a coin flip (p=0.37), and the corrected flags mark which of the 15 grid tests survive at all. See [docs/experiments.md](experiments.md) for the cases where this reversed our own verdicts.

Before a grid starts, `scripts/preflight_grid.py` refuses the ways a run silently stops meaning
anything: an edited tree, a worker running yesterday's code, a model that spilled to the CPU, a
corpus that no longer cuts into the rows it holds, a search depth the planner has quietly stopped
walking the index at. `--verify` checks a finished run instead. Every one of those failures produces
a completed run with plausible numbers and no error anywhere, which is why the check exists rather
than a test. What each of the eighteen refuses and which incident put it there:
[docs/preflight.md](preflight.md).

Record the takeaway with `PUT /v1/experiment/{id}/conclusion` and the experiment becomes a self-contained artifact: what was varied, on what data, the numbers, the verdict.

Observability:
- `GET /v1/question-log`, `GET /v1/question-log/{id}` (the row carries what it was asked and what it read: `question_text`, `reference_answer` and `contexts`, the chunks as elements rather than the string they were joined into, because that join cannot be undone. Filters incl. `pipeline`, `faithfulness`/`relevance`/`completeness`, `run_name`; and over the recorded snapshot: `rerank`, `rerank_device`, `phased`, `empty_retrieval`, `max_distance`, `answered_via_remote` - so "show me every answer where the corpus returned nothing" is one request; detail with context)
- `GET /v1/job`, `GET /v1/job/{id}` (jobs + elapsed; filters incl. `type`, `status` and `run_name`, which the MCP console could already do and the route could not), `POST /v1/job/{id}/cancel` (cancels the job and its dependent judge). A running job stops between rows: an eval run, a retrieval comparison and all three judging passes read the cancellation, and a judging pass that was cancelled does not queue its next sweep
- `POST /v1/job/cancel` (cancel a whole run or job type at once: cancelling id by id through a paginated listing is how a supposedly stopped eval quietly kept running). A `type` with no `run_name` is refused with 400 unless the call also passes `every: true` and means it. Either door takes a run's judge down with the run

The complete reference is Swagger at `http://localhost:8000/docs`; the scenarios that use these routes are in [use_cases.md](use_cases.md).

## What two arms must share

A difference between two arms is read only after the comparison has checked that both were measured
by one instrument. `GET /v1/eval/compare` (and the ops tool `compare_pools`) answers each check with a
field. For the first nine rows, the judge and retrieval, `residency.read_this_first` names the first
that failed, in this order; the last five are named by their own fields. Arms must also share their
questions: a run that cannot be paired question by question is a 409.

| What must match | Field in the report | What a mismatch means |
|---|---|---|
| the judge's engine, by name | `residency.one_engine_name` | not comparable: two engines take the same host and port in turn, and only the name tells them apart |
| the judge's engine, by address | `residency.one_engine` | not comparable: batching, kernels and quantisation all differ |
| retrieval: the same sources in the same ranks on shared questions | `residency.one_deterministic` | the pipeline changed underneath the arms, unless retrieval is the treatment |
| the judge's prompt version per axis | `residency.one_judge_prompt` | two rulers: the contrast measures the prompt |
| the parser that cut the judge's reply | `residency.one_judge_parser` | the scores were read from different texts |
| the rule the judge's JSON was decoded by | `residency.one_judge_grammar` | the contrast measures the grammar ([17% of verdicts moved](experiments/2026-09-13_the-judge-that-looped-on-whitespace.md)) |
| the judge's temperature and seed | `residency.one_judge_sampler` | the contrast measures the sampler |
| the judge's output budget | `residency.one_judge_budget`, `judge_cut_by_run` | matters only where a verdict ended on the limit |
| the judge's residency, one load of the model | `residency.one_residency`, `remote_judge` | across a reload the pair needs its own floor; a remote judge has no residency at all |
| the model client of the answering path | `pools.*.pairs[].isolates_orchestrator` | the pair changed the client as well as its treatment |
| the generator's repetition penalty | `one_answering_penalty` | the contrast also measures the penalty ([1.05 against 1.1](experiments/2026-09-13_the-penalty-nobody-asked-for.md)) |
| the answering engines | `answering_engines_by_run` | named, not refused: two generators on two engines is what a pair of arms is for |
| the broker key of a cloud role | `residency.one_broker_key` | one instrument billed to two accounts: the money is read per key |
| the rule outcomes are read with | `outcome_rule` | refusal shares read with another rule differ ([rule 2](experiments/2026-09-13_a-refusal-the-rule-could-not-read.md)) |

All of these passing is necessary, not sufficient: two arms with identical rows, order and prompt
still differed on 4 of 50 rows, so a pair is read against its own floor rather than against a zero.
A rejudge adds `same_answers`, which says the arms scored byte-identical answers.

## The order jobs run in

The worker runs one thread per lane (`WORKER_QUEUES`, by default `default,io`), and each thread takes
the next job of its own lane by these rules, top to bottom:

| Rule | What it does | Where it lives |
|---|---|---|
| the lane | model pulls and deletes and the MCP health check go to `io`, every other type to `default`, so a download never waits behind a run | `job_specs.LANES` |
| the priority | `hand_card` at −2 goes first, every type at 0 next, and the judges (`judge_answers`, `judge_guest_axes`, `judge_language`) at 10 last, since they wait for the runs they judge | `job_specs.PRIORITY` |
| the wait | a job that has waited more than 30 minutes rises to −1: ahead of everything but a handover | `job_specs.STARVED_AFTER_MINUTES` |
| the time | within one rank, the earlier `apply_since` goes first: when the job was queued, or when a retry or a deferral set it | `job_queue._turn` |
| a retry | a job that fails with anything but a final error runs again, three attempts in all, the n-th retry n × 60 s later | `worker.MAX_ATTEMPTS` |
| a deferral | a job waiting for what it needs is set back by the delay it names, an hour in all, then fails | `worker.MAX_DEFERRED_SECONDS` |
| a restart | a worker coming up returns its own running jobs to the queue, so a restart loses at most the call in flight | `job_queue.requeue_stale` |

To move a job that has not started, cancel it (`POST /v1/job/{id}/cancel`) and queue it again: it takes
a new place by the rules above. A job that holds its lane holds it to its end, so a cloud pass slowed
by its broker keeps every other `default` job waiting behind it.
