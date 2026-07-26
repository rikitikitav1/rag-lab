# Experiments log

A lab journal of RAG-quality experiments: hypothesis → setup → result → delta → decision. The point is a reproducible, data-driven loop, not intuition.

## Methodology

- **Eval sets** live in the question bank (`set_name`). The discriminating set is `paraphrased_ru`: interview questions paraphrased and translated to Russian, so retrieval must work cross-lingually (ru query over an en corpus, FTS misses, vector-only) instead of matching source text verbatim. Raw interview questions are near-verbatim to their source (hit@k ~99%), so they hide quality differences.
- **Metrics** come from `question_logs` per `run_name`: retrieval (hit@k / MRR against `marked_sources`), generation (faithfulness / relevance / refusal via LLM-as-judge, judge = `qwen2.5:7b`, neutral to the generator to avoid co-hallucination).
- **Isolation**: change one variable at a time; hold the rest constant.
- **Reproducibility**: generation `temperature: 0` during tuning so a metric change is attributable to the change under test, not to sampling noise.
- Each run is one `eval_run` job (answers, bulk) → one `judge_answers` job (verdicts, bulk).

---

## 2026-07-25 — Reranking (cross-encoder) A/B

**Hypothesis:** a cross-encoder reranker over hybrid retrieval (retrieve-wide → rerank → narrow) improves which chunks reach the generator, most on the cross-lingual set where retrieval is vector-only.

**Change:** `bge-reranker-v2-m3` (CPU, in-process) reranks the top-20 candidates down to top-3. Baseline = hybrid RRF only.

**Setup:** sets `paraphrased_ru` / `paraphrased` (100 each); generator `llama3.1:8b`, temp 0.1 (single run each; noisy). Caveat: temp 0.1 → part of any delta is sampling.

**Result:**

| set | metric | base | rerank |
|-----|--------|------|--------|
| ru | MRR | 0.652 | 0.693 |
| ru | hit@k | 80% | 81% |
| ru | faithful | 37 | 54 |
| ru | unfaithful | 29 | 15 |
| en | MRR | 0.630 | 0.625 |
| en | hit@k | 79% | 76% |
| en | faithful | 61 | 65 |

**Delta:** retrieval hit@k/MRR ~flat (en slightly worse); faithfulness up markedly on ru (+17 faithful, −14 unfaithful), small on en.

**Conclusion:** reranking barely moves the file-level hit@k, but improves chunk ordering within the top-k enough to lift faithfulness on hard cross-lingual queries. Cost ~10s/query on CPU. **Decision: default OFF, opt-in flag** (per-request in `/chat/question`, per-run in `/eval/run`) — the latency is not worth it in the general case, but it is available for cross-lingual/noisy workloads.

---

## 2026-07-25 — Generation prompt v2 → v3 (completeness)

**Hypothesis:** the weak axis is relevance (answers judged "partially" rather than "relevant"): the generator answers loosely. A prompt pushing directness and completeness should convert partially → relevant. Retrieval is unchanged, so any delta is the prompt.

**Change:** `generate_answer` v2 (terse: answer from context, cite, language) → v3 (role line + explicit rules: answer directly and fully, cite every factual statement, do not repeat the question, state what is missing on partial answers, describe source conflicts, no speculation).

**Setup:** set `paraphrased_ru` (100); generator `llama3.1:8b`, **temp 0** (deterministic); rerank off; retrieval identical (hit@k confirms). Clean isolation of the prompt.

**Result (ru, temp 0, clean):**

| metric | v2 | v3 | Δ |
|--------|----|----|---|
| relevant | 35 | 57 | +22 |
| partially | 54 | 36 | −18 |
| irrelevant | 11 | 7 | −4 |
| faithful | 38 | 50 | +12 |
| unfaithful | 28 | 24 | −4 |
| hit@k / MRR | 80% / 0.652 | 80% / 0.652 | = |

**Cross-check (en):** relevant 33 → 44 (+11); faithful 61 → 56 (−5). Caveat: the en baseline was temp 0.1, so not a perfectly clean comparison.

**Delta:** relevance up on both languages (ru +22, en +11) — the target shift, and since retrieval and temperature were held, it is attributable to the prompt. Faithfulness up on ru (+12), slight trade-off on en (−5): pushing completeness yields longer answers with more assertions, some of which the judge marks ungrounded.

**Conclusion:** v3 is a clear, reproducible win on relevance across both languages (not a set-specific fluke). **Decision: v3 active.** Follow-up: resolve the en faithfulness dip with a clean temp-0 baseline, and/or add a "no assertions beyond what answers the question" clause to balance completeness against grounding.

---

## 2026-07-25 — Reranking on top of v3 (stacking, clean)

**Hypothesis:** does the cross-encoder reranker stack on the v3 prompt? The earlier rerank A/B was temp 0.1 (noisy); redo cleanly on v3.

**Setup:** set `paraphrased_ru` (100); v3 prompt, **temp 0**; `v3_ru` (rerank off) vs `v3_rerank_ru` (rerank on, top-20 → top-3). Only the reranker toggled.

**Result:**

| metric | v3 | v3 + rerank | Δ |
|--------|----|-------------|---|
| MRR | 0.652 | 0.685 | +0.033 |
| hit@k | 80% | 80% | = |
| relevant | 57 | 61 | +4 |
| irrelevant | 7 | 4 | −3 |
| faithful | 50 | 53 | +3 |
| unfaithful | 24 | 17 | −7 |

**Delta:** everything moves the right way — better ranking (MRR), better grounding (unfaithful −7), better relevance (+4). Modest but consistent, unlike the noisy temp-0.1 run.

**Conclusion:** rerank **stacks positively on v3** on the cross-lingual set, at ~10s/query CPU cost. Confirms the **opt-in default OFF** decision: the quality gain is real and worth enabling for cross-lingual / high-stakes queries, not for latency-sensitive general use.

---

## 2026-07-26 — ReAct agent vs single-shot (a measured loss)

**Hypothesis:** a ReAct agent that decides its own retrieval (search when needed, refine, multi-hop) should answer at least as well as single-shot RAG, and better on questions needing decomposition. New judge axis **completeness** (answer vs the question's reference answer) added specifically because faithfulness/relevance cannot see the agent's expected advantage.

**Change:** agentic pipeline (`agent.run`, ReAct loop over a `search_corpus` tool, `agent_system` prompt) vs single-shot (`chat.answer`, `generate_answer` prompt). Same `paraphrased_ru` set (100), same gold labels, same judge (qwen2.5:7b).

**Result:**

| axis | base_ru_t0 | v3_ru (single-shot) | agent_ru |
|------|-----------|---------------------|----------|
| retrieval hit@k | 80% | 80% | **31%** |
| retrieval MRR | 0.652 | 0.652 | **0.277** |
| faithful | 38 | 50 | **23** |
| unfaithful | 28 | 24 | **55** |
| relevant | 35 | 57 | 47 |
| complete / partially / incomplete | 0/73/27 | 1/85/14 | 1/72/**27** |
| avg hops | — | — | 2.03 (max 3) |

**Diagnosis (the point of the experiment):** the agent **lost on every axis**. Root cause is retrieval, not generation: **54/100 agent answers had zero retrieved sources**. Every question triggered a search (no run stopped at hop 1), so the agent did not skip retrieval — its self-formulated queries **returned empty ~54% of the time**, while single-shot embeds the full question and retrieves at 80%. Handing query formulation to the 8B model collapsed the cross-lingual retrieval that direct embedding gets for free (reformulated queries drift past the distance threshold). Empty context then propagates downstream: unfaithful jumps to 55 (the model answers ungrounded when the search comes back empty). Only 3/100 runs did 3 hops, so there is no decomposition benefit to offset the retrieval loss.

**Note on metrics:** retrieval hit@k/MRR here are computed the same way for both pipelines; for the agent this is really recall-across-hops, not precision@k, so it is not a clean head-to-head (see PROGRESS backlog). In this run the caveat did not matter — the agent barely multi-hops and its searches often return empty, so the number is deflated, not inflated. (An earlier worry that the agent would *inflate* hit@k via recall was wrong; measuring corrected it.)

**Conclusion:** on this workload (cross-lingual, mostly single-fact questions, 8B model, single retrieval tool) the naive ReAct agent is **strictly worse** than single-shot v3. Agentic RAG is not free: giving a weak model control over query formulation destroyed retrieval. **Decision: single-shot v3 stays the default.** Follow-ups (backlog): have the agent search with the original question (or add it alongside its reformulation), and/or relax the distance threshold for tool searches; the agent's value likely needs genuinely multi-part questions and a stronger model, not this set.

**Follow-up (same day) — the loss was a bug, not the agent (the diagnosis above was wrong).** Inspecting the actual tool-call arguments disproved the "query drift past the distance threshold" theory. The empties came from a different place: the 8B model **always filled the optional `category` parameter with a hallucinated ltree path** (`numpy.array_operations`, `databases.amazon`, `angularjs`), and `category ~ lquery` returns zero for paths that do not exist in the taxonomy. The query *text* the model produced was fine (full, on-topic). Even the rare non-empty case was wrong-domain (a `databases.redis` guess returned redis chunks for a reinforcement-learning question). **Fix: remove `category` from the tool schema offered to the model** (the `_search_corpus` function still supports it; it is just not exposed). General tool-design lesson: never offer a free-form parameter the model cannot ground — use an enum (small fixed set) or a discovery tool; here the taxonomy is too large for an enum, so drop it.

**Re-run (`agent_nocat_ru`, category removed):**

| axis | v3_ru (single-shot) | agent_ru (category trap) | agent_nocat_ru (fixed) |
|------|---------------------|--------------------------|------------------------|
| retrieval hit@k | 80% | 31% | **84%** |
| retrieval MRR | 0.652 | 0.277 | **0.773** |
| faithful | 50 | 23 | 37 |
| unfaithful | 24 | 55 | 32 |
| relevant | 57 | 47 | **70** |
| complete / partially / incomplete | 1/85/14 | 1/72/27 | **2/89/9** |
| empty-source answers | — | 54 | **0** |

**Revised conclusion:** "agent is strictly worse" was an artifact of the category bug, not a property of agentic RAG. With the trap removed the agent **beats single-shot on retrieval (84 vs 80, MRR 0.773 vs 0.652), relevance (70 vs 57), and completeness (incomplete 9 vs 14)** — its query reformulation (often ru→en) retrieves better cross-lingually than embedding the raw Russian question. The one axis it loses is **faithfulness (37 vs 50 faithful, 32 vs 24 unfaithful)**: answering more fully from a multi-hop context union yields more assertions the judge marks ungrounded (the completeness-vs-grounding tradeoff, plus a different context denominator than single-shot's top-3). **Takeaway: measuring caught a bug the naive read would have shipped as a conclusion.** Open follow-up: close the faithfulness gap (tighter grounding instruction; or judge the agent against a deduped/narrowed context).

---

## 2026-07-26 — Corpus ablation: disable developer-roadmap

**Hypothesis:** several `paraphrased_ru` misses are "right topic, wrong repo" — the generalist `developer-roadmap` corpus overlaps many domains and its chunks outrank the true single-source gold chunk. If none of the eval's gold sources live in roadmap, disabling it should raise hit@k rather than lower it.

**Change:** runtime source toggle (`PUT /v1/source/3 {active:false}`), no re-index. `hybrid_search` filters chunks to active sources only. One variable: roadmap in vs out.

**Setup:** set `paraphrased_ru` (100); retrieval identical otherwise; `base_ru_t0` (roadmap on) vs `no_roadmap_ru` (roadmap off). Retrieval-only comparison.

**Result:**

| metric | roadmap on | roadmap off | Δ |
|--------|-----------|-------------|---|
| hit@k | 80% | 90% | +10pp |
| MRR | 0.652 | 0.832 | +0.180 |
| misses | 20 | 10 | −10 |

**Delta:** large. Removing one corpus lifted hit@k by 10 points and MRR by 0.18. The gold sources for this eval set are not in roadmap, so its chunks were pure distractors ranking above true sources.

**Conclusion:** corpus composition is a first-order retrieval lever — a broad, topically-overlapping source hurts a single-source-labeled eval more than reranking or prompt tuning helped. Confirms the earlier miss diagnosis (disambiguation, not corpus gap). **Decision: drop developer-roadmap from the corpus entirely.** It is a generalist roadmap index whose chunks are shallow and topically diffuse; it added retrieval noise across every domain without being a canonical source for any of them. The showcase corpus is cleaner as a set of focused sources (interview banks, system-design-primer, redis-doc). New default retrieval on `paraphrased_ru` is **90% / MRR 0.832** (previously 80% / 0.652). The source-active toggle proved out as the ablation tool that justified the removal. Follow-up: multi-source gold labels would quantify how much of any residual "distraction" is genuinely wrong vs adjacent-correct.
