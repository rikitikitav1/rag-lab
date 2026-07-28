# Experiments log

A lab journal of RAG-quality experiments: hypothesis → setup → result → delta → decision. The point is a reproducible, data-driven loop, not intuition.

## Methodology

- **Eval sets** live in the question bank (`set_name`). The discriminating set is `paraphrased_ru`: interview questions paraphrased and translated to Russian, so retrieval must work cross-lingually (ru query over an en corpus, FTS misses, vector-only) instead of matching source text verbatim. Raw interview questions are near-verbatim to their source (hit@k ~99%), so they hide quality differences.
- **Metrics** come from `question_logs` per `run_name`: retrieval (hit@k / MRR against `marked_sources`), generation (faithfulness / relevance / refusal via LLM-as-judge, judge = `qwen2.5:7b`, neutral to the generator to avoid co-hallucination).
- **Isolation**: change one variable at a time; hold the rest constant.
- **Reproducibility**: generation `temperature: 0` during tuning so a metric change is attributable to the change under test, not to sampling noise.
- Each run is one `eval_run` job (answers, bulk) → one `judge_answers` job (verdicts, bulk).

---

## 2026-07-25 - Reranking (cross-encoder) A/B

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

**Conclusion:** reranking barely moves the file-level hit@k, but improves chunk ordering within the top-k enough to lift faithfulness on hard cross-lingual queries. Cost ~10s/query on CPU. **Decision: default OFF, opt-in flag** (per-request in `/chat/question`, per-run in `/eval/run`) - the latency is not worth it in the general case, but it is available for cross-lingual/noisy workloads.

---

## 2026-07-25 - Generation prompt v2 → v3 (completeness)

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

**Delta:** relevance up on both languages (ru +22, en +11) - the target shift, and since retrieval and temperature were held, it is attributable to the prompt. Faithfulness up on ru (+12), slight trade-off on en (−5): pushing completeness yields longer answers with more assertions, some of which the judge marks ungrounded.

**Conclusion:** v3 is a clear, reproducible win on relevance across both languages (not a set-specific fluke). **Decision: v3 active.** Follow-up: resolve the en faithfulness dip with a clean temp-0 baseline, and/or add a "no assertions beyond what answers the question" clause to balance completeness against grounding.

---

## 2026-07-25 - Reranking on top of v3 (stacking, clean)

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

**Delta:** everything moves the right way - better ranking (MRR), better grounding (unfaithful −7), better relevance (+4). Modest but consistent, unlike the noisy temp-0.1 run.

**Conclusion:** rerank **stacks positively on v3** on the cross-lingual set, at ~10s/query CPU cost. Confirms the **opt-in default OFF** decision: the quality gain is real and worth enabling for cross-lingual / high-stakes queries, not for latency-sensitive general use.

---

## 2026-07-26 - ReAct agent vs single-shot (a measured loss)

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
| avg hops | - | - | 2.03 (max 3) |

**Diagnosis (the point of the experiment):** the agent **lost on every axis**. Root cause is retrieval, not generation: **54/100 agent answers had zero retrieved sources**. Every question triggered a search (no run stopped at hop 1), so the agent did not skip retrieval - its self-formulated queries **returned empty ~54% of the time**, while single-shot embeds the full question and retrieves at 80%. Handing query formulation to the 8B model collapsed the cross-lingual retrieval that direct embedding gets for free (reformulated queries drift past the distance threshold). Empty context then propagates downstream: unfaithful jumps to 55 (the model answers ungrounded when the search comes back empty). Only 3/100 runs did 3 hops, so there is no decomposition benefit to offset the retrieval loss.

**Note on metrics:** retrieval hit@k/MRR here are computed the same way for both pipelines; for the agent this is really recall-across-hops, not precision@k, so it is not a clean head-to-head (see PROGRESS backlog). In this run the caveat did not matter - the agent barely multi-hops and its searches often return empty, so the number is deflated, not inflated. (An earlier worry that the agent would *inflate* hit@k via recall was wrong; measuring corrected it.)

**Conclusion:** on this workload (cross-lingual, mostly single-fact questions, 8B model, single retrieval tool) the naive ReAct agent is **strictly worse** than single-shot v3. Agentic RAG is not free: giving a weak model control over query formulation destroyed retrieval. **Decision: single-shot v3 stays the default.** Follow-ups (backlog): have the agent search with the original question (or add it alongside its reformulation), and/or relax the distance threshold for tool searches; the agent's value likely needs genuinely multi-part questions and a stronger model, not this set.

**Follow-up (same day) - the loss was a bug, not the agent (the diagnosis above was wrong).** Inspecting the actual tool-call arguments disproved the "query drift past the distance threshold" theory. The empties came from a different place: the 8B model **always filled the optional `category` parameter with a hallucinated ltree path** (`numpy.array_operations`, `databases.amazon`, `angularjs`), and `category ~ lquery` returns zero for paths that do not exist in the taxonomy. The query *text* the model produced was fine (full, on-topic). Even the rare non-empty case was wrong-domain (a `databases.redis` guess returned redis chunks for a reinforcement-learning question). **Fix: remove `category` from the tool schema offered to the model** (the `_search_corpus` function still supports it; it is just not exposed). General tool-design lesson: never offer a free-form parameter the model cannot ground - use an enum (small fixed set) or a discovery tool; here the taxonomy is too large for an enum, so drop it.

**Re-run (`agent_nocat_ru`, category removed):**

| axis | v3_ru (single-shot) | agent_ru (category trap) | agent_nocat_ru (fixed) |
|------|---------------------|--------------------------|------------------------|
| retrieval hit@k | 80% | 31% | **84%** |
| retrieval MRR | 0.652 | 0.277 | **0.773** |
| faithful | 50 | 23 | 37 |
| unfaithful | 24 | 55 | 32 |
| relevant | 57 | 47 | **70** |
| complete / partially / incomplete | 1/85/14 | 1/72/27 | **2/89/9** |
| empty-source answers | - | 54 | **0** |

**Revised conclusion:** "agent is strictly worse" was an artifact of the category bug, not a property of agentic RAG. With the trap removed the agent **beats single-shot on retrieval (84 vs 80, MRR 0.773 vs 0.652), relevance (70 vs 57), and completeness (incomplete 9 vs 14)** - its query reformulation (often ru→en) retrieves better cross-lingually than embedding the raw Russian question. The one axis it loses is **faithfulness (37 vs 50 faithful, 32 vs 24 unfaithful)**: answering more fully from a multi-hop context union yields more assertions the judge marks ungrounded (the completeness-vs-grounding tradeoff, plus a different context denominator than single-shot's top-3). **Takeaway: measuring caught a bug the naive read would have shipped as a conclusion.** Open follow-up: close the faithfulness gap (tighter grounding instruction; or judge the agent against a deduped/narrowed context).

**Post-remediation re-run (`agent_v2_ru`).** After a second review round the agent/judge path was hardened: `answered` now follows retrieved evidence (honest refusal on empty), the judged context excludes the "No relevant documents found." / tool-error sentinels, judging is per-axis, and `dispatch` drops any hallucinated `category`. Re-running the same set validated the fixes and moved the faithfulness gap the predicted follow-up direction:

| axis | v3_ru (single-shot) | agent_nocat_ru | agent_v2_ru (hardened) |
|------|---------------------|----------------|------------------------|
| retrieval hit@k | 80% | 84% | **87%** |
| retrieval MRR | 0.652 | 0.773 | **0.779** |
| faithful | 50 | 37 | 39 |
| unfaithful | 24 | 32 | **29** |
| relevant | 57 | 70 | 68 |
| complete / partially / incomplete | 1/85/14 | 2/89/9 | **5/90/5** |
| empty-source answers | - | 0 | 0 |

Filtering non-evidence out of the faithfulness context (the round-2 fix) is what nudged unfaithful 32 → 29 and faithful 37 → 39: the judge was previously scoring some answers against a context that literally said "No relevant documents found." The overall conclusion stands - the agent beats single-shot on retrieval, relevance, and completeness, and trails on faithfulness, with the gap now narrower. The fix improved measurement validity, not the model.

**Metric caveats (post round-3 hardening).** Three numbers here shifted meaning and are not strictly comparable across rounds: (1) agent MRR reflects cross-hop source dedup added in `24fae88`, so pre-dedup agent MRR (`agent_nocat_ru` 0.773) is not directly comparable to `agent_v2_ru` 0.779 (hit@k is unaffected - membership is unchanged). (2) The agent's `hops` now includes the forced synthesis turn, so a run capped at N can report N+1. (3) faithfulness and relevance are counted over in-corpus logs; completeness is counted over all logs (in-corpus plus the out-of-corpus refusal probes), so the three axes do not share a denominator.

---

## 2026-07-26 - Corpus ablation: disable developer-roadmap

**Hypothesis:** several `paraphrased_ru` misses are "right topic, wrong repo" - the generalist `developer-roadmap` corpus overlaps many domains and its chunks outrank the true single-source gold chunk. If none of the eval's gold sources live in roadmap, disabling it should raise hit@k rather than lower it.

**Change:** runtime source toggle (`PUT /v1/source/3 {active:false}`), no re-index. `hybrid_search` filters chunks to active sources only. One variable: roadmap in vs out.

**Setup:** set `paraphrased_ru` (100); retrieval identical otherwise; `base_ru_t0` (roadmap on) vs `no_roadmap_ru` (roadmap off). Retrieval-only comparison.

**Result:**

| metric | roadmap on | roadmap off | Δ |
|--------|-----------|-------------|---|
| hit@k | 80% | 90% | +10pp |
| MRR | 0.652 | 0.832 | +0.180 |
| misses | 20 | 10 | −10 |

**Delta:** large. Removing one corpus lifted hit@k by 10 points and MRR by 0.18. The gold sources for this eval set are not in roadmap, so its chunks were pure distractors ranking above true sources.

**Conclusion:** corpus composition is a first-order retrieval lever - a broad, topically-overlapping source hurts a single-source-labeled eval more than reranking or prompt tuning helped. Confirms the earlier miss diagnosis (disambiguation, not corpus gap). **Decision: drop developer-roadmap from the corpus entirely.** It is a generalist roadmap index whose chunks are shallow and topically diffuse; it added retrieval noise across every domain without being a canonical source for any of them. The showcase corpus is cleaner as a set of focused sources (interview banks, system-design-primer, redis-doc). New default retrieval on `paraphrased_ru` is **90% / MRR 0.832** (previously 80% / 0.652). The source-active toggle proved out as the ablation tool that justified the removal. Follow-up: multi-source gold labels would quantify how much of any residual "distraction" is genuinely wrong vs adjacent-correct.

---

## 2026-07-27 - Generation prompt v3 → v4 (drop inline citations)

**Hypothesis:** the three citation rules in v3 buy nothing that the structured `sources` field does not already carry, and they cost answer cleanliness. Inline `[source]` markers appeared unevenly across runs (present on some single-shot answers, absent on others), so as a consumer-facing contract they were unreliable anyway. Dropping them should leave grounding untouched, since the load-bearing rules are "use only information explicitly supported by the context" and the partial/absent/conflict clauses, not the citation lines.

**Change:** `generate_answer` v3 → v4: removed exactly three lines ("cite every factual statement in the format [source]", "do not cite a source unless it directly supports the statement", "cite all relevant sources when several support a statement"). Every other rule is byte-identical to v3.

**Scope:** `generate_answer` is read only by `chat.answer` (`app/use_cases/chat.py:180`), so this affects **single_shot only**. The agent synthesizes through `agent_system` in its ReAct loop and is untouched; `prompts/agent_system.v2.txt:11` still instructs the agent to cite inline. That asymmetry is deliberate for now: changing the agent prompt would need its own agent eval, and folding it into this run would confound the two.

**Judge impact:** none by construction. `judge_faithfulness.v1`, `judge_relevance.v1` and `judge_completeness.v1` contain no reference to citations or `[source]` markers, so the rubric does not move. v3 and v4 numbers stay comparable on all three axes.

**Setup:** set `paraphrased_ru` (100); generator `llama3.1:8b`, temp 0; rerank off. Measured on the current (roadmap-removed) corpus, so a v3 control (`v3_current`) was run alongside v4 (`v4_ru`) on the same corpus. Retrieval is prompt-independent and came out identical for both (91% / MRR 0.837), confirming a clean single-variable A/B. That retrieval is higher than the pre-roadmap-drop `v3_ru` baseline above (80% / 0.652), which is the corpus change, not the prompt.

**Result (v3 with citations vs v4 without, same corpus):**

| axis | v3 (inline citations) | v4 (no citations) |
|------|-----------------------|-------------------|
| faithful / partially / unfaithful | **56 / 29 / 15** | 49 / 30 / 21 |
| relevant / partially / irrelevant | **68 / 28 / 4** | 56 / 35 / 9 |
| complete / partially / incomplete | 2 / 87 / **11** | 1 / 82 / 17 |

**v4 is worse on all three axes**, most on relevance (68 → 56, irrelevant 4 → 9) and faithfulness (unfaithful 15 → 21).

**Conclusion: the hypothesis was wrong; the citation rules were load-bearing, not cosmetic.** Forcing the 8B to attribute every statement to a source is a grounding discipline that keeps the answer tied to the retrieved context and on topic. Removing it let the model drift into less relevant, less grounded prose. The uneven inline markers were a symptom of a weak model following the instruction imperfectly, not a reason to drop it. **Decision: revert. v4 deleted (file and DB row), v3 stays active.** The consumer-facing sources contract is handled honestly at the API/doc layer instead: the structured `sources` list is the authoritative channel, and the answer prose may still carry inline `[source]` markers unevenly. Follow-up: consistent inline citation is a model-capability lever (a stronger generator), not a prompt-removal one. Measuring reverted a plausible cosmetic change before it shipped a quality regression.

---

## 2026-07-27 - RAG measurability: retrieval width (k) and max_hops

**Goal:** the agent trails single_shot on faithfulness alone (39 vs 50, see the agent-vs-single-shot run). Turn that one failing axis into a measured lever.

**Lever chosen, `results_limit` ("k"):** feed the generator more grounding chunks. This is a parameter of our own retrieval, provider-agnostic, not an LLM option. `num_ctx` was ruled out first: Ollama's openai-compatible endpoint does not honor it (a needle survived even at `num_ctx=512`), and it is non-portable to hosted providers, so `llm.py` stays neutral (temperature/max_tokens only).

**The window is finite (needle-in-haystack, llama3.1:8b):** a fact at the start of the context is recalled at ~4k tokens but lost past ~8k (`prompt_tokens` collapses to ~2050). So the width lever has a ceiling: pushed too high the context overflows, the head is truncated, grounding is lost, and faithfulness may fall rather than rise. The sweep is a search for a sweet spot, not "more is better".

**Primary hypothesis, to check BEFORE the agent runs: the agent may already sit in the truncation zone.** single_shot context = 1 search x k chunks. Agent context = hops x k chunks (default hops 4, k 3, up to ~12 chunks), which can already exceed the window. If so, the agent's faithfulness gap is truncation, not architecture: it accumulates context across hops, overflows, the earliest hops are cut, and it answers ungrounded. Consequence: for the agent, widening k may hurt, and `max_hops` becomes the primary axis.

**Cheap check (before burning hours on the sweep):** run one real agent question ("operations on whole NumPy arrays in one call"), record `prompt_tokens` per hop. k=3 gave per-hop [459, 945], k=10 gave [459, 2480]. The agent took 2 hops (one real search + final answer), not 4; at k=10 it retrieved 5 chunks, not 10 (`distance_threshold` cut the nominal width); context peaked at 2480 tokens, well under the ~4096 window. **The "already truncated" hypothesis does not hold for a typical question:** the 8B rarely multi-hops, and the distance threshold caps effective width, so hops x k stays below the window. Truncation would need 3-4 real hops at high k, which the 8B seldom triggers. Implication: the k lever has real headroom, so faithfulness can respond to k without immediate truncation; `context_tokens` on the full run will surface the tail of answers that do approach the ceiling. The sweep proceeds as planned, with the truncation prediction downgraded from "primary cause" to "watch the context_tokens tail".

**Instrumentation:** `context_tokens` (peak per-hop `prompt_tokens`) is now logged in every agent answer's `metrics`, so a full run shows how many answers hit the ceiling.

**Setup for the sweep:** set `paraphrased_ru` (100, fixed), pipeline agent, temp 0 (k the only variable), judge qwen2.5:7b fixed. Two 1-D series via `POST /v1/eval/experiment`: k in [1,3,5,7,10] at fixed hops, then max_hops at fixed k, chosen by the k-series result. Not a cartesian grid (that is a day of agent runs), two isolated curves instead.

**Result (k-series, paraphrased_ru 100, agent, temp 0, judge qwen2.5:7b):**

| k | faithful | partially | unfaithful | relevant | complete (part/incompl) |
|---|---------:|----------:|-----------:|---------:|:--|
| 1 | 36 | 29 | 35 | 65 | 90/10 |
| 3 | **41** | 31 | 28 | 74 | 88/8 |
| 5 | 34 | 43 | 23 | 70 | 91/6 |
| 7 | 23 | 43 | 34 | 77 | 91/6 |
| 10 | 22 | 38 | 40 | 77 | 88/8 |

**The widen-retrieval hypothesis was wrong: faithfulness does not rise with k, it peaks at k=3 and then falls** (41 -> 22). More chunks = more noise in the context, the model holds the source less, unfaithful climbs (28 -> 40). Relevance meanwhile rises with k (65 -> 77): more context more often covers the question. Classic metric-tension: k up moves relevance up and faithfulness down.

Crucially this is NOT truncation: the agent takes ~2 hops, at k=10 the context peaks ~2480 tokens, well under the window (see the cheap-check). So the faithfulness drop is pure noise dilution, not the window ceiling, a cleaner result than the truncation hypothesis predicted.

**Takeaway (categorical run): the default k=3 looked like the faithfulness sweet spot.** This conclusion was WRONG, the categorical judge lacked resolution. See the numeric re-run below.

## 2026-07-28 - k-sweep re-judged with a numeric 0-10 judge (the categorical verdict reversed)

**Why:** the categorical judge (faithful/partially/unfaithful) buries a good-but-slightly-extrapolated answer ("7") and a mostly-invented one ("3") in the same "partially" bucket. Suspecting the metric hid the truth, replaced it with a numeric 0-10 judge (rubric per axis, `score` via structured output, `judge.py`/`generation_metrics.py` on averages, prompts `judge_*.v2`). Same runs re-judged (no regeneration, only re-scoring). This is a "quality of measurement" fix, not a RAG change: you cannot optimize a system with a lying ruler.

**Numeric curve (0-10, paraphrased_ru 100, agent, temp 0, judge qwen2.5:7b v2):**

| k | faithfulness | relevance | completeness |
|---|-----:|-----:|-----:|
| 1 | 6.79 | 8.60 | 5.48 |
| 3 | 6.98 | 8.68 | 5.86 |
| **5** | **7.18** | 8.90 | **6.16** |
| 6 | 6.83 | 8.87 | 5.99 |
| 7 | 6.78 | **9.06** | 6.06 |
| 10 | 6.99 | 8.94 | 6.09 |

**The numeric judge REVERSED the categorical verdict.** Categorical said faithful peaks at k=3 (41) then collapses to 22 -> "wider retrieval HURTS faithfulness". Numeric shows faithfulness RISES to a peak at k=5 (7.18) then plateaus ~6.8-7.2, it does not collapse. Reason: as k grows, faithful answers drift into "partially" (categorical counter drops), but those are SEVENS (good answers, lightly extrapolated). The average sees it, the counter does not. **The categorical metric lied in the SIGN of the conclusion, not just the magnitude.** Peaks: faithfulness/completeness at k=5, relevance at k=7.

**RRF composite over the 3 axes -> k=5.** Ranking each axis and fusing (RRF, k=60, equal weights): k=5 wins (0.0487), first on faithfulness and completeness, third on relevance. k=10 second (stable #2 on every axis), k=7 third (first on relevance but ranked last on faithfulness, which tanks it). Classic fusion property: stability beats a single peak.

**Weighted RRF / "relevance matters most":** to move the optimum to k=7 relevance needs weight x5 (i.e. drop the other axes); at x3 k=10 wins on stability. If one axis dominates that hard, skip the composite and take argmax(relevance)=k7. Better framing than weighting: constrained optimization. Faithfulness is a FLOOR (grounding is the point of RAG, a fluent but ungrounded answer is the worst case), relevance is the OBJECTIVE. `max relevance s.t. faithfulness >= threshold`. Faithfulness plateaus above k=1, so the threshold passes k=3..10, and among them max relevance is k=7. So the relevance-first view gives k=7 the honest way, not via weight x5. Product-dependent: fintech (grounding critical) keeps the floor high; a general assistant relaxes it.

**Decision: default k=5** (`results_limit` 3 -> 5). Balanced composite optimum, best faithfulness+completeness, relevance only 0.16 below its own peak. Not truncation-driven (agent ~2 hops, context well under the window); the earlier categorical "collapse" was pure metric artifact.

**Meta-lesson (headline):** quality of measurement is a prerequisite to optimization. A metric with too few levels can invert the conclusion, not just blur it. Fixed the ruler (numeric judge) before trusting any k-conclusion.

## 2026-07-28 - Reranking re-measured with the numeric judge (per-k, agent pipeline)

**Why redo it:** the earlier rerank A/Bs (2026-07-25) were categorical. Now that the judge is numeric 0-10 and default k moved to 5, re-measure the reranker as a per-k on/off sweep. Baseline OFF is the numeric agent k-sweep already judged (no need to re-run OFF). Only the reranker toggles; same set `paraphrased_ru` (100), generator `llama3.1:8b`, temp 0, judge `qwen2.5:7b`.

**Plumbing change:** rerank became a per-run knob in the agent pipeline too (previously agent read only the global `rerank.enabled`). `use_rerank` now threads `agent.run -> dispatch -> _search_corpus -> chat.search_chunks`; config stays the default, the per-run value overrides. Removed the "rerank override not supported for agent" guard in `/eval/run` and `/eval/experiment`.

**k=5 data point (single_shot ON):** run `ss_rerank_k5`, faithfulness 7.57, relevance 8.55, completeness 6.27 (100/100). Caveat: this run is single_shot while the OFF baseline and the 3/7/10 ON sweep are agent, so k=5 is not a clean same-pipeline delta. Kept as a cross-pipeline reference, not a comparison row. A clean agent ON k=5 can be added later.

**Agent ON sweep k=3/7/10:** done (`paraphrased_ru_agent_rerank_k{03,07,10}`), 100/100 judged each, backfilled as Experiment id=6 (`rerank_sweep_agent_on`) against the OFF baseline Experiment id=5 (`k_sweep_numeric_baseline`).

| k | faith OFF | faith ON | Δ | rel OFF | rel ON | compl OFF | compl ON | Δ |
|---|-----------|----------|-----|---------|--------|-----------|----------|-----|
| 3 | 6.98 | 6.93 | −0.05 | 8.68 | 8.62 | 5.86 | 6.08 | +0.22 |
| 5 | 7.18 | 7.57 (ss) | (ss) | 8.90 | 8.55 (ss) | 6.16 | 6.27 (ss) | (ss) |
| 7 | 6.78 | 7.03 | +0.25 | 9.06 | 8.99 | 6.06 | 6.46 | +0.40 |
| 10 | 6.99 | 6.97 | −0.02 | 8.94 | 9.15 | 6.09 | 6.28 | +0.19 |

**Conclusion: on the numeric judge rerank does NOT deliver the big faithfulness lift the categorical judge once showed (~+17 faithful count).** Faithfulness is essentially flat (−0.05 / +0.25 / −0.02 at k=3/7/10); the real, consistent gain is on **completeness (+0.19 to +0.40)**, with relevance mixed (best at k=10, +0.21). So the earlier "rerank stacks positively, worth enabling" verdict was largely a categorical-counter artifact: rerank helps, but modestly and on a different axis (completeness) than advertised, not a faithfulness jump. Same lesson as the k-sweep: the categorical ruler exaggerated an effect the numeric ruler sizes correctly. Reranking stays **default OFF** (the ~10s/query CPU cost is not justified by a completeness bump), opt-in for completeness-sensitive workloads.

## 2026-07-28 - Generator A/B: llama3.1 8b vs 70b (CPU) - does a bigger model earn its cost?

**Hypothesis:** faithfulness is the axis where the 8B generator is presumed weakest (lightly-extrapolated "sevens" dominate the distribution). A 70B generator should extrapolate less. Question: how much quality is the small model actually costing us?

**Setup:** single_shot, k=5, rerank off, temp 0, set `paraphrased_ru` (100), judge `qwen2.5:7b` fixed. The 70B does not fit an 8 GB GPU, so Ollama ran it on CPU (~1.9 tok/s; the client timeout had to become tunable - `LLM_TIMEOUT` env - because 120s killed the first probe). Same questions, same retrieval: hit@k and MRR came out **identical to the third digit** (0.93 / 0.845), confirming the model is the only variable. Backfilled as Experiment id=7 (`model_ab_8b_70b`).

| | 8b | 70b | Δ |
|---|-----|-----|-----|
| faithfulness | 7.22 | **7.60** | +0.38 |
| relevance | 8.55 | 8.51 | −0.04 |
| completeness | 6.19 | 6.09 | −0.10 |
| sec/question | **7.5** | 175.7 | 23x |

**Conclusion:** the 70B buys **+0.38 faithfulness and nothing else, at 23x the latency** (~3 min/question on CPU: batch-only territory, unusable interactively). Two framings make this interesting:
- The cross-encoder reranker on the 8b (`ss_rerank_k5`: faith 7.57) reproduces almost the entire 70B faithfulness gain at ~1/8 of the cost. "Scale the generator" and "rerank the context" turn out to be near-substitutes on this corpus, and the reranker is the cheaper lever.
- The equal-weight RRF composite actually ranks **8b first** (0.04892 vs 0.04865): 8b takes two axes by hair-thin margins, 70b takes one by a large one, and rank fusion is blind to magnitude. A composite says "no overall win"; the per-axis view says "one real effect, on faithfulness". Read both, decide from components.

Pending: the +0.38 is a point estimate; paired CI / significance testing (in progress as an aggregator extension) should confirm it before it hardens into a claim. **Decision: generator stays llama3.1:8b.** The 70B is deleted; the faithfulness lever of choice remains reranking (opt-in) or corpus/prompt work, not scale.

## 2026-07-28 - Paired significance testing lands in the aggregator (the audit answered)

**Why:** an external audit named the sharpest weakness of the whole lab: point estimates without uncertainty. The RRF winner k=5 beat k=10 by 0.0487 vs 0.0484 - third-decimal territory - and rerank deltas of +0.2..0.4 on a 0-10 judge scale could live entirely inside noise. Since every comparison here is paired (the same question answered in both configs), paired statistics are nearly free.

**Change:** the aggregator now computes, for every experiment, `composite.pairwise` - winner vs each other value, per generation axis: mean paired delta, bootstrap 95% CI (10k resamples, fixed seed) and a Wilcoxon signed-rank p-value (rank-based: right for ordinal judge scores with outliers; zero-delta series short-circuit to p=1). Exposed via `pairwise_stats(run_a, run_b)` and stored in experiment `results`. The tolerant numeric parser is gone too: after the full re-judge of the categorical era (1442 logs re-scored by the numeric judge), a non-numeric verdict in the DB is a bug and now fails loudly instead of being silently skipped.

**What the statistics did to our own conclusions:**

| comparison | axis | delta | CI95 | p | verdict |
|---|---|---|---|---|---|
| 70b vs 8b (single_shot k5) | faithfulness | +0.38 | [-0.04, 0.81] | 0.034 | borderline real |
| 70b vs 8b | relevance | -0.04 | [-0.68, 0.60] | 0.58 | noise |
| 70b vs 8b | completeness | -0.10 | [-0.50, 0.31] | 0.49 | noise |
| k=5 vs k=10 (agent baseline) | faithfulness | +0.19 | [-0.19, 0.57] | 0.37 | indistinguishable |
| k=5 vs k=10 | relevance | -0.04 | [-0.49, 0.40] | 0.98 | indistinguishable |
| k=5 vs k=10 | completeness | +0.07 | [-0.33, 0.48] | 0.89 | indistinguishable |

**Honest corrections this forces:**
- **k=5 vs k=10 is a statistical coin flip on every axis.** The k=5 default stands, but the honest justification is now cost, not proven quality: at indistinguishable quality, feed the generator half the chunks. The earlier "k=5 wins the composite" phrasing overstated what the data supports.
- **The 70B's faithfulness gain is real-ish but marginal**: Wilcoxon says significant (p=0.034), the bootstrap CI grazes zero - a borderline effect. "23x latency for a borderline single-axis gain" is an even stronger version of the stay-on-8b decision.
- Rerank per-k deltas (completeness +0.2..0.4) remain to be tested pair-wise; treat them as suggestive until then.

**Meta:** the lab's own headline ("fix the ruler before optimizing") now applies twice: first the judge's resolution (categorical → numeric), then the conclusion's resolution (point estimate → interval + significance). Both times the correction changed what we would have claimed.

**Postscript - the multiple comparisons trap (found immediately).** Running the full winner-vs-each grid on the k-sweep (15 tests: 5 pairs × 3 axes) "found" two significant results that are almost certainly false positives: k=5 beating its neighbors k=6 (faith +0.35, p=0.026) and k=7 (faith +0.40, p=0.045) while being indistinguishable from the distant k=10 (p=0.37). That non-monotonic shape is what noise looks like, not what a retrieval-width effect looks like - and at 15 tests with α=0.05 one expects ~0.75 false alarms per grid. Bonferroni (0.05/15 ≈ 0.003) kills both, and grazes even the one honest-looking signal: **k=5 vs k=1 completeness +0.68, p=0.009** - a single chunk visibly hurts answer completeness, the only defensible per-pair conclusion in the sweep. Revised k-sweep verdict: avoid k=1; between k=3 and k=10 no difference is provable at n=100; keep k=5 as the cheap middle. Lesson stacked on the lesson: intervals fixed the point-estimate problem, and immediately exposed the next one - the more comparisons you look at, the stricter your threshold must be, or something "significant" will always turn up.

**Full-matrix verdict (all 15 k-pairs × 3 axes, 45 tests).** One coherent signal survives scrutiny: **k=1 hurts completeness**, losing to every k≥5 (+0.51..+0.68, p=0.009..0.047) with k=3 pointing the same way - four same-direction significant pairs plus a plausible mechanism outweigh the fact that no single pair clears Bonferroni. The two k=5-beats-neighbors faithfulness hits are non-monotonic (k5≈k10), mechanism-free, and dismissed as multiple-comparison noise. Everything between k=3 and k=10 is a statistical plateau on every axis at n=100. Final answer to "which k is best": provably not 1; within 3..10 undecidable; k=5 stays as the plateau middle with the best point estimates (all deltas vs k=3 positive but unproven). To actually separate 5 from 3, the lever is sample size, not a cleverer test: n≈500-1000 shrinks the CIs ~3x - exactly the multi-fidelity step (search at 100, confirm at 1000) the experiment design already anticipates.
