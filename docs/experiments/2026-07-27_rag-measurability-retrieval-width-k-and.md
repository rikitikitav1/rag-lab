# 2026-07-27 - RAG measurability: retrieval width (k) and max_hops

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
