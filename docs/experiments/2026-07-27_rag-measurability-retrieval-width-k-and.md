# 2026-07-27 - RAG measurability: retrieval width (k) and max_hops

The agent trails single-shot on faithfulness alone (39 against 50, see [the agent-vs-single-shot
run](2026-07-26_react-agent-vs-single-shot-a.md)). This entry turns that one failing axis into a
measured lever: feed the generator more grounding chunks and see what faithfulness does.

## Setup

**Set** `paraphrased_ru` (n=100) · **corpus** pre-variant era, post-ablation · **judge**
`qwen2.5:7b`, categorical

The lever is `results_limit` ("k"), a parameter of our own retrieval rather than an LLM option.
`num_ctx` was ruled out first: Ollama's OpenAI-compatible endpoint does not honour it (a needle
survived at `num_ctx=512`) and it does not port to hosted providers, so `llm.py` stays neutral with
temperature and max_tokens only.

Pipeline agent, temp 0, judge fixed. Two 1-D series through `POST /v1/eval/experiment`: k over [1,
3, 5, 7, 10] at fixed hops, then `max_hops` at the k the first series picks. Two isolated curves
rather than a cartesian grid, which would be a day of agent runs.

**The truncation hypothesis, checked before the sweep.** The window is finite: for llama3.1:8b a
fact at the start of the context is recalled at about 4k tokens and lost past about 8k, where
`prompt_tokens` collapses to roughly 2050. Single-shot context is one search by k chunks; agent
context is hops by k chunks (four hops at k=3 is up to about 12), so the agent might already sit in
the truncation zone, and the
faithfulness gap would be truncation rather than architecture.

One real question ("operations on whole NumPy arrays in one call"), `prompt_tokens` per hop: k=3
gave [459, 945], k=10 gave [459, 2480]. The agent took two hops, not four, and at k=10 retrieved
five chunks rather than ten, because the distance threshold cut the nominal width. The peak was 2480
tokens, well under the roughly 4096-token window of that era. The hypothesis does not hold for a typical question, so the sweep
proceeds and truncation is downgraded to a tail to watch. `context_tokens` (peak per-hop
`prompt_tokens`) is now logged in every agent answer.

## Result

| k | faithful | partially | unfaithful | relevant | complete (part/incompl) |
|---|---------:|----------:|-----------:|---------:|:--|
| 1 | 36 | 29 | 35 | 65 | 90/10 |
| 3 | **41** | 31 | 28 | 74 | 88/8 |
| 5 | 34 | 43 | 23 | 70 | 91/6 |
| 7 | 23 | 43 | 34 | 77 | 91/6 |
| 10 | 22 | 38 | 40 | 77 | 88/8 |

Faithfulness does not rise with k. It peaks at k=3 and falls to 22, while relevance rises with k (65
→ 77): more chunks more often cover the question and at the same time dilute the grounding. The
tension is the readable part of the curve.

It is not truncation: the agent takes about two hops and peaks near 2480 tokens at k=10, well under
the window.

## Decision

Read at the time as "k=3 is the faithfulness sweet spot". That verdict did not survive the change of
judge, see the correction below.

## Correction, measured on 2026-07-28

The categorical judge lacked the resolution this curve needed, and [the numeric
re-judge](2026-07-28_k-sweep-re-judged-with-a.md) reversed the sign of the conclusion: faithfulness
rises to a peak at k=5 and plateaus rather than collapsing. What fell here was the count of answers
labelled "faithful", as good answers drifted into the "partially" bucket.

## Caveats

- counts, not scores, single run per k, no intervals
- the truncation check is one question, not a distribution; it bounds the tail loosely
