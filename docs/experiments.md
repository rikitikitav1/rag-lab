# Experiments log

A lab journal of RAG-quality experiments: question → setup → result → decision. The point is a reproducible, data-driven loop, not intuition.

## Methodology

- **Eval sets** live in the question bank (`set_name`). The discriminating set is `paraphrased_ru`: interview questions paraphrased and translated to Russian, so retrieval must work cross-lingually (ru query over an en corpus, FTS misses, vector-only) instead of matching source text verbatim. Raw interview questions are near-verbatim to their source (hit@k ~99%), so they hide quality differences.
- **Metrics** come from `question_logs` per `run_name`: retrieval (hit@k / MRR against `marked_sources`), generation (faithfulness / relevance / completeness / refusal via LLM-as-judge, judge = `qwen2.5:7b`, neutral to the generator to avoid co-hallucination).
- **Isolation**: change one variable at a time; hold the rest constant.
- **Reproducibility**: generation `temperature: 0` during tuning so a metric change is attributable to the change under test, not to sampling noise.
- Each run is one `eval_run` job (answers, bulk) → one `judge_answers` job (verdicts, bulk).

## How an entry is written

The format is a rule of this repository, not a preference. Entries are read months later, by
someone deciding whether a number still holds, so every entry carries the same load-bearing
sections in the same order:

```
# YYYY-MM-DD - Title

Lead: two to five lines. The question this run asks, and why it is worth asking.

## Setup            opens with the coordinates line below, then what varies, what is pinned, which code ran
## Result           the numbers, in a table, with n
## <free sections>  as many as the analysis needs, named for what they say
## Decision         what became the default, what changed in code or config, what was rejected
## Caveats          what this entry does not show, and what dies on the next re-index
```

`Setup` opens with one line naming where the numbers come from, so a reader who scrolls straight to
a table can find the coordinates directly above it:

```
**Set** `paraphrased_v2_ru` (n=823) · **corpus** `clean_1024` against `baseline` · **judge** `qwen2.5:7b`
```

Rules that decide whether an entry is worth keeping:

- **One entry, one question.** A second question gets a second entry, linked, not a longer file.
- **Every number carries its n, its set and its corpus variant.** Not once in the entry: beside the
  table it belongs to, so a number cannot be lifted out of the file without them. Two numbers taken
  on different sets are not two measurements of one thing, and without the n printed next to them
  that difference reads as instrument noise. This rule exists because a set difference was nearly
  reported as hnsw instability. For runs older than the `variant` column, name the corpus of that
  era in words rather than leaving the field blank.
- **Every comparison carries an interval** or says out loud that it is a point estimate. A delta
  without a spread is not a result.
- **The decision rule is written before the run** for anything shaped as an A/B, and the entry
  says where it was written down. A criterion chosen after seeing the numbers is named as such.
- **Corrections are appended, never rewritten.** A later run that overturns an earlier number gets
  a `## Correction, measured on YYYY-MM-DD` section in the entry it corrects, pointing at the entry
  that did the correcting. The original text stays as it was written.
- **Link files, not lines.** `app/use_cases/chat.py`, never `chat.py:180`: line numbers rot within
  a week and the reader cannot tell a stale pointer from a wrong one.
- **No praise of the method.** The entry records what was measured and what it cost. Whether that
  was a good way to work is the reader's call.

---

## Entries (chronological)

- [2026-07-25 - Reranking (cross-encoder) A/B](experiments/2026-07-25_reranking-cross-encoder-a-b.md)
- [2026-07-25 - Generation prompt v2 → v3 (completeness)](experiments/2026-07-25_generation-prompt-v2-v3-completeness.md)
- [2026-07-25 - Reranking on top of v3 (stacking, clean)](experiments/2026-07-25_reranking-on-top-of-v3-stacking.md)
- [2026-07-26 - ReAct agent vs single-shot (a measured loss)](experiments/2026-07-26_react-agent-vs-single-shot-a.md)
- [2026-07-26 - Corpus ablation: disable developer-roadmap](experiments/2026-07-26_corpus-ablation-disable-developer-roadmap.md)
- [2026-07-27 - Generation prompt v3 → v4 (drop inline citations)](experiments/2026-07-27_generation-prompt-v3-v4-drop-inline.md)
- [2026-07-27 - RAG measurability: retrieval width (k) and max_hops](experiments/2026-07-27_rag-measurability-retrieval-width-k-and.md)
- [2026-07-28 - k-sweep re-judged with a numeric 0-10 judge (the categorical verdict reversed)](experiments/2026-07-28_k-sweep-re-judged-with-a.md)
- [2026-07-28 - Reranking re-measured with the numeric judge (per-k, agent pipeline)](experiments/2026-07-28_reranking-re-measured-with-the-numeric.md)
- [2026-07-28 - Generator A/B: llama3.1 8b vs 70b (CPU) - does a bigger model earn its cost?](experiments/2026-07-28_generator-a-b-llama3-1-8b.md)
- [2026-07-28 - Paired significance testing lands in the aggregator (the audit answered)](experiments/2026-07-28_paired-significance-testing-lands-in-the.md)
- [2026-07-29 - Judge vs judge: qwen2.5 7b against 32b on the same 100 answers (Spearman)](experiments/2026-07-29_judge-vs-judge-qwen2-5-7b.md)
- [2026-07-29 - Agent becomes an MCP client (DeepWiki first) - comparability boundary](experiments/2026-07-29_agent-becomes-an-mcp-client-deepwiki.md)
- [2026-08-24 - Phased eval runs, and the unload that freed nothing](experiments/2026-08-24_phased-eval-runs-and-the-empty-cache.md)
- [2026-08-24 - Corpus-first, and the fallback that never fired](experiments/2026-08-24_corpus-first-and-the-fallback-that-never.md)
- [2026-08-25 - The gate that fires, and the refusal that never comes](experiments/2026-08-25_the-gate-that-fires-and-the-refusal-that.md)
- [2026-08-25 - A cheaper gate signal, and a win on the wrong axis](experiments/2026-08-25_a-cheaper-gate-signal-and-a.md)
- [2026-08-25 - A refusal at last, and a threshold that measured nothing](experiments/2026-08-25_a-refusal-at-last-and-the.md)
- [2026-08-26 - The same agent written four ways, and what the standard costs](experiments/2026-08-26_the-same-agent-written-four-ways.md)
- [2026-08-26 - A corpus you can keep two of, and the instrument that measures it](experiments/2026-08-26_a-corpus-you-can-keep-two-of.md)
- [2026-08-27 - Corpus hygiene that moved the number, and four instruments that were lying](experiments/2026-08-27_hygiene-that-moved-the-number.md)
