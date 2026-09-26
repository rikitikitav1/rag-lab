# Experiments log

A lab journal of RAG-quality experiments: question → setup → result → decision. The point is a reproducible, data-driven loop, not intuition.

## Methodology

- **Eval sets** live in the question bank (`set_name`). Verdicts are read on `paraphrased_v2_ru` and `paraphrased_v2` (`verdict.criterion_sets` in `config.yaml`; older entries used `paraphrased_ru`): interview questions paraphrased, and for the `_ru` sets translated to Russian, so retrieval must work cross-lingually (ru query over an en corpus, FTS misses, vector-only) instead of matching source text verbatim. Raw interview questions are near-verbatim to their source (hit@k ~99%), so they hide quality differences.
- **Metrics** come from `question_logs` per `run_name`: retrieval (hit@k / MRR against `marked_sources`), generation (faithfulness / relevance / completeness / refusal via LLM-as-judge, judge = `Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM since 2026-09-11, `qwen2.5:7b` on ollama before that and reproduced by [stand mode 6](stand_modes.md), neutral to the generator to avoid co-hallucination).
- **Isolation**: change one variable at a time; hold the rest constant.
- **Reproducibility**: the generator's sampler is recorded with the run; the default is `temperature: 0.1`, and a run pins another with `generation_sampler`. Even at temperature 0 two runs of one generator differ, so a change is read against the generator's own floor (README, "Why the numbers hold"), not against zero.
- Each run is one `eval_run` job (answers, bulk) → one `judge_answers` job (verdicts, bulk).
- **Where a number's file lives**: a pass writes it under `datasets/measurements/` beside the stand and not into git, so an entry carries the table itself and names the file and the job id as its address.

## How an entry is written

Entries are often read months later, by someone checking whether a number still holds. Every
entry uses the same sections in the same order, so that reader can find the setup, the result, the
decision and the caveats in the same place each time:

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

Each entry also follows these rules:

- **One entry, one question.** A second question gets a second entry, linked from the first. This
  keeps each file short enough to check against its own numbers.
- **Every number carries its n, its set and its corpus variant**, written beside the table it belongs
  to rather than once for the whole entry. A number copied out of the file then keeps its context.
  Two numbers taken on different sets are not two measurements of one thing, and without the n next
  to them that difference can look like instrument noise. A set difference was once nearly reported
  as hnsw instability for exactly this reason. For runs older than the `variant` column, name the
  corpus of that era in words instead of leaving the field blank.
- **Every comparison carries an interval**, or states that it is a point estimate. A delta without
  a spread cannot be told apart from noise.
- **For anything shaped as an A/B, the decision rule is written before the run**, and the entry says
  where it was written down. If a criterion was chosen after the numbers were seen, the entry says so.
- **Corrections are appended, not rewritten.** When a later run overturns an earlier number, the
  entry it corrects gets a `## Correction, measured on YYYY-MM-DD` section pointing at the entry that
  did the correcting. The original text stays as written, so the history of the number stays visible.
- **Link files, not lines.** Write `app/use_cases/chat.py`, not `chat.py:180`. Line numbers go stale
  within a week, and the reader cannot tell a stale pointer from a wrong one.
- **Record results, not opinions about the method.** The entry states what was measured and what it
  cost, and leaves the judgement of the approach to the reader.

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
- [2026-08-24 - Phased eval runs: `empty_cache()` did not free live models](experiments/2026-08-24_phased-eval-runs-and-the-empty-cache.md)
- [2026-08-24 - Corpus-first agent: the empty-result fallback fired 0 of 20 times](experiments/2026-08-24_corpus-first-and-the-fallback-that-never.md)
- [2026-08-25 - Coverage gate on weak retrieval: it opens, but no policy refuses](experiments/2026-08-25_the-gate-that-fires-and-the-refusal-that.md)
- [2026-08-25 - Gate signal: vector distance against the cross-encoder, no difference found](experiments/2026-08-25_a-cheaper-gate-signal-and-a.md)
- [2026-08-25 - Topic axis for refusals at threshold 0.50, and a 0.709 threshold caught before the run](experiments/2026-08-25_a-refusal-at-last-and-the.md)
- [2026-08-26 - Hand-rolled agent loop against its LangGraph ports: the port matches](experiments/2026-08-26_the-same-agent-written-four-ways.md)
- [2026-08-26 - Corpus variants: two chunkings side by side, and how they are measured](experiments/2026-08-26_a-corpus-you-can-keep-two-of.md)
- [2026-08-27 - Corpus hygiene (`clean_1024` against `baseline`) moves retrieval; four broken instruments found](experiments/2026-08-27_hygiene-that-moved-the-number.md)
- [2026-08-28 - Third heading level in the cut (`prefix_1024`): no retrieval gain](experiments/2026-08-28_a-third-heading-level-in-the-cut.md)
- [2026-08-28 - Conditional reranking gate: rejected](experiments/2026-08-28_a-gate-that-degenerated-the-other-way.md)
- [2026-08-28 - Reranking by question language: helps Russian, hurts English on `baseline` only, no language switch](experiments/2026-08-28_reranking-and-the-language-of-the-question.md)
- [2026-08-29 - Generator grid: 4b against 8b, reranking, and two languages](experiments/2026-08-29_generator-grid-4b-against-8b.md)
- [2026-08-29 - Russian against English on 820 paired questions (post hoc)](experiments/2026-08-29_the-same-question-in-two-languages.md)
- [2026-08-30 - Veto set over four unseen source families (`veto_v1`): hygiene confirmed](experiments/2026-08-30_the-questions-the-criterion-cannot-see.md)
- [2026-08-30 - Dropping blocks repeated across a source (`noboiler_1024`): not adopted](experiments/2026-08-30_dropping-what-repeats-across-a-source.md)
- [2026-08-30 - Chunk ceiling 2048 against 1024: no difference, the tie-break keeps 1024](experiments/2026-08-30_the-ceiling-that-changed-nothing.md)
- [2026-08-31 - Cross-language clause in the judge prompts: not tested, not adopted](experiments/2026-08-31_one-sentence-two-instructions.md)
- [2026-09-06 - RAGAS ID-based retrieval metrics against our hit@k and MRR](experiments/2026-09-06_our-ranks-against-the-standard-without-a-judge.md)
- [2026-09-06 - Topic axis switched off: the agent refuses far less](experiments/2026-09-06_the-axis-that-makes-our-refusals.md)
- [2026-09-06 - Our faithfulness judge against RAGAS Faithfulness, and the judge noise floor](experiments/2026-09-06_our-judge-against-the-standards.md)
- [2026-09-07 - Agent graph phase split, checked by replay without model calls](experiments/2026-09-07_the-phases-split-and-the-replay-that-checked-it.md)
- [2026-09-07 - RAGAS arc summary: worth integrating, not worth keeping in the loop](experiments/2026-09-07_what-the-standard-was-worth.md)
- [2026-09-08 - Answer-language directive fix, and the judge's penalty for Russian](experiments/2026-09-08_the-directive-nobody-gave.md)
- [2026-09-09 - Batch invariance on the AWQ judge: 8.7x slower, no verdict changed](experiments/2026-09-09_what-batch-invariance-costs-on-an-awq-judge.md)
- [2026-09-09 - Same rows judged on two engines: 39% of verdicts differ, not attributable to the engine](experiments/2026-09-09_the-same-rows-judged-by-two-engines.md)
- [2026-09-13 - Judge looping on whitespace: the grammar rule fix moved 17% of verdicts](experiments/2026-09-13_the-judge-that-looped-on-whitespace.md)
- [2026-09-13 - A refusal the rule could not read: "the context does not contain"](experiments/2026-09-13_a-refusal-the-rule-could-not-read.md)
- [2026-09-13 - Default repetition penalties on vLLM and ollama, now set explicitly](experiments/2026-09-13_the-penalty-nobody-asked-for.md)
- [2026-09-13 - Fifteen pairs the owner judged, read again by the new judge](experiments/2026-09-13_fifteen-pairs-the-owner-judged.md)
- [2026-09-13 - Same generator on ollama and vLLM: not worse, faster, default unchanged](experiments/2026-09-13_the-same-generator-on-two-engines.md)
- [2026-09-13 - DeepSeek against llama3.1:8b at the same retrieval, and where each refuses](experiments/2026-09-13_a-bigger-model-at-the-same-retrieval.md)
- [2026-09-14 - Cloud judge against our judge on the same answers: 2.5 points kinder on grounding](experiments/2026-09-14_a-cloud-judge-on-the-same-answers.md)
- [2026-09-14 - Language probe: a fixed 60-row panel and a regime read against a reference](experiments/2026-09-14_a-panel-for-the-language-probe.md)
- [2026-09-19 - LLM chunk grader before generation: does not clear the bar, off by default](experiments/2026-09-19_a-grader-that-does-not-buy-it.md)
- [2026-09-20 - Rewriting the question when the search looks weak](experiments/2026-09-20_a-rewrite-that-does-not-clear-the-bar.md)
- [2026-09-23 - Schema-guided reasoning instead of the model's own tool calling](experiments/2026-09-23_a-schema-that-makes-the-loop-worse.md)
- [2026-09-23 - Stripping the noise inside the chunks the grader kept](experiments/2026-09-23_a-strip-that-does-not-buy-the-answer.md)
- [2026-09-25 - Which converter turns each kind of document into markdown](experiments/2026-09-25_two-converters-one-per-regime.md)
