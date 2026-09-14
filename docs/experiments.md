# Experiments log

A lab journal of RAG-quality experiments: question → setup → result → decision. The point is a reproducible, data-driven loop, not intuition.

## Methodology

- **Eval sets** live in the question bank (`set_name`). Verdicts are read on `paraphrased_v2_ru` and `paraphrased_v2` (`verdict.criterion_sets` in `config.yaml`; older entries used `paraphrased_ru`): interview questions paraphrased, and for the `_ru` sets translated to Russian, so retrieval must work cross-lingually (ru query over an en corpus, FTS misses, vector-only) instead of matching source text verbatim. Raw interview questions are near-verbatim to their source (hit@k ~99%), so they hide quality differences.
- **Metrics** come from `question_logs` per `run_name`: retrieval (hit@k / MRR against `marked_sources`), generation (faithfulness / relevance / completeness / refusal via LLM-as-judge, judge = `Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM since 2026-09-11, `qwen2.5:7b` on ollama before that and reproduced by [stand mode 6](stand_modes.md), neutral to the generator to avoid co-hallucination).
- **Isolation**: change one variable at a time; hold the rest constant.
- **Reproducibility**: the generator's sampler is recorded with the run; the default is `temperature: 0.1`, and a run pins another with `generation_sampler`. Even at temperature 0 two runs of one generator differ, so a change is read against the generator's own floor (README, "Why the numbers hold"), not against zero.
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
- [2026-08-28 - A third heading level in the cut, and what it did not buy](experiments/2026-08-28_a-third-heading-level-in-the-cut.md)
- [2026-08-28 - A gate that degenerated the other way](experiments/2026-08-28_a-gate-that-degenerated-the-other-way.md)
- [2026-08-28 - Reranking and the language of the question, and why the switch is not the language](experiments/2026-08-28_reranking-and-the-language-of-the-question.md)
- [2026-08-29 - Generator grid: 4b against 8b, reranking, and two languages](experiments/2026-08-29_generator-grid-4b-against-8b.md)
- [2026-08-29 - The same question in two languages](experiments/2026-08-29_the-same-question-in-two-languages.md)
- [2026-08-30 - The questions the criterion cannot see](experiments/2026-08-30_the-questions-the-criterion-cannot-see.md)
- [2026-08-30 - Dropping what repeats across a source, and the six sections it nearly took](experiments/2026-08-30_dropping-what-repeats-across-a-source.md)
- [2026-08-30 - The ceiling that changed nothing, and the rule that decided anyway](experiments/2026-08-30_the-ceiling-that-changed-nothing.md)
- [2026-08-31 - One sentence, two instructions, and the arm that told them apart](experiments/2026-08-31_one-sentence-two-instructions.md)
- [2026-09-06 - Our ranks against the standard, on the half that has no judge](experiments/2026-09-06_our-ranks-against-the-standard-without-a-judge.md)
- [2026-09-06 - The axis that makes our refusals, priced by switching it off](experiments/2026-09-06_the-axis-that-makes-our-refusals.md)
- [2026-09-06 - Our judge against the standard's, and the ruler that had to come first](experiments/2026-09-06_our-judge-against-the-standards.md)
- [2026-09-07 - The phases split, and the replay that checked it without asking a model twice](experiments/2026-09-07_the-phases-split-and-the-replay-that-checked-it.md)
- [2026-09-07 - What moving onto the standard was worth, and what it cost](experiments/2026-09-07_what-the-standard-was-worth.md)
- [2026-09-08 - The directive nobody gave, and what our judge charges for Russian](experiments/2026-09-08_the-directive-nobody-gave.md)
- [2026-09-09 - What batch invariance costs on an AWQ judge, and what it buys](experiments/2026-09-09_what-batch-invariance-costs-on-an-awq-judge.md)
- [2026-09-09 - The same rows judged by two engines, and what that comparison cannot say](experiments/2026-09-09_the-same-rows-judged-by-two-engines.md)
- [2026-09-13 - The judge that looped on whitespace, and a grammar rule that moved 17% of its verdicts](experiments/2026-09-13_the-judge-that-looped-on-whitespace.md)
- [2026-09-13 - A refusal the rule could not read: "the context does not contain"](experiments/2026-09-13_a-refusal-the-rule-could-not-read.md)
- [2026-09-13 - The penalty nobody asked for, and a door that drops it](experiments/2026-09-13_the-penalty-nobody-asked-for.md)
- [2026-09-13 - Fifteen pairs the owner judged, read again by the new judge](experiments/2026-09-13_fifteen-pairs-the-owner-judged.md)
- [2026-09-13 - The same generator on two engines, and a refusal that was a rule](experiments/2026-09-13_the-same-generator-on-two-engines.md)
- [2026-09-13 - A bigger model at the same retrieval, and where it refuses](experiments/2026-09-13_a-bigger-model-at-the-same-retrieval.md)
- [2026-09-14 - A cloud judge on the same answers, two and a half points kinder on grounding](experiments/2026-09-14_a-cloud-judge-on-the-same-answers.md)
- [2026-09-14 - A panel for the language probe, and a regime read against a reference](experiments/2026-09-14_a-panel-for-the-language-probe.md)
