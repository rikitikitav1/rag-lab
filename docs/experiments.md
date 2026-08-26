# Experiments log

A lab journal of RAG-quality experiments: hypothesis → setup → result → delta → decision. The point is a reproducible, data-driven loop, not intuition.

## Methodology

- **Eval sets** live in the question bank (`set_name`). The discriminating set is `paraphrased_ru`: interview questions paraphrased and translated to Russian, so retrieval must work cross-lingually (ru query over an en corpus, FTS misses, vector-only) instead of matching source text verbatim. Raw interview questions are near-verbatim to their source (hit@k ~99%), so they hide quality differences.
- **Metrics** come from `question_logs` per `run_name`: retrieval (hit@k / MRR against `marked_sources`), generation (faithfulness / relevance / refusal via LLM-as-judge, judge = `qwen2.5:7b`, neutral to the generator to avoid co-hallucination).
- **Isolation**: change one variable at a time; hold the rest constant.
- **Reproducibility**: generation `temperature: 0` during tuning so a metric change is attributable to the change under test, not to sampling noise.
- Each run is one `eval_run` job (answers, bulk) → one `judge_answers` job (verdicts, bulk).

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
