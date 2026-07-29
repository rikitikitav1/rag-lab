# 2026-07-26 - Corpus ablation: disable developer-roadmap

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
