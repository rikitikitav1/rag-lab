# 2026-08-25 - A cheaper gate signal, and a win on the wrong axis

The previous entry left the coverage gate working but expensive: every question paid a
cross-encoder pass before the agent was allowed to answer. This run asks whether that model has
to be in the agent runtime at all, or whether the vector distance the retrieval already computed
says the same thing for free.

## Setup

One variable, `gate_signal`, three arms of 200 questions each:

| arm | what calls retrieval weak |
|-----|---------------------------|
| `cross_encoder` | best of the top 5 hits scored by `bge-reranker-v2-m3`, weak below 0.5 |
| `distance` | vector distance of the best hit, weak at 0.39 and above |
| `either` | weak when one of the two says so |

Everything else pinned: `corpus_first_weak`, `max_hops` 4, `k` 5, reranking off, one external
integration (DeepWiki), llama3.1:8b generating, qwen2.5:7b judging, `agent.system` v5,
`agent.tool_match` v3. Pools scored separately: 100 in-corpus (`paraphrased_ru`) and 100
out-of-corpus (repository internals the corpus does not hold).

The off-domain pool sat this one out, and the reason is structural rather than budgetary: no tool
is admitted for a question about carbonara (there is no `owner/repo` in it), so nothing is there
to hand off to, weak context is kept instead of dropped, and the gate signal changes a flag in
the log without changing behaviour. While `topic_threshold` stays off, all three arms are
identical on that pool by construction. The moment the topic axis is switched on, dropping starts
happening there too and off-domain has to come back into the grid.

The grid ran on the tree of the commit titled "Add a topic axis that lets the agent decline
instead of reaching out" (`f1118aa` as the branch stands; the title is the stable half, hashes
move when history is reshaped), with the worker process started a minute before the first
answer and never restarted, so the four commits that landed while it was draining never reached
it. That is checkable from the logs rather than from trust: the next commit adds a `corpus` key to
the config snapshot, and no log in these three runs carries it.

The distance threshold 0.39 was not tuned for this run. It came from an earlier calibration
against the same flag budget the cross-encoder spends, and the run confirms the budgets match: on
the hundred in-corpus questions the distance flags 24 and the cross-encoder flags 24. The arms are
compared at equal willingness to open, not at equal thresholds.

## What the run says

In-corpus, nothing moves:

| | `cross_encoder` | `distance` | `either` |
|---|---|---|---|
| faithfulness / relevance | 7.01 / 9.18 | 7.19 / 8.86 | 7.21 / 8.85 |
| gate fired | 25/100 | 22/100 | 33/100 |
| latency avg / p50 | 11.0 / 10.2 | **8.7 / 7.9** | 10.3 / 9.6 |

Every paired test lands between p=0.15 and p=0.99. All three arms answer all hundred questions
from the corpus and never leave, including the ones where the gate opened: the admission check
refuses to hand them a repository tool, so an open gate costs nothing.

Out of corpus:

| | `cross_encoder` | `distance` | `either` |
|---|---|---|---|
| faithfulness / relevance | 4.90 / 7.20 | 5.45 / 7.35 | **5.84 / 8.26** |
| answers with an `mcp:` source | 56 | 55 | **63** |
| answered from the corpus, never left | 0.40 | 0.44 | **0.34** |
| gate fired | 92/100 | 90/100 | 97/100 |
| latency avg / p50 | 19.6 / 19.5 | **17.9 / 14.3** | 21.3 / 21.0 |

A p-value alone says nothing about how large a difference could still be hiding, so the deltas
come with bootstrap intervals over the paired differences (10k resamples, per question):

| comparison | axis | delta | CI95 | better / worse |
|---|---|---|---|---|
| ce → distance, in-corpus | faithfulness | +0.18 | [-0.25, +0.60] | 21 / 14 |
| ce → distance, in-corpus | relevance | -0.32 | [-0.77, +0.09] | 21 / 22 |
| ce → distance, outside | faithfulness | +0.49 | [-0.49, +1.46] | 37 / 29 |
| ce → distance, outside | relevance | +0.02 | [-0.83, +0.87] | 23 / 28 |
| distance → either, outside | faithfulness | +0.46 | [-0.42, +1.35] | 31 / 27 |
| distance → either, outside | relevance | +0.99 | [+0.21, +1.78] | 36 / 12 |

Only one interval stays clear of zero: the relevance of `either` against `distance` outside the
corpus.

The empty rule fired once in 200 questions in every arm, which is the same lesson as before in
smaller print: hybrid retrieval almost always returns something, so a fallback that waits for
nothing at all waits forever.

## What it means

**The cross-encoder leaves the agent runtime.** No axis separates it from `distance`, and the
intervals say how strong that statement is allowed to be: inside the corpus the difference is
bounded within about ±0.8 of a point out of ten, outside it only within ±1.5, which is wide
enough that a real effect of a point could still be hiding there. `distance` opens the gate as
often (90 against 92 outside) and costs 2.3s less per in-corpus question and five seconds less on
the median external one. So the honest claim is not "identical" but "no difference found at this
sample size, and the cheaper arm is 20% faster". One thing worth not hiding under that: in-corpus
relevance leans against `distance` (9.18 against 8.86, interval [-0.77, +0.09]), the only axis
with a visible tilt. The default in `config.yaml` is now `gate_signal: distance`.

**`either` is real, and it is bought on an axis we already distrust.** It opens more (97 of 100),
sends eight more questions outside, and cuts the share answered from the corpus without ever
leaving from 0.44 to 0.34. Those ten questions are the strongest thing in its favour: the corpus
does not hold their answer, and it answered anyway because both thresholds stayed shut.

The decision against it is a decision about price, not about significance. Grounding, the axis
that would justify the extra seconds, moves +0.46 with an interval [-0.42, +1.35], so at this
sample size `either` is not shown to ground answers better. Relevance moves +0.99 with an
interval clear of zero, and the previous entry measured relevance holding at 8.4-8.8 on answers
grounded in nothing, which is exactly why it is not the axis we buy on. Naming grounding as the
target axis after the numbers were in is worth admitting out loud; the pre-registered version of
this decision belongs in the next grid, not this one. What does not carry the argument is a
multiple-comparison correction: the family of six was drawn after the run, and a family of two
flips the verdict, so the correction is a description here, not a reason. `either` stays a
documented option with a price tag (about 1.6s in-corpus, 3.4s outside).

**Equal budgets, different mistakes.** The two signals flag nearly the same number of questions
and disagree about which ones: this is why `either` opens seven more than `distance` alone rather
than the same set. An earlier probe on in-corpus answers showed the same shape, both signals
catching 5 of the 10 worst answers at a 22-flag budget while overlapping on only 12 of the 22
flags. Two cheap signals that fail differently are worth more together than either alone, which
is exactly what `either` sells and what its latency bill charges for.

## Caveats

- The thresholds (0.39 distance, 0.5 cross-encoder) are calibrated to this corpus and die on
  re-index. Corpus hygiene is queued before ingest-back, and it will invalidate both.
- **The distance threshold sits on a cliff.** On the same hundred in-corpus questions it flags 37
  at 0.37, 24 at 0.39 and 9 at 0.41: two hundredths change the firing rate fourfold. The equal
  flag budget that makes this comparison fair (24 flags against the cross-encoder's 24 at 0.5)
  holds at exactly one point, and a curve of three thresholds per arm would say more than one
  point per arm does. `weak_distance` is a run parameter now, so that grid is one call away.
- `completeness` is scored only where a reference answer exists, so the out-of-corpus pool has no
  completeness column. The two axes reported there are grounding in whatever source answered and
  relevance to the question.
- Relevance is measured against the question, not against the truth, and this pair of runs is now
  the second measurement in a row where it moves while grounding does not. Treat it as a
  readability signal, not a correctness one.
- Latency numbers include the judge-free answer path only; the judge runs afterwards as its own
  job.

Numbers pulled with `GET /v1/eval/compare?runs=crag2_sig_cross_encoder&runs=crag2_sig_distance&runs=crag2_sig_either`,
which reports the pools separately and the paired tests per pair of arms.
