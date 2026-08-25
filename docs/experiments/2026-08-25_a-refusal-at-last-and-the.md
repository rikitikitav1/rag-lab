# 2026-08-25 - A refusal at last, and a threshold that measured nothing

Two entries ago the agent learned to leave the corpus when the corpus is weak. It also turned out
never to refuse: on a hundred questions no available source can answer, every policy answered,
fluently, with a citation to whatever chunk the hybrid returned. This run adds the second axis and
measures whether it closes that hole without breaking anything that works.

Coverage and topic are different questions. Coverage asks whether the corpus answers *this*
question; topic asks whether the question belongs to our domain at all. The gate handles the
first. The topic axis is the distance from the question to the nearest chunk in the corpus: above
the threshold the agent offers no external tool, drops the retrieved context and answers with a
refusal (`agent.no_evidence`).

## The threshold that measured nothing

The run was queued with `topic_threshold: 0.709`, a number carried over from the calibration
notes, and the audit round caught it before the arm started: 0.709 lives on the scale of a PCA
reconstruction residual, an earlier candidate signal that lost to plain distance and was dropped
from the code. On the distance scale the largest value across all 300 questions is 0.690. The arm
would have fired zero times and produced a clean, wrong conclusion that the axis does nothing.

Nothing in the code was broken: the number is a valid float, the comparison runs, no test could
see it. What made it findable was that the same quantity can be recomputed outside the runtime,
straight from the stored question embeddings, and someone did.

Recalibrated on the distance scale, still descriptively and before the run:

| threshold | off_domain | out_of_corpus | paraphrased_ru | curated in-corpus |
|---|---|---|---|---|
| 0.47 | 82 | 19 | 0 | 7/30 |
| **0.50** | **71** | **5** | **0** | **4/30** |
| 0.52 | 60 | 1 | 0 | 4/30 |
| 0.55 | 35 | 0 | 0 | 3/30 |

0.50 went into the run and was not touched afterwards.

## Setup

Two arms on the same questions: `topic_threshold` off against 0.50. Pools: 100 in-corpus
paraphrases, 100 out-of-corpus (repository internals), 100 off-domain (cooking, medicine, law,
legacy stacks, post-cutoff tech), plus the 33 hand-written questions of the `curated` set, added
on the auditor's point that machine paraphrases of the corpus are the easiest possible test for
false refusals. Everything else pinned: `corpus_first_weak`, gate signal `distance` at 0.39,
`max_hops` 4, k 5, reranking off, DeepWiki as the only integration, llama3.1:8b generating,
qwen2.5:7b judging.

The decision rule was written into the audit mailbox **before** the run: primary axis is the
refusal rate on off-domain, tested paired (McNemar over the same questions); a veto on false
refusals inside the corpus above 5 of 100; a veto on external questions turning into refusals
instead of tool calls, above a paired delta of 10 of 100; everything else descriptive.

## What the run says

| | off | 0.50 | paired |
|---|---|---|---|
| **refusals, off-domain (primary)** | 11/100 | **50/100** | +41 / -2, p<0.001 |
| false refusals, in-corpus (veto 1) | 0/100 paraphrase, 0/30 curated | 0/100 paraphrase, 2/30 curated | +2 / -0, p=0.5 |
| answers lost on hand-written questions (descriptive, no veto attached) | 0/30 | 4/30 (2 refusals, 2 narrated calls) | |
| refusals instead of the tool (veto 2) | 7/100 out-of-corpus, 0/3 curated | 9/100, 1/3 | +7 / -5, p=0.77 |
| the axis fired | n/a | 71/100 off-domain, 7/103 external, 4/130 in-corpus | |
| in-corpus faithfulness / relevance | 7.17 / 9.03 | 7.11 / 8.90 | -0.06 [-0.46, +0.34] / -0.13 [-0.53, +0.29] |
| latency in-corpus / external | 8.9s / 16.2s | 8.9s / 17.3s | |

The primary axis moves and both vetoes hold, so the axis goes on by default at 0.50.

Every flagged question reached the log as `fallback_reason: off_topic`, but the reason is not the
outcome, and on the outcome the two mechanisms did disagree twice out of 71. "Who was the first
president of France?" scores 0.605 on the axis and still comes back answered, with five sources
from `redis-doc` and a confident "Louis-Napoleon Bonaparte, 1848-1852"; a sourdough baking question
does the same. The cause is structural rather than a fluke of those two questions: the axis judges
the raw question, then the *gate* decides whether to drop the context, and the gate judges whatever
the agent typed into the search. Both retrievals came back just under the gate's 0.39, at 0.384 and
0.389, so the gate overruled the axis on a hundredth of a point. When the axis has already said "not our
domain" the gate has nothing left to decide, so the context should go unconditionally. That landed
after this run rather than inside it, and it changes `corpus_first` too: that policy has no gate
thresholds, so its verdict used to be empty and the context stayed even when the axis had fired.
That every flagged question now loses its corpus sources is a property of the code and is covered
by a test; what those runs turn into instead is a question for the model, and it was measured by
re-running the 82 questions the axis had flagged:

| outcome on the 82 flagged questions | before the fix | after |
|---|---|---|
| refused | 56 | 49 |
| narrated tool call | 21 | 28 |
| unsupported answer | 3 | 5 |
| **answered with corpus sources** | **2** | **0** |

The two answers built on somebody else's citation are gone, which is what the change guarantees.
The rest drifted the other way: seven refusals became narrated tool calls, and a narrated call
reaches the user as a bare "No relevant documents found" rather than a sentence explaining that
nothing here covers the question. Dropping the context hands the model a `NO_RESULTS` tool result,
and that is exactly the prompt that makes an 8B type a tool call as prose. So the fix trades two
confident fabrications for seven blunter refusals, which is a trade worth making and an argument
for the change queued next: a final no-evidence turn whenever a hop narrates a call with no
sources behind it.

This was a re-run of the flagged subset, not a second A/B: same threshold, same system, one line
of difference, so run-to-run variation counts against the reading rather than for it.

## Where it does not work, which is the interesting half

The catch is not spread across off-domain questions, it is concentrated in one kind of them:

| subgroup | n | refusals off → 0.50 |
|---|---|---|
| distant topics (cooking, chemistry, law, everyday life) | 83 | 11 → 49 |
| legacy stacks (FoxPro, Delphi, Lotus, AS/400) | 11 | 0 → 1 |
| post-cutoff tech (MCP, PG 17, Valkey, DeepSeek) | 6 | 0 → 0 |

The subgroups above are split by keyword over the question text; the auditor's independent split
(79 / 12 / 9 instead of 83 / 11 / 6) puts the same picture on the same three rows, so the
conclusion survives the way the line is drawn.

A question about FoxPro sits at the same distance from an IT corpus as a question about Postgres,
because for cosine distance it *is* the same topic. So the axis catches carbonara and does nothing
for Valkey, and no threshold fixes that: those questions are not off-topic, they are inside the
topic and outside the corpus, which is coverage plus recency. Reporting one number (71% caught)
would have read as a solved problem.

Of the 71 questions the axis flagged, 50 ended in a refusal. Most of the gap is the model printing
a tool call as prose (16 narrated calls) even with no tool in the schema, an old friend from the
llama3.1 template.

## The classifier was corrected after seeing the data

Refusals are detected by phrase, and the phrase this model prefers, "I couldn't find any
information on ... from my available sources", was not in the list. Thirteen refusals in the
treated arm and two in the control had been landing in `unsupported_answer`. Both numbers are
reported: with the original markers the primary is 9 → 37, with the corrected ones 11 → 50. The
correction was written after reading the answers, applied to both arms by the same classifier, and
it changed no answer, only their labels: outcomes are recomputed from the stored text at analysis
time rather than frozen at run time, which is the whole reason a relabel is possible without a
rerun.

The new rule keeps weak phrases from over-firing: "not found" or "не удалось" count as a refusal
only when the answer also names its sources, because "if the key is not found, nil is returned" is
ordinary technical prose.

## Caveats

- 0.50 is calibrated on this corpus and dies on re-index, like every other threshold here.
- The cost lands on `curated`, the hand-written questions, not on the paraphrases, and counting
  only refusals understates it: 2 questions were refused and 2 more ended as a narrated tool call,
  whose text the pipeline replaces with a bare "No relevant documents found". Four answers lost out
  of 30 is 13%, with a Clopper-Pearson interval running from 4% to 31%. The veto was pre-registered on the
  paraphrase set and passed there at 0 of 100, but a paraphrase of the corpus never comes near the
  threshold at any value in the grid, so that 0 measures very little. The real price is the 4 of
  30, and the next measurement of this veto belongs on a hand-written set grown to a hundred,
  before corpus hygiene moves every threshold anyway. The two questions the axis refused were
  "what is the difference between kafka and rabbit?" and "top 3 features of PG?", both of which the
  corpus does cover; conversational phrasing is what put them past 0.50.
- Grounding on the off-domain answers that survive rises from 0.62 to 1.45 out of ten, but the
  judge scored 81 answers in one arm and 31 in the other. That is not a quality result, it is
  arithmetic: the axis removes the questions the model was answering from nothing, and what
  remains is a smaller, differently composed set.
- The off-domain set is written by one person, and the subgroup split above is coarse (keyword
  matching over question text).
- The judge does not score refusals, so the axes above are computed on what remains, and the two
  arms leave different amounts to score.
