# 2026-09-20 - Rewriting the question when the search looks weak

When retrieval comes back weak, Corrective RAG (Yan et al., 2024) rewrites the query and searches
again. This entry records what that buys on our corpus, measured without a generator and without a
judge, and why the rewrite node stays off by default.

**This is not a reproduction of the paper.** CRAG decides with a fine-tuned evaluator, refines
documents strip by strip and, when retrieval looks wrong, searches the web. We have no web search and
no trained evaluator: the decision is a threshold on the vector distance the stand already calibrated,
and the rewrite is one prompted call on the model that serves. What follows says what that buys here.

## The trigger, declared before anything ran

The node cannot fire on the grader's confidence: measured on the recorded verdicts of the previous
entry, that signal has precision 0.6 and recall 0.08 to 0.27 on the task "retrieval missed", and it
costs 4.4 seconds a question. Two signals that cost nothing were measured on the frozen pool instead,
and the trigger was declared before the run:

**the nearest chunk sits at or beyond 0.39, or the five served chunks lie within 0.017 of each other.**

The first number is not tuned here: it is `agent.gate.weak_distance`, the stand's own definition of a
weak search, calibrated on another question set three weeks earlier. The second is the range of the
served distances, chosen on the sweep over all 820 and checked afterwards on halves: cuts tuned on one
half keep precision 0.42 to 0.44 on the other.

On the 820 English questions the trigger fires on 136 rows (0.166), with precision 0.419 and recall
0.352 against "the gold section is not in the top five". It reads the nearest of the served distances,
which is the quantity the stand's own fallback gate reads. Two signals were rejected with their numbers:
the fusion score of the top chunk is degenerate here (801 rows of 820 carry the same value, because the
keyword arm returns a rank on 41 candidates of 16394), and "all five chunks come from one file" points
the wrong way, firing on rows where retrieval usually hit.

## Setup

**Population** the 820 English questions of `paraphrased_v2`, corpus variant `clean_1024`, 12102
chunks, embedder `bge-m3` on ollama, the top five by fusion, which is what the serving path retrieves ·
**rewrite** `llama3.1:8b` on the generation seat, prompt `rewrite.question` v1, temperature 0, one
sentence through a JSON schema · **treatment** on a fired row the rewritten question is searched and
its top five replaces the first, which is CRAG's own action for "retrieval is wrong" · **runs** job
2876 (820 rows, 5 minutes) and 2877 (the repeat on the fired rows), both on code fingerprint
`0b38749e21c9`, no generator and no judge anywhere in the measurement.

Rows whose question sits beyond the topic gate are not rewritten: they stay refusals, and they stay in
the denominator unchanged. There were seven, gated at the English threshold the agent applies to an
English question, 0.4374. A later reading of this entry briefly moved the gate to 0.456 and the net to
+10 rows; that was a defect in the bench, not in the stand, and it was withdrawn the same evening. The
numbers above are the ones the preregistered design produced.

## What it bought

| | |
|---|---|
| the gold section in the top five, before | 658 of 820 (0.802) |
| recovered by the rewrite | 22 |
| lost to the rewrite | 14 |
| net | +8 rows, +0.0098 |
| interval, 10000 draws, paired by row | [-0.0049, +0.0244] |

The lower edge is below zero, so the bar this MR was read against is not cleared. On the coarser
predicate, whether the right file is in the top five at all, the delta is exactly zero: 13 rows gained,
13 lost. The recovery is real and so is the loss, and the corpus is too small to tell their difference
from zero: 22 against 14 discordant rows is a two-sided sign test of about 0.24, and an effect of one
point would need roughly 1850 questions to clear the interval.

**Where the losses come from.** The prompt tells the model to name the technology the question is about
in the words documentation uses, and where the question no longer implies one, an 8B model guesses.
"What kind of component remembers information from previous uses?" became "What is a non-volatile
memory component?": the answer lived in a React section, the rewrite guessed hardware, and the search
moved to deep learning. Six of the fourteen lost rows have a new top five that shares no file with the
old one, against three of the twenty-two recoveries.

**The instrument, twice.** The first search of the live pass reproduced the frozen pool exactly on
hits (658) and on the topic gate (7), and fired on 136 rows rather than 137: three rows sit within
0.0004 of a cut and cross it between two searches. A cut on a cosine distance is a point on a
continuum, not a cliff, and a fired set of this size is stable to about two rows.

**The floor of the pair.** The same rows rewritten again in the same residency give a net of +9 against
+8, one row apart, where the interval's own half-width is twelve rows. The text is far less stable than
the ranking: at temperature 0, only 81 of 129 rewrites came back byte-identical, and 84 of 129 produced
the same top five. Pass-to-pass wobble does not explain the interval; the size of the effect does.

## What is left in the tree, and what is not

**The node is not.** The owner removed it after this reading: the measured gain does not pay for a
second model call and a second search on one row in six, the model that would write the rewrites is an
8B one, and the code was one more branch in the graph for everybody to carry. What stays is this entry,
the measurement files it was computed from, and the trigger's numbers, which are reusable: a distance
of 0.39 on the nearest served chunk, or a top five within 0.017 of each other, picks a row that missed
about two times more often than chance.

Three readings that outlived the node. The reciprocal-rank fusion of two searches is bounded by
arithmetic: with `k = 60`, a chunk both searches found always outranks a chunk only one of them found,
so a gold section that only the rewrite finds can almost never reach the top five. Keeping the first
two chunks and filling the rest from the rewrite loses half as often as replacing the list, because it
cannot drop a gold at rank one or two. And the rewrite's own text is not reproducible: at temperature 0
in one residency, 81 of 129 rewrites came back byte-identical, while the ranking they produced moved
the net by one row.

If anybody returns to this, the cheap shape is the one measured here: retrieval only, on the direct
path, both ranked lists recorded at full depth, and a population of about 1850 questions rather than
820, which is what an effect of one point would need.
