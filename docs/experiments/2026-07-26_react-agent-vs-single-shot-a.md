# 2026-07-26 - ReAct agent vs single-shot (a measured loss)

A ReAct agent that decides its own retrieval (search when needed, refine, multi-hop) should answer
at least as well as single-shot RAG, and better on questions that need decomposition. A new judge
axis, completeness (answer against the question's reference answer), is added for this run:
faithfulness and relevance cannot see the advantage the agent is supposed to have.

## Setup

**Set** `paraphrased_ru` (n=100) · **corpus** pre-variant era, `developer-roadmap` still in · **judge** `qwen2.5:7b`, categorical

Agentic pipeline (`agent.run`, ReAct loop over a `search_corpus` tool, `agent_system` prompt)
against single-shot (`chat.answer`, `generate_answer` prompt). Same questions, same gold labels,
same judge in both arms.

## Result

| axis | base_ru_t0 | v3_ru (single-shot) | agent_ru |
|------|-----------|---------------------|----------|
| retrieval hit@k | 80% | 80% | **31%** |
| retrieval MRR | 0.652 | 0.652 | **0.277** |
| faithful | 38 | 50 | **23** |
| unfaithful | 28 | 24 | **55** |
| relevant | 35 | 57 | 47 |
| complete / partially / incomplete | 0/73/27 | 1/85/14 | 1/72/**27** |
| avg hops | - | - | 2.03 (max 3) |

The agent lost on every axis, and the cause is retrieval rather than generation: 54 of 100 agent
answers had zero retrieved sources. Every question triggered a search, so the agent did not skip
retrieval; its self-formulated queries came back empty about 54% of the time while single-shot
embeds the full question and retrieves at 80%. Empty context then propagates: unfaithful jumps to
55, because the model answers ungrounded when the search returns nothing. Only 3 of 100 runs took
three hops, so there is no decomposition benefit to offset the loss.

## Correction, measured the same day: the loss was a bug, not the agent

Inspecting the actual tool-call arguments disproved the first diagnosis (queries drifting past the
distance threshold). The empties came from somewhere else: the 8B always filled the optional
`category` parameter with a hallucinated ltree path (`numpy.array_operations`, `databases.amazon`,
`angularjs`), and `category ~ lquery` returns zero rows for paths that do not exist in the
taxonomy. The query text itself was fine. Even the rare non-empty case was wrong-domain: a
`databases.redis` guess returned redis chunks for a reinforcement-learning question.

Fix: remove `category` from the tool schema offered to the model. The underlying function still
supports it. The general lesson is about tool design rather than this agent: never offer a
free-form parameter the model cannot ground. Use an enum over a small fixed set, or a discovery
tool. Here the taxonomy is too large for an enum, so the parameter goes.

Re-run with the category removed (`agent_nocat_ru`):

| axis | v3_ru (single-shot) | agent_ru (category trap) | agent_nocat_ru (fixed) |
|------|---------------------|--------------------------|------------------------|
| retrieval hit@k | 80% | 31% | **84%** |
| retrieval MRR | 0.652 | 0.277 | **0.773** |
| faithful | 50 | 23 | 37 |
| unfaithful | 24 | 55 | 32 |
| relevant | 57 | 47 | **70** |
| complete / partially / incomplete | 1/85/14 | 1/72/27 | **2/89/9** |
| empty-source answers | - | 54 | **0** |

"The agent is strictly worse" was an artifact of the category bug. With the trap removed the agent
beats single-shot on retrieval (84 against 80, MRR 0.773 against 0.652), relevance (70 against 57)
and completeness (incomplete 9 against 14): its reformulation, often ru into en, retrieves better
cross-lingually than embedding the raw Russian question. It loses on faithfulness (37 against 50
faithful, 32 against 24 unfaithful): answering more fully from a multi-hop context union yields
more assertions the judge marks ungrounded.

## Correction, measured after the second review round (`agent_v2_ru`)

The agent and judge path was hardened: `answered` now follows retrieved evidence (honest refusal on
empty), the judged context excludes the "No relevant documents found." and tool-error sentinels,
judging is per axis, and `dispatch` drops any hallucinated `category`.

| axis | v3_ru (single-shot) | agent_nocat_ru | agent_v2_ru (hardened) |
|------|---------------------|----------------|------------------------|
| retrieval hit@k | 80% | 84% | **87%** |
| retrieval MRR | 0.652 | 0.773 | **0.779** |
| faithful | 50 | 37 | 39 |
| unfaithful | 24 | 32 | **29** |
| relevant | 57 | 70 | 68 |
| complete / partially / incomplete | 1/85/14 | 2/89/9 | **5/90/5** |
| empty-source answers | - | 0 | 0 |

Filtering non-evidence out of the faithfulness context is what moved unfaithful from 32 to 29 and
faithful from 37 to 39: the judge had been scoring some answers against a context that literally
said "No relevant documents found." The fix improved the measurement, not the model.

## Decision

Single-shot v3 stays the default for now, and the agent's faithfulness gap becomes the next lever
(retrieval width, then hop cap). Follow-ups on the backlog: search with the original question
alongside the reformulation, and judge the agent against a deduped context.

## Caveats

- retrieval hit@k and MRR are computed identically for both pipelines, but for the agent this is
  recall across hops rather than precision@k, so the comparison is not clean. In this run the
  caveat cuts the agent's way: it barely multi-hops and its searches often came back empty, so the
  number is deflated, not inflated
- three numbers shifted meaning across the rounds and are not comparable between them: agent MRR
  reflects cross-hop source dedup added in `24fae88` (membership, and so hit@k, is unaffected); `hops`
  now includes the forced synthesis turn, so a run capped at N can report N+1; faithfulness and
  relevance are counted over in-corpus logs while completeness is counted over all logs, so the
  three axes do not share a denominator
- counts, not scores: the categorical judge of this era was later shown to invert conclusions
