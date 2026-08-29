# 2026-07-27 - Generation prompt v3 → v4 (drop inline citations)

The three citation rules in v3 look like they buy nothing the structured `sources` field does not
already carry, and they cost answer cleanliness: inline `[source]` markers appear unevenly across
runs, so as a consumer-facing contract they are unreliable anyway. Dropping them should leave
grounding untouched if the load-bearing rules are the ones about using only supported information.

## Setup

**Set** `paraphrased_ru` (n=100) · **corpus** pre-variant era, post-ablation · **judge** `qwen2.5:7b`, categorical

`generate_answer` v3 → v4 removes exactly three lines ("cite every factual statement in the format
[source]", "do not cite a source unless it directly supports the statement", "cite all relevant
sources when several support a statement"). Every other rule is byte-identical.

Scope is single-shot only: `generate_answer` is read by `chat.answer`, while the agent synthesises
through `agent_system`, which still instructs inline citation. That asymmetry is deliberate for
now, since changing the agent prompt would need its own agent eval and folding it in would confound
the two.

The judge does not move by construction: `judge_faithfulness.v1`, `judge_relevance.v1` and
`judge_completeness.v1` contain no reference to citations or `[source]` markers.

Generator `llama3.1:8b` at temp 0, rerank off, on the roadmap-removed corpus, with a v3 control
(`v3_current`) run alongside v4 (`v4_ru`). Retrieval is prompt-independent and came out identical
for both (91% / MRR 0.837, n=100).

## Result

| axis | v3 (inline citations) | v4 (no citations) |
|------|-----------------------|-------------------|
| faithful / partially / unfaithful | **56 / 29 / 15** | 49 / 30 / 21 |
| relevant / partially / irrelevant | **68 / 28 / 4** | 56 / 35 / 9 |
| complete / partially / incomplete | 2 / 87 / **11** | 1 / 82 / 17 |

v4 is worse on all three axes, most on relevance (68 → 56, irrelevant 4 → 9) and faithfulness
(unfaithful 15 → 21). The hypothesis was wrong: the citation rules were load-bearing rather than
cosmetic. Forcing an 8B to attribute every statement to a source keeps the answer tied to the
retrieved context; removing it let the model drift into less grounded prose. The uneven inline
markers were a symptom of a weak model following the instruction imperfectly, not a reason to drop
the instruction.

## Decision

Revert. v4 deleted (file and DB row), v3 stays active. The consumer-facing contract is handled at
the API layer instead: the structured `sources` list is authoritative, and the prose may still
carry inline markers unevenly. Consistent inline citation is a generator-capability lever, not a
prompt-removal one.

## Caveats

- retrieval here (91% / 0.837) is higher than the pre-ablation `v3_ru` baseline (80% / 0.652); that
  is the corpus change, not the prompt
- counts, not scores, single run per arm, no interval
