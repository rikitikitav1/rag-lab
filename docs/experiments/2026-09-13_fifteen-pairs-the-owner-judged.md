# 2026-09-13 - Fifteen pairs the owner judged, read again by the new judge

Every other check of the judge on this stand says whether it repeats itself. Only a person can say
whether it is right. On 08.09 the owner picked the better side in 15 pairs of answers, and the judge
of that time agreed on 4. Since then the judge moved to vLLM and to JSON without free whitespace, so
the same pairs were judged again by the new judge to see whether it agrees with the person more, less
or the same.

## Setup

**Sheet** `20260908`, 15 pairs from the corpus pool, both sides answered and scored, one side from
`arc3_ported_after_split` and the other from `arc3_agent_baseline` on the same question
(`datasets/measurements/human_anchor_arc4_20260908.json`) · subgroups declared before the sheet was
filled: same-language against cross-language pairs (10 and 5), clean answers against answers with a
leak of prompt scaffolding (10 and 5) · **the owner's repeat** the next day, 7 of the pairs shown
again with the sides swapped (`human_anchor_repeats_pairs_20260908_20260909.json`) · **new judge**
`Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM, JSON without free whitespace, repetition penalty 1.05, over
copies of the 15 questions of each side: experiments 59 and 60, kind `rejudge`, equal answers ·
read with `scripts/human_anchor.py read 20260908 --copy`, which takes the judge's difference on each
pair from the copies

## Result

Agreement with the owner's choice, 15 pairs.

| judge | agreed | missed (called equal or not separated) | against |
|---|---|---|---|
| old judge, `qwen2.5:7b` on ollama | 4 (27%) | 5 | 6 |
| new judge, Qwen AWQ on vLLM | 7 (47%) | 4 | 4 |
| RAGAS guest, faithfulness | 4 (27%) | 5 | 6 |

New judge by the declared subgroups: same language 5 of 10, cross language 2 of 5, clean 6 of 10,
with a scaffolding leak 1 of 5.

The owner against the owner: of the 7 pairs shown again with the sides swapped, the same answer was
chosen in 5; in the other 2 one of the two readings called the sides equal. A judge cannot be
expected to agree with a person more often than the person agrees with themself, and on this sheet
that ceiling is below 100%.

The judges' means on the copies did not move against the old verdicts on either side (faithfulness
-0.27 [-1.73, +1.07] and +0.47 [-0.60, +1.73], relevance and completeness within 0.2, none
significant, n=15 each).

## Reading

A judge that names a side on every pair agrees with a coin half the time. 7 of 15 is the level of a
coin; 4 of 15 is below it. On 15 pairs the interval is about 25 points either way, and telling 47%
from 65% would take about 80 pairs. So the new judge is not worse than the old on the person, and on
15 pairs it cannot be told from a random choice.

The anchor catches a gross inversion, a judge that prefers the worse answer almost every time. It is
not a calibration of the judge.

## Decision

- The judge's default does not change on this: nothing here speaks for or against it.
- Widening the sheet to the size that would separate 47% from 65% costs the owner's time, and it is
  the owner's call.

## Caveats

- 15 pairs and one person. The subgroups hold 5 to 10 pairs each and are shown for completeness, not
  read.
- The old judge's and the guest's differences come from the sheet's own key, taken when it was built;
  the new judge's come from copies judged on 13.09.
- The pairs come from arc 3's answers; the corpus and the generator of that time, not today's.
