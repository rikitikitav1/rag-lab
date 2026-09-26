# 2026-09-13 - Same generator on ollama and vLLM: not worse, faster, default unchanged

Moving the generator from ollama to vLLM would free the stand from one engine's drift and put the
answering path on the engine the judge already uses. The question is whether the answers get better
or worse for it. The pair measured here changes the engine together with the quantisation, since the
same 7B in the same precision does not exist on both, and it runs each arm twice so the difference
between engines can be read against each engine's own spread.

## Setup

**Set** 100 `paraphrased_ru` questions, the corpus pool · **corpus** `clean_1024`, `single_shot` ·
**generators** `Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM and `qwen2.5:7b` Q4_K_M on ollama, both with a
repetition penalty of 1.1 (on vLLM set per run through `generation_sampler`, on ollama held in the
model), so `compare_pools` reports `one_answering_penalty: true` for every pair · **judge**
`Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM, JSON without free whitespace, all four arms in one residency,
no judge reply cut · runs `arc5_engine_pair_{vllm,ollama}_{a,b}`, jobs 2816-2819, judged by 2820-2822
and 2824

No prediction was written before this pair; the result below is recorded without one.

## Result

ollama minus vLLM, paired over the same questions; verdicts moved and answers equal byte for byte
over all scored rows.

| pair | n | faithfulness | relevance | completeness | verdicts moved | equal answers |
|---|---|---|---|---|---|---|
| vLLM a against ollama a | 96 | +0.24 [-0.17, +0.66] | +0.10 [-0.29, +0.51] | +0.09 [-0.16, +0.37] | 101 of 288, 35.1% | 0 of 100 |
| vLLM b against ollama b | 95 | +0.10 [-0.24, +0.43] | +0.02 [-0.44, +0.53] | +0.16 [-0.19, +0.54] | 109 of 285, 38.2% | 0 of 100 |
| vLLM floor, a against b | 95 | +0.11 [-0.20, +0.44] | -0.04 [-0.31, +0.20] | -0.11 [-0.41, +0.17] | 78 of 285, 27.4% | 8 of 100 |
| ollama floor, a against b | 100 | -0.08 [-0.45, +0.27] | -0.05 [-0.37, +0.28] | -0.01 [-0.32, +0.29] | 101 of 300, 33.7% | 2 of 100 |

Every interval crosses zero, the smallest p is 0.41. On faithfulness alone 52 of 96 and 50 of 95
verdicts move between the engines, against 37 of 95 and 49 of 100 between two runs of one engine.

## Reading

- **On means over 100 rows the engines cannot be told apart.**
- **Row by row the answers always differ between engines** (0 of 200 equal), and verdicts move 1.5 to
  4.5 points more than ollama's own floor and 8 to 11 points more than vLLM's. Whether that excess
  is larger than the noise between two measurements of a floor was not checked, so it is not claimed
  as an engine effect.
- **vLLM is faster**: 7.7 and 7.6 s a question against 9.5 and 8.5 (p50 7.1 against 8.2 and 8.1), a
  run of 800 and 785 s against 965 and 870 s, at about the same number of answer tokens (32.1 and 30.5
  thousand against 35.0 and 30.2 thousand).
- **Chinese characters in a Russian answer** turn up equally on all four arms, 9 to 11 rows of 100: a
  trait of qwen2.5 7b, not of an engine.

## The refusals were a rule, not an engine

vLLM "refused" 5 and 6 questions, ollama 1 and 0. Both arms open with "the context does not contain"
on the same 7 or 8 questions. vLLM stops there, at 290 to 380 characters; ollama carries on with an
answer (6 of 7 and 8 of 8 run past 400 characters). The stand reads a refusal only under 400
characters ([the refusal entry](2026-09-13_a-refusal-the-rule-could-not-read.md)), so the same opening
is a refusal on one arm and an answer on the other. That is a difference of style in how long an
answer runs after the hedge, not of how either engine reads the context.

## Decision

- The generator's default stays `llama3.1:8b` on ollama. This pair shows vLLM as a generator is not
  worse on means and is faster; it does not show it is better.
- The refusal rule is not changed for this: its ceiling of 400 characters is a known limit, recorded
  with the rule.

## Caveats

- The pair changes the engine and the quantisation together (AWQ against Q4_K_M), as said above.
- The judge is the same AWQ model that answers on the vLLM arm. A judge that favours its own weights
  would lift the vLLM arm; the means show no lift, but the pair cannot rule the effect out.
- 100 Russian paraphrases from the corpus only; nothing outside the corpus.
- The two engines' floors were each measured once, so the excess of movement between engines over
  either floor has no spread of its own.
