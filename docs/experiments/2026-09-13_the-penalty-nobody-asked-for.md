# 2026-09-13 - The penalty nobody asked for, and a door that drops it

Both engines on this stand apply a repetition penalty that no request asks for, each its own, and
ollama's OpenAI-compatible door silently drops the one a request does name. None of this shows in a
request log, and all of it moves answers. This entry records what each server applies by default,
measured by byte equality of the answer, and what the stand now does about it.

## Setup

**Input** one row, 41941 (grid, `llama3.1:8b`, `paraphrased_ru`, corpus `clean_1024`, an answer of
about 300 tokens): the active `generate_answer` prompt with that row's context and "Respond in
Russian."; on vLLM also the judge's faithfulness prompt 2 with its JSON schema · `temperature` 0,
`seed` 0, a ceiling of 400 tokens (300 on the processor) · **servers** ollama 0.32.0 on the card and
on the processor, vLLM 0.29.1 · read only, nothing written to the base

Each cell is one call, and cells are compared by the first four characters of the answer's sha1. The
penalty is varied: absent, 1.0, 1.1 (vLLM: 1.05), and on ollama also through its OpenAI door without
and with 1.0.

## Result

| engine, model | absent | 1.0 | 1.1 (vLLM 1.05) | OpenAI door: absent / 1.0 |
|---|---|---|---|---|
| ollama, card, `llama3.1:8b` | `8a39`, 283 tokens | `5977`, 131 | `8a39`, 283 | `8a39` / `8a39` |
| ollama, card, `qwen2.5:7b-w16384` | `9f9f`, 343 (first call) | `7a77`, 333 | `6c0f`, 367 | `6c0f` / `6c0f` |
| ollama, processor, `qwen2.5:7b` | `0fda`, 300 | `26d3`, 300 | `0fda`, 300 | `0fda` / `0fda` |
| vLLM, judge model, generation prompt | `8a67`, 279 | `788c`, 337 | `8a67`, 279 | n/a |
| vLLM, judge model, judge prompt | `6fb1`, 102 | `5b05`, 49 | `6fb1`, 102 | n/a |

Point observations on one input, no interval: the claim is equality of bytes, and one call per cell
is enough to show that "absent" is not "1.0".

## What the servers do

- **ollama applies 1.1 when nothing is asked.** "Absent" equals 1.1 and differs from 1.0, on the card
  and on the processor. None of our models sets it in its Modelfile; it is the server's default.
- **ollama's OpenAI door drops the penalty.** Through `/v1/chat/completions` an explicit 1.0 returns
  the same bytes as no penalty at all, that is, 1.1.
- **vLLM applies 1.05 when nothing is asked**, taken from the model's `generation_config.json`, on
  the generation prompt and on the judge's. An explicit 1.05 does not move the judge.
- The two engines' defaults differ, so any comparison of an arm on ollama with an arm on vLLM was
  also a comparison of 1.1 against 1.05 until the stand named both.

## The first call answers differently

On both engines the first call after a model loads, or the first with a new prompt, returned other
bytes than every call after it (llama: first `31c2`, 287 tokens, then `8a39`; qwen on the card
`9f9f`; the vLLM judge first `a36f`, 106 tokens, then `6fb1`). The order absent, absent, 1.1, absent,
1.0, absent gave `31c2`, `8a39`, `8a39`, `8a39`, `5977`, `8a39`, which reads like a cache of the
prompt's shared beginning. At scale on the judge it did not show: two passes in one residency over
the same 275 verdicts differed in none, although the first pass began on a cold cache
([the grammar entry](2026-09-13_the-judge-that-looped-on-whitespace.md)). The effect is real and
rare: inside the judge's measured floor on vLLM, and on ollama indistinguishable from the engine's
own drift. Prefix caching stays on.

## Decision

- Both values are named in `config.yaml` (owner's call: explicit over implicit, or the next ollama
  release moves the stand without a word). They are the values the roles already ran with, so no
  recorded number moves.
- vLLM: `repetition_penalty: 1.05` in the judge role's options, sent in the request body.
- ollama: `llm.repetition_penalty: 1.1`. The door drops a per-call value, so the stand writes it into
  each role's model through `/api/create` from the same model under the same name, after every pull
  and at start, and only when the model does not carry it yet. Embedders are skipped.
- Every run's stamp carries the penalty each engine actually applied, and `/v1/health/stand` shows
  the declared value beside what the servers hold, with a `drift` list. `compare_pools` says per arm
  which penalty its generator ran with and whether the arms share it.
- Commit "Name the repetition penalty in config.yaml and in the stamp, show each arm's, and let a
  run set its generator's sampler for itself alone".

## Caveats

- One input row. The defaults are properties of the servers, and one row is enough to read them;
  how much 1.1 against 1.0 moves scores on a set was not measured here.
- ollama 0.32.0 and vLLM 0.29.1. Another release may change either default; the stamp records the
  server's version beside ollama's value for that reason.
- The first-call effect was read on a handful of calls and checked at scale on the judge only, not on
  a generator.
