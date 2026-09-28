# 2026-09-27 - Comparing question generators

For the next arc, a cloud model will generate questions from sections in the corpus. Each section gets one call,
and each fact gets a pair of questions, one in English and one in Russian. This entry compares the candidate
generators using the same sections and prompt, with the reader blind to which model produced each question. It
reports the measurements but does not draw conclusions.

## Setup

**Sections:** 20 sections from `redis-doc`, selected with seed 20260927, split evenly between `commands/` and `docs/`. **Reader:** a Claude agent, blind to the model. **Services:** neuraldeep (free tier), gonka, and the stand's own ollama.

**Two prompts.**
- **Prompt 1.** Five en/ru pairs on every section, whatever its length.
- **Prompt 2**, rewritten after the first blind review:
  - sections under 60 words are skipped (5 of 20), so 15 sections remain;
  - pairs scale with length: one pair under 150 words, two under 400, three under 900, four above;
  - the answer must be stated in the section;
  - the question is asked in the asker's own words, not the section's phrasing, and is not answered by the heading or the command name;
  - no meta questions ("what does the documentation say");
  - the Russian is not a translation of the English.

The expectations for Prompt 2 were recorded in the arc log before the run.

**Labels**, one set for every item:

| Label | Values | Meaning |
|---|---|---|
| answerable | yes / partial / no | whether the gold section states the answer |
| tests retrieval | yes / no | false when the heading or the command name alone answers it, or when it is general knowledge |
| natural | yes / no | whether a developer would ask it this way |
| Russian clean | yes / no | read on the Russian items only |

**Reader's sample:**
- Prompt 1: 25 English and 25 Russian items a model, drawn by seed.
- Prompt 2: every item.

**Scripts:** `temp_files/question_probe.py`, `temp_files/question_probe_v2.py` (calls), and `temp_files/question_probe_read.py` (automatic columns and blind sample).

**Data files:** `datasets/measurements/question_probe_20260927*`, `question_probe_v2_20260927*`, and `question_probe_v2_more_20260927*` (gonka and local).

## Result

All figures are point estimates based on one reader's labels. Here, n is the number of items labelled for each model.

**Prompt 1** (20 sections, n = 50 a model, 25 of them Russian)

| model | cloud | answerable yes / partial / no | tests retrieval | natural | Russian clean | median s a call | tokens in / out |
|---|---|---|---|---|---|---|---|
| gemma-4-31b | neuraldeep | 48 / 2 / 0 | 37 of 50 | 48 of 50 | 25 of 25 | 72.7 | 9 103 / 25 152 |
| qwen3.8-27b | neuraldeep | 43 / 7 / 0 | 42 of 50 | 46 of 50 | 24 of 25 | 19.3 | 9 530 / 17 584 |
| qwen3.6-35b-a3b | neuraldeep | 41 / 5 / 4 | 44 of 50 | 45 of 50 | 24 of 25 | 28.6 | 8 930 / 74 998 |
| gpt-oss-120b | neuraldeep | 41 / 7 / 2 | 41 of 50 | 45 of 50 | 21 of 25 | 26.0 | 9 442 / 15 145 |

**Prompt 2** (15 sections; qwen3.8-27b has 44 items because one of 15 calls did not parse; the other models have 46; half are Russian)

| model | cloud | answerable yes / partial / no | tests retrieval | natural | Russian clean | median s a call | tokens in / out |
|---|---|---|---|---|---|---|---|
| gemma-4-31b | neuraldeep | 46 / 0 / 0 | 40 of 46 | 46 of 46 | 23 of 23 | 88.0 | 9 003 / 16 883 |
| qwen3.8-27b | neuraldeep | 44 / 0 / 0 | 42 of 44 | 42 of 44 | 19 of 22 | 19.8 | 9 292 / 13 013 |
| qwen3.6-35b-a3b | neuraldeep | 44 / 2 / 0 | 40 of 46 | 44 of 46 | 22 of 23 | 17.6 | 8 842 / 46 667 |
| DeepSeek-V4-Flash-0731 | gonka | 41 / 3 / 2 | 46 of 46 | 44 of 46 | 22 of 23 | 5.1 | 8 675 / 4 094 |
| MiniMax-M2.7 | gonka | 40 / 0 / 0 | 40 of 40 | 40 of 40 | 19 of 20 | 42.6 | 7 807 / 27 040 |
| gemma2:9b | local, ollama on the card | 36 / 6 / 2 | 40 of 44 | 40 of 44 | 16 of 22 | 6.8 | 8 482 / 1 415 |
| llama3.1:8b | local, ollama on the card | 22 / 6 / 2 | 30 of 30 | 26 of 30 | 10 of 15 | 3.6 | 8 497 / 2 120 |

The item counts differ because a call was included only if it returned the requested number of pairs. MiniMax-M2.7
completed 13 of 15 calls; two remained unanswered after three waits for the cloud's `Retry-After`. gemma2:9b
completed 14 calls, and llama3.1:8b completed 7 of 15, often returning the wrong number of pairs. The neuraldeep
results and the gonka and local results were labelled in two blind batches by the same reader using the same labels.
The batches did not include each other's items. Local models ran in the stand's ollama with an 8192-token window.

The two prompts were not tested on the same set: they used different sections and question counts, and the reader
reviewed different samples.

**Not run:** gpt-oss-120b was not tested again with Prompt 2. Kimi K2.6 is not available on neuraldeep's free tier.

## What the reader saw across models

Failure types identified by the reader, counted across all models for each prompt. The review files contain the item IDs.

| kind | prompt 1 (200 items) | prompt 2, neuraldeep (136) | prompt 2, gonka and local (160) |
|---|---|---|---|
| not answerable from the section | 6 | 0 | 6 |
| answered by the heading or command name | about 18% | 10 | 4 |
| meta question about the document | 6 | 0 | 0 |
| squeezed out of a one-sentence section | 32 on three stubs | stubs skipped | stubs skipped |
| the Russian item translates an English item | most | most | most |

## Decision

No decision was made in this entry.

## Caveats

- The test used one corpus (`redis-doc`) and 15 to 20 sections. This sample cannot reliably distinguish a lead of one or two items between models.
- One Claude agent labelled every item. There was no second reader, so agreement between readers was not measured.
- This entry does not evaluate the question set generated for the next arc. Whether those questions distinguish good retrieval from poor retrieval remains to be measured.
- Cloud response times reflect the service on the day of the run, not just the model.
