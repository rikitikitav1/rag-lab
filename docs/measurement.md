# Measurement

How the bench tells a change from noise. The reasons the numbers hold are listed in the [README](../README.md#why-the-numbers-hold); this page keeps what grows with each new instrument.

## How much each instrument moves on its own

How much each instrument moves on its own, measured by running it twice on the same material. A verdict is one judge score on one axis of one answer; a reload is the model unloaded from its server and loaded again.

| Instrument | Moved between two passes | Entry |
|---|---|---|
| judge Qwen2.5-7B on ollama, across a reload | 14% of scores, 58% of reason texts | [entry](experiments/2026-09-07_what-the-standard-was-worth.md) |
| judge Qwen2.5-7B-AWQ on vLLM, across a reload | 0 of 388 verdicts | [entry](experiments/2026-09-09_what-batch-invariance-costs-on-an-awq-judge.md) |
| the same judge in one load, JSON without free whitespace | 0 of 275 verdicts | [entry](experiments/2026-09-13_the-judge-that-looped-on-whitespace.md) |
| DeepSeek-V4-Flash as a judge, passes hours apart | 154 of 573 verdicts, 27% | [entry](experiments/2026-09-14_a-cloud-judge-on-the-same-answers.md) |
| generator Qwen2.5-7B-AWQ on vLLM, two runs | 78 of 285 verdicts, 27%; 8 of 100 answers equal | [entry](experiments/2026-09-13_the-same-generator-on-two-engines.md) |
| generator qwen2.5:7b on ollama, two runs | 101 of 300 verdicts, 34%; 2 of 100 answers equal | [entry](experiments/2026-09-13_the-same-generator-on-two-engines.md) |
| generator llama3.1:8b on ollama, two runs | about a point per row; 4% of answers equal | [entry](experiments/2026-09-13_a-bigger-model-at-the-same-retrieval.md) |
| generator DeepSeek-V4-Flash, two runs | 0 of 50 answers equal; faithfulness moved on 18 of 44 | [entry](experiments/2026-09-13_a-bigger-model-at-the-same-retrieval.md) |
| a human rater (the author), the same pairs a day later | the same choice in 5 of 7 | [entry](experiments/2026-09-13_fifteen-pairs-the-owner-judged.md) |

A difference smaller than its instrument's own movement is read as noise.
