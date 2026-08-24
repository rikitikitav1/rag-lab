# 2026-08-24 - Phased eval runs, and the unload that freed nothing

Question: an eval run pays for the reranker twice - once in latency, once in evicted models.
Can the same run get the card to itself, phase by phase, without touching quality?

## What changed

The per-question loop became three phases over the whole set: embed and search everything,
rerank everything, generate everything. Between phases the ollama models are unloaded
(`keep_alive: 0`) and the reranker is dropped, so each phase owns the GPU alone. The reranker
also stopped being called per question: all pairs of the run (100 questions x 20 candidates =
2000) go through one `predict`, and the scores are sliced back per question.

Interactive answers were deliberately left alone: a single question cannot be batched, and the
agent decides for itself when to search.

## Result: 3.2x, and most of it came from one line

Same 100 questions, rerank on, k=5, judge fixed at qwen2.5:7b:

| run | retrieve | rerank | generate | total |
|-----|----------|--------|----------|-------|
| CPU rerank, per question (Jul) | - | - | - | 2092s |
| GPU rerank, per question (Jul) | - | - | - | 1161s |
| phased, batched | 51.0s | 44.7s | 1367.5s | 1463s |
| **phased, batched, `gc.collect()` before `empty_cache()`** | **7.8s** | **31.2s** | **612.9s** | **652s** |

The third row is the interesting failure. Phases were in place, the batch was in place, and the
run got *slower* than the naive July one. The reason was in ollama's log at the moment
generation started: `offloaded 26/33 layers to GPU`. Our `unload()` dropped the reference and
called `torch.cuda.empty_cache()`, which frees cached blocks but not a live object - measured
directly: 1144 MB allocated before unload, 1144 MB after, 9 MB once `gc.collect()` ran. Ollama
saw an occupied card and loaded llama two thirds onto the GPU and one third onto the CPU.
Generation halved when the memory was actually returned - a component we never touched.

Note that batching alone bought nothing on CPU (54.5s -> 57.2s on a 6-question probe):
sentence-transformers already batches internally by 32, so a flat list only removes call
overhead. Batching pays off on the card, where 2000 pairs finish in ~31s against roughly 16
minutes on CPU.

## Quality: unchanged, and that is the point

| run | faith | rel | compl | hit@5 | MRR |
|-----|-------|-----|-------|-------|-----|
| CPU rerank, per question | 7.57 | 8.55 | 6.27 | 0.95 | 0.885 |
| GPU rerank, per question | 7.67 | 8.78 | 6.08 | 0.95 | 0.885 |
| phased, batched, gc | 7.59 | 8.56 | 5.98 | 0.95 | 0.885 |

`hit@5` and `MRR` are identical to the third decimal across all three epochs. That is the
assertion that matters: had the batch sliced scores back to the wrong questions, ranking would
have drifted and MRR would have moved first. It did not, and fp16 on the card produced the same
ordering as fp32 on the CPU. Paired stats between the two GPU epochs: all three generation axes
have intervals covering zero (p = 0.28 to 0.51), so the 0.1-0.3 wobble is judge noise between
runs, not an effect.

## Recorded

- Runs now snapshot `phased` and `rerank_device` into `metrics.config`, next to `rerank` and `k`.
  In phased mode `elapsed` covers generation only, so without the flag the two eras would not be
  comparable.
- Lesson for the ops list: the stack degrades silently. Nothing failed, nothing warned, the run
  was simply twice as slow, and only a measurement showed it.

## Postscript: what an external audit found in the same change

Five defects, all real, all fixed in `d5595a7`. Two of them would have quietly corrupted this
very experiment had it been run differently:

- **The unload asked for the wrong model.** It expired the model assigned to the role, not the
  one the run actually used, so any sweep with a model override (`param=model`) would have
  started the rerank phase on an occupied card - the exact condition the phases exist to avoid.
  The run measured above used the role default, which is why the numbers hold.
- **The log recorded intent, not reality.** `rerank_device` came from the environment plus
  "is a card visible", so a run that hit CUDA OOM and silently continued on CPU would still be
  filed as a GPU run. A comparison across such runs would have been meaningless while looking
  perfectly clean. The device now comes from the model that is actually resident.

The rest: the OOM fallback repeated the same free-nothing bug that made the phases slow in the
first place; the retrieval phase embedded the whole set in one request with no error handling
(2565 questions under a 120s timeout, one failure costing three full worker retries); and
cancellation was only checked inside generation, so a cancelled job still finished the entire
rerank while holding the GPU.

Worth recording as a pattern: three of the five are the same species as the bug this experiment
started with - something that keeps working, reports success, and is simply wrong or slow. None
of them would surface without a measurement or an explicit check.
