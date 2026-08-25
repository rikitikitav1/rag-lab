# 2026-08-25 - The gate that fires, and the refusal that never comes

Three fallback policies, three pools of a hundred questions each, one variable. The previous
entry ended on a gate that never opened; this one is the rerun after the gate was taught to
open on weak retrieval instead of on empty retrieval, and after four rounds of review took the
measurement apart.

## Setup

Agent pipeline, llama3.1:8b, judge qwen2.5:7b, `agent.system` v5, `agent.fallback` v3,
`agent.tool_match` v3, one external integration (DeepWiki). Each policy answers the same 300
questions, split into pools that are scored separately:

| pool | what it is | what should happen |
|------|-----------|--------------------|
| in-corpus (100, `paraphrased_ru`) | questions the corpus answers | answer from the corpus, never leave |
| out-of-corpus (100) | repository internals absent from the corpus | leave, answer from the tool |
| off-domain (100) | cooking, chemistry, law, legacy stacks, post-cutoff tech | refuse |

The policies: `agent_choice` shows every tool from hop one, `corpus_first` withholds them until
the corpus returns nothing, `corpus_first_weak` also runs a cross-encoder gate over the top five
hits and treats a best score below 0.5 as a miss.

## What the run says

| | `agent_choice` | `corpus_first` | `corpus_first_weak` |
|---|---|---|---|
| in-corpus faithfulness / relevance | 7.36 / 9.03 | 7.00 / 9.11 | 7.07 / 9.28 |
| hit@5 / MRR | 0.92 / 0.825 | 0.92 / 0.799 | 0.92 / 0.799 |
| answers with an `mcp:` source | 53/100 | **0/100** | **60/100** |
| out-of-corpus grounding | 7.23 | - | 6.95 |
| off-domain grounding | 0.66 | 0.53 | 0.77 |
| off-domain refusals | **0/100** | **0/100** | **0/100** |
| narrated tool calls | 14 | 12 | 12 |
| answer rate | 0.933 | 0.950 | 0.953 |
| seconds per question, in-corpus / out / off-domain | 8.8 / 15.7 / 6.1 | 8.6 / 6.7 / 6.1 | 10.6 / 18.9 / 8.0 |

Paired, question by question:

| comparison | pool | axis | before → after | p |
|---|---|---|---|---|
| `corpus_first` → `corpus_first_weak` | out-of-corpus | faithfulness | 2.31 → 5.08 (49 better, 17 worse) | **<0.001** |
| | out-of-corpus | relevance | 6.82 → 7.60 | 0.039 |
| | in-corpus | faithfulness | 7.00 → 7.07 | 0.843 |
| | in-corpus | relevance | 9.11 → 9.28 | 0.655 |
| `agent_choice` → `corpus_first_weak` | out-of-corpus | faithfulness | 5.03 → 5.13 | 0.874 |
| | in-corpus | faithfulness | 7.36 → 7.07 | 0.251 |

## A gate works, and the signal behind it is not settled

The empty rule is still a null policy: `corpus_first` went outside zero times out of a hundred
questions it could not answer, exactly as the previous entry measured on twenty. The
cross-encoder gate turns that into 60 out of 100 and lifts grounding on those questions from
2.31 to 5.08, with 49 questions better and 17 worse. That is the largest effect this lab has
measured on the generative side.

What this run does *not* establish is that the cross-encoder is what earns it. Measured the day
after, the nearest-chunk distance ranks about as well for coverage (AUC 0.930 against 0.912 on
pool labels; inside the corpus the flagged questions average 6.00 against 5.82 faithfulness, a gap
of 1.37 against 1.60) while costing nothing, and the two signals flag different questions: twelve
of twenty-two overlap. Neither is a retrieval evaluator inside the corpus, where both catch five of
the ten badly answered questions at a twenty-two question budget. So the honest claim is that *a*
gate works, and which signal to keep is the next experiment: cross-encoder, distance, or either.

The in-corpus half does not move (p=0.84 and p=0.66), and hit@5 is identical across all three
arms. Note that the retrieval numbers only stay honest because the weak arm records the chunks it
hides from the model; scored naively, hiding context reads as a retrieval regression on questions
where the search was byte-for-byte the same.

Against the always-open baseline, `corpus_first_weak` is indistinguishable everywhere
(p from 0.25 to 0.87). So corpus priority is free in quality, not in time: the gate runs a
cross-encoder on the CPU for every search (+2 s per question, 8.6 to 10.6 in-corpus, +23%), and
the questions where it opens the toolbox cost an external hop (6.7 to 18.9 seconds, x2.8). Quality
parity with an always-open toolbox is bought with latency, and on this hardware that is the trade
worth naming out loud.

The layer that makes this work is not the gate alone. Inside the corpus the gate fired on 23 of
100 questions, and all 23 stayed inside only because the admission check refused to offer a
GitHub-repository tool for them: `drop_weak_context` was false in 235 of 300 rows. Without that
layer those 23 would have gone out with an invented `repoName`, which is exactly the 6.42 to 4.91
regression the previous entry measured. The gate decides that retrieval is weak; the admission
check decides whether anything else can do better.

## Nobody refuses

Zero refusals out of a hundred, in all three policies, on questions no available source can
answer. Grounding on those answers is 0.53 to 0.77 out of ten, which is the judge saying the
answer is supported by nothing.

To be precise about the hundred: 78 to 81 of them get an actual answer, and the remaining 19 to 22
are not refusals either. They are runs that spent every hop calling tools over an empty corpus
(`exhausted`, 10 to 13 per policy) or printed a tool call as prose (8 to 9). The agent has no way
to say "nobody can answer this": the forced final turn only runs when there are sources, so a run
that found nothing ends without a sentence rather than with a refusal.

The instructive part is the axis next to it: relevance on the same answers is 8.4 to 8.8. The
answer is about what was asked, fluent, and completely unsupported. Relevance cannot see a
hallucination; only grounding can. A dashboard built on relevance would have shown this system
as excellent on exactly the questions where it invents.

What it looks like in the log: asked in Russian how to cook carbonara without cream, the agent
returns a recipe, with `xamarin-interview-questions/README.md` in `sources`, because the hybrid
search always returns its five nearest chunks and the model answers from parametric memory while
the citation rides along. The provenance is not a lie the model tells, it is a lie the pipeline
tells for it.

The gate does not help here and was never meant to: it decides whether the corpus covers the
question, not whether anybody does. That is the next arc, and it needs a second axis (is this
our topic at all) next to the coverage axis we now have.

### What changed after this run

The silent half of the problem is already closed. The forced final turn used to run only when the
agent had sources, so the runs that found nothing ended with an empty answer instead of a refusal:
that is the 10 to 13 `exhausted` rows per policy. It now runs without sources too, with an
instruction to say plainly that the available sources do not cover the question, and those runs
come back as refusals in the language of the question.

The loud half is untouched. An answer built on irrelevant chunks has text and sources, so nothing
in the pipeline objects, and only a topic signal can. Measured on the same pools, the nearest-chunk
distance separates off-domain questions from both technical pools (AUC 0.995 where such an axis
would be used, against 0.970 for a PCA residual over corpus embeddings), which makes the second
axis a threshold on a number the retrieval already produces rather than a new model.

## What was fixed to make these numbers trustworthy

Four review rounds landed between the previous entry and this one, and most of them were about
the measurement lying rather than the system misbehaving:

- `answered=false` meant both an honest refusal and a model producing text with no evidence, and
  the refusal metrics counted every one of them as a correct refusal. Runs now record an outcome
  (`answered`, `refused`, `unsupported_answer`, `narrated_call`, `exhausted`, `error`), and the
  metrics recompute it from the stored answer, so older runs are comparable.
- Refusal was checked only when there were no sources, which made the pools differ by
  construction: the arm that hides weak chunks registered refusals, the arms that keep them did not.
- The answer text was replaced with the "no relevant documents" sentinel before being written to
  the log, which laundered every narrated tool call into a refusal.
- Deduplicating retrieval results kept the first score rather than the best, so a repeated path
  could hide a strong hit from the gate.
- The verdict was computed per tool call rather than per turn, so two searches in one turn could
  attach two contradicting notices.
- Twelve rows that looked like errors were the model spending every hop on tool calls with an
  empty corpus; they are now `exhausted`, not failures.

Two routers were tried and rejected on measurement, both in the previous entry: the cross-encoder
scores a (question, tool description) pair at zero for everything, and bi-encoder cosine puts
repo questions at 0.40-0.43 against 0.42 for an unrelated interview question. What works is
asking whether the question already carries the values the tool's required arguments need.

## Caveats

- The share of `weak` verdicts on the out-of-corpus pool is not a result. The set was filtered
  with the same cross-encoder at the same threshold, so that number is a property of the filter.
  The judged axes are not affected: different model, different signal.
- `remote_grounding` measures grounding in what the tool returned, not whether the tool was right.
  DeepWiki being wrong about langgraph is not something this lab measures.
- Refusals are not judged, so a policy that refuses more would look better by leaving the average.
  This is why the tables carry the answer rate and the pool sizes next to every axis.
- The three pools were trimmed to exactly 100 by taking the head of each set, so the questions
  dropped are the last ones authored: four post-cutoff topics and four legacy stacks left the
  off-domain pool that way. The pool still contains both flavours from the earlier batch, but the
  share of the questions most tempting to hallucinate on is lower than the description suggests.
  Next time the cut goes through a seeded sample.
- The gate's signal is not settled. Measured the day after this run, the nearest-chunk distance ranks
  as well as the cross-encoder for coverage (AUC 0.930 against 0.912 on the pool labels, gap 1.37
  against 1.60 inside the corpus) while costing nothing, and it beats a PCA residual over corpus
  embeddings as a topic signal (0.995 against 0.970 where such an axis would be used). The end-to-end
  gain above is real; whether the cross-encoder is what earns it is the next experiment.
- The grid ran on the branch state that this PR squashes into the commit "Gate the agent's
  fallback on retrieval strength" (`0c6d361` before the squash). Later work on the branch
  (bulk job cancel, requeue of stale jobs, the refusal turn above) came after the run.
- 100 questions per pool, Wilcoxon paired per pool. With four comparisons per family, the
  out-of-corpus faithfulness result survives Bonferroni comfortably; the relevance one (p=0.039)
  does not.
