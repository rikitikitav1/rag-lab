# 2026-09-06 - Topic axis switched off: the agent refuses far less

This run was meant to build a set the guest axes had never been shown: rows where our agent
refuses **while holding a non-empty context**. It failed at that, three rows out of two hundred
against a declared floor of thirty, and in failing it priced something the stand had never put a
number on. Switching off one threshold does not change how well the agent refuses. It changes
whether the agent refuses at all.

## Setup

**Rows** the same 300-question CRAG-2 shape as the baseline, run twice: a ten-row smoke and 190
rows · **corpus** `clean_1024`, identical to the baseline · **judge** ours, `qwen2.5:7b`, the guests
were run separately afterwards

One knob differs from the baseline and nothing else: `topic_threshold: 0`, which switches off the
topic axis. The prediction was that an agent no longer stopped by topic would search, come back
with irrelevant chunks, and refuse on them, producing refusals that carry context.

- runs `arc3_refusals_smoke` (10) and `arc3_refusals_with_context` (190)
- baseline for comparison `arc3_agent_baseline`, same questions, topic axis on
- outcome counts read through `run_metrics`; this run was also the first to record `transcript`
  and `spans` on every row, which the replay in the next entry needs

## What was declared before the run

Four expectations, written before the queue was filled:

1. at least 30 rows refuse with a non-empty context, else the run is read as a search for where the
   context goes, not as a set for the sub-phase
2. the share of refusals falls, because the toolbox opens more often
3. `transcript` and `spans` are filled on every row
4. false refusals are not measured here, the run holds no corpus questions

## Result

Expectation 1 failed, and by a wide margin: **3 rows of 200** refused with a context. The context
did not go anywhere; 177 rows of 200 carry one. What went away was the refusals.

```
                        baseline, axis on        this run, axis off
off_domain              refuses 65/100           refuses 12/100
out_of_corpus           refuses 30/100           refuses  3/100
overall refused         95 of 300                15 of 200
answered_ungrounded     13 of 300                114 of 200
```

Expectation 2 held, and the reason given for it was wrong. The prediction said the fall would come
from the toolbox opening and the agent answering from outside. **One row of two hundred** went
outside. The real mechanism is the other one: with the topic gate open the model finds something to
say in chunks that do not answer the question, and returns an ungrounded answer instead of a
refusal. That is what `answered_ungrounded` going from 13 to 114 means.

Expectation 3 held on every row. Expectation 4 held by construction.

## What is actually new

The topic axis was introduced as a politeness filter, a way to stop the agent wandering off the
corpus. This run says it is the **main producer of our refusals**. Take it away and the refusal
rate on off-domain questions falls from 0.65 to 0.12, not because the agent found the answer
elsewhere, but because it agreed to answer from whatever the corpus returned.

That is the price of the knob, stated as a number for the first time, and it is much larger than
what was written down when the knob was added.

## Reading

- the axis is not a filter of manners. It is what makes the difference between a refusal and an
  ungrounded answer on a question the corpus cannot serve
- a set of refusals-with-context cannot be built with this knob, because the knob removes the
  refusal, not the emptiness of the context. It needs corpus questions whose answer is absent from
  the chunks that were retrieved, which is a different run
- `answered_ungrounded` at 114 of 200 is the failure mode to watch when anyone proposes loosening
  this threshold to raise the answer rate. The answer rate does rise. The answers are not grounded
- the three refusals that did carry a context turned out to be worth more than the two hundred
  that did not: they are where our judge and the standard's disagree by construction, and that is
  written up in the entry beside this one
