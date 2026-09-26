# 2026-09-07 - Agent graph phase split, checked by replay without model calls

Our graph had one node that searched the corpus, judged what came back, and handed the model its
tool messages, all in one step. The standard's shape puts a boundary there, and moving to it means
changing the code that produces every answer this stand measures. The usual way to check such a
move is to run both versions and compare scores, which compares two samplings of a model rather
than two graphs. This entry is about the other way.

## Setup

**Rows** three runs: `arc3_refusals_with_context` (190) and `arc3_control_b` (300), both recorded by
pre-split code, and `arc3_ported_after_split` (300), recorded by the split code · **corpus**
`clean_1024` throughout · **judge** none, no model is called at any point in the comparison

- the split: `tools_node` became `retrieve` (dispatch, raw results into the state, no message for
  the model), `fallback` (drop weak context, announce, open the external toolbox) and `emit` (the
  only node that speaks to the model, and only after the verdict). The coverage verdict became an
  **edge** out of `retrieve`
- the replay: `evals/replay.py` drives a recorded row through today's graph with the model turns
  taken from the row's `transcript` and the tool answers rebuilt from `contexts`, `chunks` and
  `sources` through `metrics.spans`. Prompt templates are pinned to the versions the row recorded
- compared field by field, byte for byte: `transcript`, `sources`, `chunks`, `contexts`,
  `fallback_reason`, `outcome`, `announced`
- numbers from `replay_equality_20260906.json`, `replay_equality_control_b_20260907.json` and
  `replay_equality_ported_20260907.json`

## Result

```
arc3_refusals_with_context   190 rows, 190 replayed, 190 identical
                             branches: none 148, empty 42, announced on 1
arc3_control_b   (old code)  300 rows, 300 replayed, 300 identical
                             branches: none 144, off_topic 156
arc3_ported_after_split      300 rows, 300 replayed, 300 identical
                             branches: none 144, off_topic 156
```

`arc3_control_b` is the one that matters: it was recorded by the **pre-split** code and replayed
through the split graph, so this is the old against the new, not the new against itself. The split
changed nothing any of the seven fields can see, on 790 rows across three runs and three of the
four branches the graph has.

**The gate branch opened without the fix that was supposed to open it.** The plan was to record what
the tool returned before `_drop_weak` rewrote it, because the replay could not rebuild that call.
The fix was written. But every drop in all three runs turned out to be `off_topic`, and an
`off_topic` verdict is decided **before** the source scores are read:

```python
verdict = policy.verdict([...], gate) if corpus else None
if gate.off_topic and corpus:
    verdict = policy.FallbackReason.off_topic
```

So the erased scores were never an input on that path, and narrowing the refusal to `weak` alone
turned 169 replayable rows into 300. The recording fix stays, because `weak` still needs it, and it
was not what opened the door. The lesson is cheaper than the fix: before repairing a record for a
branch, look at what actually decides on that branch.

## What the replay found on the way

It was built as a checker and worked as a detector. Three defects in what the row records, each
invisible to the test suite because the suite compares a run with itself:

1. **the answering turn was never kept.** The graph put every tool-calling turn into `messages` and
   dropped the final one, so no replay could reproduce an answer
2. **spans counted sources instead of naming them.** A file returned on two hops was collapsed by
   `_unique_sources`, and every later slice of the flat `contexts` list slid by one
3. **a call that found nothing was rebuilt as empty content.** The live tool answers `NO_RESULTS`
   there, which does not count as context, while an empty string does. 41 rows of 190 gained a
   phantom piece

## What a review round took away from the number

The first version of this measurement claimed seven fields and full coverage. Three of those claims
did not hold, and the entry keeps them because the correction is the interesting part.

- the seventh field was `prompts`, and both sides of the comparison read it from the same row. It
  compared the record with itself and could not fail. It was replaced by `announced`, which the
  replayed side computes and the recorded side reads from the prompt versions. That one is
  falsifiable and was falsified: switch the announce flag off in the graph and 190 becomes 189,
  naming the row
- the `weak` and `off_topic` branches were called unwalked. The round replied that they are worse
  than unwalked, they are **unwalkable**, since `_drop_weak` rewrites the tool content before the
  span is written. That reply was half right and is corrected in the section above: it holds for
  `weak` and not for `off_topic`, which decides before the erased scores are read. Two claims in a
  row about this branch were too strong in opposite directions
- the replay copied the tail of `agent.run` rather than calling it, and the two copies had already
  drifted on tool names and on a language directive the run snapshot did not carry. A first run
  with an external tool or a language would have reported a difference of the graph that was a
  difference of the copies

## Reading

- equality is established on the `none`, `empty` and `off_topic` branches, the last of which
  includes 156 rows per run where the gate dropped the context. The `weak` branch has not fired in
  any run yet, so it is untested rather than untestable
- the scripted chat used to ignore the `tools` argument, so the toolbox handed to the model was in
  no comparison at all. It is recorded per hop now (`metrics.tools_offered`) and compared. For these
  three runs the hole was empty and provably so: remote tools were admitted on 22 to 25 rows each,
  but `fallback_opened` is false on all 900, so `external` never turned on
- `announced` says the notice fired, not what it said. Tool message content is not kept in the
  transcript at all, so the text of the fallback notice is observable by nothing
- a replay is worth more than a score comparison here and worth less than it first appeared. It
  proves the machine, and only on the paths the record can rebuild
- the honest form of the claim: **790 of 790 replayable rows identical on seven fields across three
  of four branches, one run of them recorded by the code the split replaced**, not "the split
  changed nothing" without qualification
- a live comparison of three 300-row arms, run beside this, finds a small divergence in hop count
  concentrated on the drop branch. It does not survive the correction over its own declared family
  of three axes, and the replay above is the reason to read it as the model's own noise rather than
  as behaviour: given the same turns, the graph builds the same messages
