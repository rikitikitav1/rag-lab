# 2026-09-23 - Schema-guided reasoning instead of the model's own tool calling

Does a rigid reasoning schema make `llama3.1:8b`, an eight billion parameter model, handle our agent
loop better than its own tool calling? Schema-Guided Reasoning promises that it lifts a weak model:
instead of letting the model call a tool, you make it fill a schema whose action field is a
discriminated union, so the choice of tool is a validated field rather than a free decision. Measured
on 1020 questions a night, it made the loop worse. On the 200 out-of-corpus questions, 0.9900 of the
arm's rows ended narrated or exhausted against 0.7450 for the control, so the arm is not kept.

**Method note: this is not a reproduction of the reference.** The differences from `sgr-agent-core`:

- Its `NextStep` carries `current_state`, `plan_remaining_steps_brief`, a `task_completed` boolean and
  a `function` union with a terminal `ReportTaskCompletion`. Its loop stops when the model picks the
  terminal variant, not on the boolean.
- Our arm keeps that stopping mechanism and that per-hop replanning. It uses four reasoning fields of
  our own (`what_is_known`, `enough_data`, `plan`, `remaining_steps`) and has no `task_completed` at
  all.
- The reference's headline of 86% on SimpleQA was taken on gpt-4.1-mini. Nobody has checked the
  thesis "a rigid schema lifts a weak model" on an 8B, and that is what this entry measures.

## What was declared before the code

The arm replaces one thing and nothing else: `ctx["chat"]` inside the existing graph. The reasoning
schema is built from the very tool schemas the control would have sent as `tools`, and goes out as
`response_format: json_schema`; the filled JSON becomes the same `ChatTurn` the control produces. Every
node, edge, gate, hop budget and reader downstream is the control's, so the comparison isolates the
calling mechanism. The arm is recorded on the same client, `openai-compat`, because it changes the shape
of the request, not the client.

**The transcript contract, declared before the code existed**, because the schema turns off the
engine's own tool calling and that blinds both readers: the arm must write its chosen tool into the same
`tool_calls` shape the control writes, and put only the answer in the text. Without it `agent_trace`
would read zero calls on the arm and `outcomes.classify` would call every row a narrated call, and both
numbers would be tautologies. A test pins it, and the ten-row smoke read `repeats.calling_hops` off a
written row before the night.

**The closing column, one, with a veto.** On the 200 out-of-corpus questions (`off_domain` 100,
`out_of_corpus` 100): the share of rows that ended `finished_by = hops_exhausted` **or**
`outcome = narrated_call`, paired by question, control minus arm, 10000 draws. The bar is the lower edge
of that interval above the **upper** edge of the floor, and the floor is the same column on a second
control pass of the same night. The veto: rows ending `unsupported_answer` must not grow above their own
floor, because an arm that turns forty narrated calls into forty confident sourceless answers would
otherwise clear the bar.

**The unflattering expectation, written before the run:** "the schema removes the narrated call but
moves the same rows into hop exhaustion, swapping one form of refusal for another and costing seconds".

**And one price declared before the code, which did not materialise.** A grammar replaces the engine's own
tool calling, so an 8B filling arguments under it might fill them worse; that was named as part of the
price rather than as a side effect, with a stop rule attached: if a single smoke row searched with empty
arguments, the night would not start. It was measured and stayed at zero on both arms, on the smoke and
across the night, so the harm below is not an argument-filling artefact.

## Setup

**Population** 1020 questions: 200 out of corpus where the waste lives, plus the 820 English questions
of `paraphrased_v2` as a guard · **arms** control `langgraph_ported` and the schema arm, both on
`llama3.1:8b` on ollama, `restate_tools` off on both, same system prompt on both, same corpus variant
`clean_1024` · **order** control, arm, control repeat, the arm paired with the first control and the
repeat serving only the floor · **runs** jobs 2887, 2888, 2889, one code fingerprint `1d8240a860bf` on
all 3060 rows · **price** 13 hours, against the 9 to 10 predicted from the control's arithmetic.

## What it bought

Nothing, and it took away.

| on the 200 out of corpus | control | arm | control repeat | control − arm | floor (control − repeat) |
|---|---|---|---|---|---|
| ended narrated or exhausted (closing) | 0.7450 | **0.9900** | 0.7400 | **−0.245 [−0.305, −0.190]** | +0.005 [−0.030, +0.035] |
| answer without support (veto) | 0.0450 | **0.1150** | 0.0500 | −0.070 [−0.125, −0.020] | −0.005 [−0.040, +0.030] |

The bar is not cleared and the sign of the comparison never comes into play: the whole interval of the
improvement sits below zero. The arm is 24.5 points worse on the column the MR was opened to move, and
the floor came out honest and tight, the two controls half a point apart, which is what the third pass
was bought for. The veto fired on its own interval.

The preregistration door recomputes the same numbers from the declaration. Its `hops_exhausted` re-derives
the hop ceiling and drops failed rows for rows older than `finished_by`; every row of this night carries
`finished_by`, so that reading changes nothing here.

**The transition table, which is the declared predicate and the actual finding.** Of the 149 control
rows that ended badly, **not one** recovered under the arm. Of the 51 rows the control handled, **49
broke**. Narrated calls went 65 to 0; exhausted rows went 84 to 198 of 200.

**The mechanism.** The grammar removes the narrated call by construction, since a tool call cannot be
written unless the model puts it inside the `answer` string. But out of corpus the topic gate empties
the retrieved context, and the arm then never picks `final.answer`: it keeps choosing to search, reaches
the hop ceiling, and the forced final node writes the answer under the arm's own grammar. The node is
shared with the control; the door inside it is not. So the arm does not decide to answer, it is dragged
to an answer at the ceiling, and on the way it spends the budget of rows the control answered on hop
two. The filled fields do not prevent it: the transcripts show `enough_data: true, remaining_steps: 1`
on a hop that calls the tool anyway, and the same query issued twice in a row. The schema makes the
model fill the fields and ties none of them to the action it picks.

**In corpus the harm is larger and the `outcome` reading hides it.** On the 820, rows that spent the
whole hop budget went **2 → 602 → 2** (control, arm, repeat), −0.732 [−0.761, −0.702] against a floor of
0.000 [−0.004, +0.004]. Read by `outcome` the arm looks fine there, because it does answer; read by
`finished_by` it spends four hops on questions the control answered in one. That is exactly why
`finished_by` is the bar of this entry and `outcome` is an exploratory reading that does not enter the
verdict. Retrieval is untouched, as it must be, since only the loop changed: `hit@5` 0.891 / 0.890 /
0.891.

**The guard's judge side, and it is the widest margin in the entry.** On the 812 in-corpus rows both
passes scored (the rest are refusals, where the axes abstain by design):

| axis | control | arm | control − arm | margin |
|---|---|---|---|---|
| faithfulness | 8.151 | 8.039 | +0.112 [−0.097, +0.321] | 0.18 |
| relevance | 9.142 | **5.550** | **+3.591 [+3.349, +3.829]** | 0.18 |
| completeness | 6.488 | **4.038** | **+2.450 [+2.254, +2.640]** | 0.18 |

Relevance is 3.6 points of ten worse, twenty times the margin the owner named; completeness thirteen
times. **Faithfulness is not decided, and the pass that would have decided it was cancelled.** Its
interval covers zero, so no harm is shown, and its upper edge of 0.321 sits above the 0.18 margin, so
non-inferiority is not established. That is not a close call: the half-width of this cell on 812 rows is
**0.209**, wider than the margin itself, so at that margin the cell could not have established
non-inferiority whatever its point value turned out to be. The margin was chosen as the smallest readable
one; on this axis and this many rows it is smaller than the noise, which is a fact about the margin, not
about the arm.

**That asymmetry is the mechanism, and it is the same one.** The median answer is 1263 characters on the
control and **247** on the arm, five times shorter, while the median number of sources is two on both, so
retrieval found the same material. The judge's reasons on the arm's worst rows say what happened to it:
"merely restates the topic without providing any substantive comparison", "focuses on methods for
combining model predictions, misses the core question", "tangential to the core". The arm reaches the
ceiling, the forced final writes under a grammar whose only field is `answer`, and what comes out is a
short grounded statement about an adjacent topic. Faithfulness survives because a short true sentence is
still supported by its sources; relevance and completeness collapse because it does not answer the
question that was asked.

**Neither reader would have described this alone, and that is why the guard carries a judge.** By
`outcome` the in-corpus arm looks untouched: 812 of 820 rows `answered`, seven refused, one exhausted.
`finished_by` saw the hops burned, the judge saw what the answer turned into, and the two together are
the harm. A guard read on outcomes alone would have reported no harm in corpus at all.

The third judge pass, over the control repeat, was cancelled once the judge's measured rate turned the
declared two hours into two hours a pass, so these three deltas are read against the margin the owner
named and against the generator floor of 0.18 taken on 14.09, not against a floor of this night.

**Repeated search, the secondary the MR hoped to move.** On the smoke the arm repeated half its calling
hops against the control's zero, and carrying the reasoning forward into the dialogue moved it 0.5 to
0.4 on forty hops, which is noise. Repetition is a property of this arm, not of how the transcript
contract was read; that hypothesis was tested with a second smoke and refused.

**Price.** Median seconds a row, out of corpus 7.8 / 17.3 / 6.9, in corpus 10.9 / 21.8 / 9.8. The arm
costs twice the control on both populations, which a twenty-row probe said before the night and which
turned the declared 9 to 10 hours into 13.

**One secondary favours the arm and it is trivial.** Refusals out of corpus rose 0.415 to 0.670,
−0.255 [−0.330, −0.180]. An arm that exhausts 198 of 200 rows refuses more by construction; it is
reported here so that nobody later quotes it as a gain.

## What this measured, and what it did not

It measured this form of SGR: the terminal variant as the stop, four reasoning fields, no
`task_completed`, on `llama3.1:8b`. The smoke pointed at a different form, one that ties the admission
of the terminal variant to a completion flag, and the reference does not do that either. It was not
built, by decision, because tuning the arm after seeing the smoke is what preregistration exists to
prevent. So the honest scope of this entry is: the union-stop form of SGR does not lift this model on
this loop, and the form with a gated stop remains unmeasured.

Two things on the night's record. The three passes carry one fingerprint on every row, but the working
tree was switched to another branch for seventy minutes while the control repeat ran, so 420 of its rows
carry `tree_differs`. The night is certified by process identity: one worker process, started before the
night and never restarted, with no module differing between the branches on its lazy import path.

The door could not tell a tree moving beside a run from code moving under it, and that was a defect of
the door. It is fixed in the same MR: the process now keeps a digest per file as it started,
`loaded_differs` names the loaded files that stopped matching it, and `one_code` is decided by that
rather than by a tree moving beside the run.

**It does not rescue this night, and the door says so rather than pretending.** These rows were written
on the older snapshot and carry no such field, so their silence is not a clean bill; the door answers
`one_code: null` and names the run in `too_old_to_tell`. The argument above stands for this night on the
rows and the process, and the door will answer it for the next one. The sharp flag would have fired here
had the worker been restarted or had it imported one of the six differing modules for the first time
inside the window, which is the point: it hides nothing real and stops answering for what the process
never ran.

## What stays in the tree, and what does not

The arm's code goes, and the enum member is retired rather than deleted, because the night's rows name it
and the log route filters on that name. What stays is what the measurement needed and nothing else: the
repeat and blind-call counters in the trace door, the canonical comparison of tool arguments, and the
fingerprint taken over a config that sits outside the tree. `llm.chat` gave back the response schema it
had been widened for, since the arm was its only caller.
