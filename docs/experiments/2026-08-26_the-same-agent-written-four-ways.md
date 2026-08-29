# 2026-08-26 - The same agent written four ways, and what the standard costs

The agent in this lab was a hand-rolled ReAct loop: our own hop counter, our own dispatch, our own
coverage gate stitched between the turns. That is fine for a lab and useless as evidence that the
author can work the way the industry works. So the loop was ported to LangGraph twice, once node by
node and once as the framework would rather have it, and all three were measured against each other
on the same questions.

The point was never that one is faster. The point is what a policy costs when you express it the way
a framework wants it expressed, and what quietly stops working when you hand a piece of your
behaviour to somebody else's abstraction.

## Setup

**Set** `paraphrased_ru` (n=100 in-corpus) plus an out-of-corpus pool (n=100), per arm · **corpus**
pre-variant era, the corpus later frozen as `baseline` · **judge** `qwen2.5:7b`, numeric 0-10

## Four arms, and why the fourth is not what it looks like

| arm | orchestrator | client | what it has |
|---|---|---|---|
| `lg_control_a` | the loop | openai-compat | everything |
| `lg_control_b` | the loop | openai-compat | everything, again: the noise ruler of the stand |
| `lg_ported` | our graph | openai-compat | everything, node by node |
| `lg_idiomatic` | `create_agent` | ChatOllama | the corpus tool plus admitted external tools from the first hop, no gate, no context drop, no notice, no nudge, no final turn |

The second control is the part that makes the rest readable. Two runs of the same code on the same
questions do not agree with themselves: on this stand they match on 99 of 100 in-corpus outcomes and
on 86 of 100 outside it (and on hop counts, only 81 of 100). Without that number a port that matches
the control on 85 out of 100 looks like a regression, when it is in fact quieter than the stand's
own noise.

The idiomatic arm is not "the standard without our policies". Tool admission and the topic axis run
before the branch point, so it inherits both. What it drops is the gate, the context drop, the
fallback notice, the nudge and the forced final turn. Reading it as a verdict on standard ReAct
would be wrong, and the decision rule says so in writing.

## The decision rule, written before the run

Primary criterion, pair control_a to ported: the share of questions where `outcome`,
`fallback_reason` and `hops` agree must not fall more than 5 points below the same share in the pair
control_a to control_b, **computed separately per pool**. The per-pool part came from the auditor:
on a mix of 200 the in-corpus agreement (0.99) drowns the external one (0.86) and the threshold
stops meaning anything.

Guard: the ported median latency on the corpus pool must stay within 1.15x of control_a. Everything
else, judge axes with bootstrap intervals included, is descriptive.

## Result: the port against the loop

The port matches the loop everywhere the criterion looks, and the second control is what makes that
sentence mean anything:

| pool | key | noise a→b | ported | verdict |
|---|---|---|---|---|
| in_corpus | outcome | 0.990 | **1.000** | pass |
| in_corpus | fallback_reason | 0.980 | 0.990 | pass |
| in_corpus | hops | 0.960 | 0.950 | pass |
| out_of_corpus | outcome | 0.860 | 0.850 | pass |
| out_of_corpus | fallback_reason | 1.000 | 1.000 | pass |
| out_of_corpus | hops | 0.810 | 0.830 | pass |

Six of six, and none of the bootstrap intervals on the difference of the two rates reaches the
threshold. Inside the corpus the graph agreed with the control more often than the control agreed
with itself. Outside it the disagreements are symmetric: six questions moved from an answer to a
narrated call, five moved back, two from a refusal to an answer. That is the model shaking, not the
graph drifting.

Latency guard on the corpus pool: 7.90s median against 8.25s, or 0.96x where 1.15x was allowed. The
second control sits at 1.06x, so the graph is inside the spread of two runs of one code.

Judge axes, paired over the questions every arm judged: 0.00 difference inside the corpus ([-0.36,
+0.37]), +0.40 outside ([-0.49, +1.29]). The noise pair's own interval outside is a point and a half
wide, so this grid cannot resolve anything smaller than about 0.9 of a point there. That belongs
next to the numbers, not instead of them.

## What the arm without our policies actually does

Its outcomes match the control within noise (0.97 inside, 0.85 outside), which is the least
interesting thing about it. The hop counts do not: 0.41 agreement outside against 0.81 noise, and an
average of 2.12 hops against 3.10. It never has to wait for a gate to open a toolbox.

The number that says what the policies buy is not a judge axis at all. It is how often each arm
handed the question to an external tool on the pool where the corpus has no answer:

| arm | handed off | answered | of those, from the corpus | grounding of those |
|---|---|---|---|---|
| control A | 50 | 84 | 42 | 2.69 |
| control B | 49 | 87 | 45 | 2.31 |
| ported | 51 | 84 | 44 | 3.14 |
| idiomatic | **32** | 93 | **61** | 2.48 |

The bare arm answers more often and refuses never, and it does that by answering sixty-one questions
out of a corpus that does not contain the answer. Those answers score 2.48 on grounding over the 61
that were answered, 2.25 over all 67 judged rows of that group, against 7.13 for the ones it did
hand off. The policy set it lacks (the coverage gate, the context drop, the fallback notice, the
nudge and the final turn) is together worth eighteen handoffs per hundred external questions, and it
costs nothing measurable in time: that arm's latency is 1.01x.

Two caveats belong in the same breath. This arm also swaps the model client, so on its own it cannot
separate the policies from the transport; what closes that gap is the middleware arm further down,
which runs the same policies on the same client and hands off 51 times. And 1.01x is per question,
not per hop: at 2.12 hops against 3.10 it does less and still takes the same time, which this grid
does not explain and did not try to.

One more number to keep honest: inside the corpus the bare arm scores +0.44 on relevance ([+0.03,
+0.90]), the only interval in the whole grid that does not cross zero. Relevance is the axis this
lab has already shown to be blind to hallucination, and leaving it out because it is inconvenient
would be as wrong as leaving out the handoff count.

## Reading latency on the external pool

The median of that pool is a trap. A question answered locally takes about 7 seconds, one that goes
to DeepWiki takes about 25, and the pool median sits exactly on the boundary between the two humps,
so a single extra handoff moves it by five seconds. Read straight, the ported arm looks 1.51x
slower. Split by whether the arm went outside, all four arms are within a second of each other (25.3
against 25.2 when both went out, 7.0 against 6.6 when neither did). The guard was placed on the
corpus pool for exactly this reason.

## Where the time actually goes

The grid also carries a per-stage breakdown now, because "eight seconds per question" hides which
part of the run spent them. On an external question, averaged:

| stage | loop | hooks |
|---|---|---|
| DeepWiki | 15.1s | 15.4s |
| tool admission | 0.36s | 0.35s |
| topic axis | 0.22s | 0.20s |
| corpus search | 0.24s | 0.23s |
| model | 8.4s, wall clock around our client | 4.9s, reported by the server (0.6 reading the prompt, 4.3 writing) |

The model row is the one to distrust: the two arms measure it with different instruments. Ours wraps
the HTTP call, the server's number excludes the network and the load, so the rows are not comparable
to each other, only each to itself. Everything above the model row is measured the same way in both.

Two thirds of an external question is waiting for somebody else's service. The admission check,
which was first on the list of things to remove for speed, costs about a third of a second, roughly
three times less than assumed. Measuring before optimising saved that particular effort.

## What the standard costs

Known in advance and confirmed:

- error kinds do not survive the tool contract. Our dispatch classifies a failure as
  `timeout|connect|auth|client|server|tool|empty` and the policy differs per kind. The standard tool
  contract has room for a success and an error, nothing else. The ported arm keeps our dispatch node
  and keeps the classification; the idiomatic arm cannot.
- the idiomatic arm has no final turn without tools. When the model wants a fifth hop, the run does
  not answer, it hits the recursion guard and ends with empty text, which the judge skips. On the
  previous grid of the same shape that is 1 question in 100 in-corpus and 10 in 100 outside it, so
  its judge averages are lifted by selection rather than by quality. In this grid the effect turned
  out small: the guard ended exactly one run of 200, and it was question 10593, the same one that
  has been the record holder for prompt size in every grid this lab has run. The mechanism is real,
  the exposure this time was not.

## The arm that expresses our policies as the framework's own hooks

**This arm was removed on 2026-08-27.** Keeping a second implementation of policies that the graph
already carries cost a branch in every change and bought nothing the numbers below could resolve.
Its 222 runs stay queryable in the log under `orchestrator=langgraph_middleware`; runs cannot ask
for it.


It ran later, against the same controls, on the same questions: five checks of six pass, latency
1.04x, handoffs 51 against 50 and 49 for the controls, zero tool errors. Inside the corpus it is
indistinguishable from the loop.

Outside it, two numbers stand out, and both come with the same caveat. Agreement on hop counts is
0.65 against a noise floor of 0.81, and the bootstrap interval on that difference sits entirely
below the threshold, so it is not a point estimate wobbling. And its grounding is **higher** than
the control, 5.77 against 4.74 and 4.63, paired delta +1.02 with an interval of [+0.17, +1.86]: the
only interval in the whole grid that does not cross zero.

The caveat is the same for both: this arm ran nine hours after the controls and twenty three minutes
after DeepWiki spent half an hour answering 503, and in that hour its answers were on average 294
characters shorter. So neither number is attributable yet. "The hooks are worse at counting steps"
and "the hooks are better at grounding" are equally unearned readings of the same unseparated
effect, and the honest record says so.

So the arms were run again, interleaved: twenty questions where both had gone outside, the loop and
the hooks alternating in blocks (the loop was still in the tree at that point), one time window, one
state of the external service. In that window DeepWiki's answers to the same questions differed by
28 characters on average, against 294 on the night of the grid, which is the clearest single sign
that the night's numbers were about the service and not about the harness. That run is more sober
than the night one:

```
hop count difference     0.00   interval [-0.40, +0.40]
hop agreement            0.70   interval [ 0.50,  0.90]
grounding difference    +0.85   interval [-0.85, +2.80], p=0.64
```

The average hop counts came out identical, so there is no systematic shift. The grounding advantage
does not survive a single time window: the interval covers zero comfortably. And the agreement rate
cannot be resolved at twenty pairs at all, its interval spans both the control's level and the
night's number.

Recorded as it stands: **not reproduced, means identical, resolving it needs a full pool in one
window.** Which is fine, because the arm this grid actually decides on is the graph, and it needs no
such caveat.

The rule that came out of this is worth more than the number it failed to produce: **an arm compared
on the external pool runs in the same window as its control, or the control is repeated next to
it.** A remote service is part of the setup, and a setup that drifts between arms is not a control.

Expressing our policies as the framework's own hooks is the version an outside reader recognises
fastest, and it is where the port stopped being mechanical. Three facts had to be right before it
worked at all: a jump between nodes only exists if the hook declares it, `before_*` hooks run in
list order while `after_*` run in reverse, and the tool set for one call is changed by overriding
the request rather than by assignment.

## The context window, and a number that was never real

The stand ran on Ollama's default context of 4096 tokens for every grid before this one. The server
does not report an over-long prompt: it drops whole messages first and then counts what is left, so
a prompt larger than the window never appears as a number above the window. The window is now a
parameter (8192), the value the server reports goes into every snapshot, and a hop that needs fewer
tokens than the one before it is recorded as trimmed.

An early reading of this said 261 rows of the old grids ran over the window, with a maximum of 13321
tokens. That was the sum of prompt tokens across hops, not the size of any prompt. The real per-hop
maximum never crossed 4075. What actually happened at the old window: exactly one row per run
touched the ceiling, and it was the same question in all seven large runs, so the trimming hit every
arm the same way and moved no paired comparison.

Latency turned out to be comparable after all. The first fifteen rows of control A looked 30% slower
than the old grids, which read like the price of the larger window, and the auditor flagged it as
such. On the full hundred the median is 8.3s against 8.2s at the old window: the first rows were
warm-up. The window is not free in VRAM, but on this hardware it is free in time.

## Which code the grid actually ran

`33c9a5b`, and not the head of the branch: the worker started an hour before the later commits
landed, and a worker holds its code in memory. That is worth recording rather than glossing, because
the arms were fixed while the grid ran and the fixes did not reach it. What the grid's code lacks:
the failure flag in the loop and the graph, the graph's own error path, a timeout on the standard
client, and the dispatch-level refusal of a withheld tool. All four arms ran the same code, so
comparing them to each other stays honest; the middleware arm, which runs later, will not.

The commit itself is a reconstruction from container start times, not a reading from the rows: the
four grid runs predate the field that records our commit, so their `code_version` is null. The later
runs do carry it, and the middleware arm carries two, one per pool: its corpus half ran on `5232911`
and its external half on `cc0dc44`, thirty minutes and a worker restart apart. The difference
between those commits is ten lines in the MCP error classifier and touches nothing on the routing
path, but the honest record names it rather than rounding it to one commit. All three are tagged
(`run/lg-grid`, `run/lg-middleware-corpus`, `run/lg-middleware-external`), because the branch was
rebuilt afterwards and none of those hashes survive in it.

The auditor's argument for why the controls remain usable is stronger than "the fixes are small":
between that commit and the head, the loop changes only on the `RuntimeError` path, and in the 400
control rows that path is never taken. Not one row is an error or an exhaustion.

## The noise between two identical runs is not only the model

One question in the external pool is a good miniature of the whole problem. Control A answered it
through DeepWiki; control B asked the same tool, got a timeout, and refused. Same code, same
question, same settings, different answer, and the cause is a remote service having a bad second.
That row sits inside the 14-in-100 disagreement rate that the second control exists to measure, and
without that second control it would have been read as evidence about an orchestrator.

## What the review found that the tests could not

Six review passes read the branch while the grid ran, each with its own lens, and a seventh verified
every finding against the code. The findings that mattered were not about the graph at all:

- the failure flag lived only inside `agent.run`. The reporting layer recomputed the outcome from
  the hop count, so a recursion guard that fired read as a run that politely spent its hops. The fix
  from the day before was real and had never reached the layer that produces the numbers.
- a tool error written by the standard `ToolNode` starts with a capital letter and slid past our
  prefix check, so a failed search read as an empty corpus and opened the external toolbox where the
  loop stays quiet.
- the preflight script passed when the stand was missing: a swallowed return code compared an empty
  window against an empty config.

- under `corpus_first` the middleware arm executed a remote tool the gate had not opened. The
  standard tool node is built once from the whole list, so narrowing what the model is offered never
  stopped a call the model invented, and this model invents them often enough that a whole nudge
  exists for it. The loop refuses the same call by name. Three reviewers found this independently,
  and the mechanism was confirmed in the installed framework rather than argued from the diff.

And one that no reviewer could have found, because reviewers were forbidden to run anything: the
agent tests scored the topic axis against a live corpus. "How do I cook carbonara" got 0.57 against
a threshold of 0.50, was ruled off-domain, its context dropped, and the test failed. In CI, with no
stand, the same test passed. The suite answered differently depending on whether the machine it ran
on had a corpus. It now silences the axis by default, and as a side effect runs four times faster.

## Decision

The port becomes the implementation: `langgraph_ported` is the default and the hand-rolled loop is
retired, its runs still readable in the log as `orchestrator=agent`. The bare `create_agent` arm
stays as the measurable answer to "what does the idiomatic version cost". Two rules come out of this
grid and outlive it: an arm compared on the external pool runs in the same window as its control,
and the second control is not optional, because without it a port that matches on 85 of 100 reads as
a regression.

## Caveats

- n=100 per pool per arm, and the noise pair itself bounds what this grid can resolve: about 0.9 of
  a judge point on the external pool
- the idiomatic arm also swaps the model client, so on its own it cannot separate the policies from
  the transport
- the four grid runs predate the field recording our commit, so `33c9a5b` is reconstructed from
  container start times rather than read from the rows
- the external service is part of the setup and drifts: the middleware arm's night numbers were
  never separated from a DeepWiki that answered 294 characters shorter that hour
