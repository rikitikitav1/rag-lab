# 2026-09-07 - What moving onto the standard was worth, and what it cost

The arc set out to put the industry's judged axes into this stand and find out how far our own
judge sits from them. It did that. The larger part of what it produced is not the standard's
numbers but what measuring against them forced us to build, and four claims of our own that did not
survive being checked.

## Setup

**Guests, not a second arm.** The standard's axes ride our own judging pass: same client, same seed,
same row (`evals/guest_axes.py`). Both scores land on one row, so a correlation between them is a
join rather than a comparison of two samplings, and the reload artefact between two passes takes no
part in it · **guest** `ragas 0.4.3`, judged by `qwen2.5:7b` · **ours** the same model, our prompts
· runs `arc3_agent_baseline`, `arc3_ported_after_split`, `arc3_interview_independent`,
`arc3_single_shot_ab`, `arc3_weak_gate`

## What the guests are, and what they are not

Our three axes are one judge call each: a whole number 0-10 with a reason. Retrieval is not judged
at all here, it is counted against labels (`hit@k`, `MRR`, `file_precision`, the same pair over
sections).

The guests differ in shape rather than in scale:

```
ragas_faithfulness         2 calls a row     question, answer, contexts
ragas_context_precision    a call per context question, contexts, reference answer
ragas_context_recall       1 call            question, contexts, reference answer
```

Their faithfulness does not score the answer whole: it breaks it into statements and checks each
against the context, and the score is the share supported. The two context axes are the part we
cannot compute from labels at all, because they judge what the retrieved chunks **say** against a
reference answer rather than whether the file name matched a label. That is also why they abstain
wherever a question carries no reference.

They cost between six and eight times our three a row: 51.4 seconds of model time against 6.59 on
one run, 6.62 on another (`guest_multiplier_*`). That is what settled their role.

## Result: what the standard told us about our judge

**The ruler moves.** Guest faithfulness has a noise floor of its own at **0.0719** over 321 pairs
with nothing changed, 70% of pairs identical, and 19 calls of 886 did not parse at all. Whatever
else follows, the standard is not ground truth.

**Agreement is weak, and on independent questions unreadable.** Pooled over a hundred questions
scored twice, rho is **0.4386 [0.336, 0.536]**. On 114 rows of questions that had never been run
before, rho is **0.161 [-0.023, 0.335]** and may not be read: the guest's spread there is 0.091
against 0.259 to 0.302 on the older pools, so the correlation is restricted by range rather than
broken by disagreement. All four declared controls passed and none of them asked whether there was
anything to correlate.

**On our refusals the two do not rank alike at all.** Means are close (0.5973 against 0.6527 over
55 rows) and Spearman is 0.1947 with p 0.154. That is what took refusals out of the population, and
later out of the axes entirely.

**Our own prompt loses a point on Russian.** Same context, same question, same claims, the answer
restated in two languages: English 8.255 against Russian 7.291, paired **0.964 [0.645, 1.282]** over
110 rows. It reproduces the size measured on 31.08 from the other side.

## What the arc cost, in its own terms

```
two days, forty commits, three review rounds and eighteen rounds with the auditor
```

**Four claims of ours were withdrawn after being written**, each by a check outside the measurement:

- a verdict on two judge prompt versions, measured over 138 rows where 91 were declared. Put back,
  zero stopped being excluded
- a reading of the language probe, taken in a regime where the judge scored restatements of a
  context line rather than answers, and gave a fifth of them zero
- a noise floor taken across a model reload and compared against an effect measured inside one
- a prediction that a debt would fall by eighty, computed with a query written by hand where the
  debt is counted by another predicate. It fell by one

The common shape is one thing: **two numbers computed by different predicates over different rows,
compared as if they were the same measurement**.

## What it bought that is not RAGAS

- the tool step split into `retrieve`, `fallback` and `emit`, with the coverage verdict on an edge,
  which is the standard's shape
- a replay that drives a recorded row through today's graph with no model call at all: **890 of 890
  replayable rows identical on nine fields across all four branches**, three hundred of them
  recorded by the code the split replaced
- one door for queueing any job type, with the options of each type checked when it is queued and
  again when the worker takes it
- and a fact about our own judge that nobody was looking for: with the same seed, the same width and
  byte-identical input it is **deterministic inside one model residency (144 of 144, reasons
  included) and moves 14% of its scores and 58% of its reason texts across a reload**

## Reading

- the guests are a **calibration**, not an axis. They do not ride every run, they draw a subsample,
  and they enter neither the composite nor our axes' Holm family: a ruler used to check a ruler is
  not a fourth measure of quality
- **was it worth integrating: yes; was it worth keeping in the loop: no**, and those are different
  questions. Before the arc "move onto the standard" sounded like picking up a correct ruler. It is
  now known to be a noisy one, weakly ranked with ours, and expensive
- the one thing that would still pay: `context_precision` and `context_recall` against our own
  retrieval axes. They are the part labels cannot express, and they have never been compared
- **the honest form of the headline claim**: our judge and the standard's agree weakly on a hundred
  questions measured twice, and on independent questions the comparison could not be made at all
  for want of a corpus whose questions are not their own section headings
