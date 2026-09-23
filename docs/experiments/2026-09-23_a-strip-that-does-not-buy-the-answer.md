# 2026-09-23 - Stripping the noise inside the chunks the grader kept

The grader of an earlier entry drops whole chunks it calls irrelevant. Corrective RAG (Yan et al., 2024)
goes one step further on the chunks it keeps: it cuts each one into strips, scores every strip against
the question, and joins the kept strips back in order. This entry records what that buys on our corpus,
measured on the answers, and why the code is not kept.

**This is not a reproduction of the paper.** The paper scores strips with a fine-tuned T5-large and a
threshold of −0.5. Ours are scored by the same prompted `llama3.1:8b` that grades the chunks, prompt
`grade.chunk` v1, dropped only on a plain "no". The strip follows the paper's definition, «a few
sentences according to the total length»: on this corpus a line is a sentence, so a strip is three
lines, a fenced code block is one strip with its fences, and the heading lines of a chunk (its source
tag, the file and section headings) are never a strip and never asked. Each strip is judged bare, as in
the paper, so a "no" on three lines of a list is partly a property of the form.

## What was declared before the code

The promise was written through the stand's preregistration door (`mr5_strip`) before the first row of
the night, and closed through the same door.

**Closing.** Faithfulness from our judge, paired by question, 10000 draws. The arm clears when the lower
edge of its gain sits above zero and above the upper edge of the control against a second pass of
itself in the same night. A band through zero closes the entry with "stripping does not pay".

**Control and arm.** The control is the chunk grader of the earlier entry; the arm is the same grader
plus the strip. The delta then belongs to the strip alone, not to the chunk filter measured before.

**Guards.** Relevance and completeness must not fall and refusals must not rise, each read against the
same night's floor. **Veto**, read before any answer was generated: on the grader bench, the share of
characters stripped out of gold chunks must not exceed the share stripped out of chunks from unrelated
files by more than 0.05. By our labels a whole section is gold, so every character cut from it is a
cost; the strip can only act as intended on the other chunks.

**The odds, written down before the night:** the point delta within ±0.10, and at most one chance in
four that its band clears zero on 200 pairs. **The unflattering expectation:** the strip cuts gold about
as much as it cuts strangers, and the delta sits inside the floor.

## Setup

**Population** 200 English questions of `paraphrased_v2`, drawn with `random.Random(23)` from the 620 on
which nothing had been chosen in the earlier entry · corpus `clean_1024`, the direct path, the top five by
fusion, reranker off · **runs** control job 2903, arm 2904, control repeat 2905, judged by 2906-2908
(Qwen2.5-7B-AWQ on vLLM), all three on one code fingerprint and one worker process · **veto** the grader
bench over the frozen candidate pool of the earlier entry on the same 200, job 2901, 4575 verdicts.

A ten-question smoke ran first with its outcomes written beforehand. All six held, and the reading of its
rows found one defect: the served chunk starts with a `[source]` line, the strip did not count it as a
heading, and the arm was judging "tag, file heading, section heading" as a strip of its own and dropping
it on 18 chunks of 28. It was fixed and the arm was re-smoked before the night: on 27 chunks shared with
the bench the arm now cut the same strips, and its strip verdicts agreed with the bench's on 98 of 98.

## What it bought

**The veto stayed quiet.** The strip removed 0.25 of the characters of gold chunks (140 rows) and 0.32 of
strangers (91 rows). After the strip the grader still kept a piece of the gold section on 0.952 of rows
(146; 0.980 before it), and dropped 0.478 of strangers (133; 0.447 before). Read against the earlier
entry's two-sided bar, the grader with the strip takes neither side; that is an explanation, not a closing.
The bench took 45 minutes against the 30 promised: 0.41 seconds a verdict on a ten-row probe, 0.59 on the
two hundred.

| faithfulness, 190 pairs | control | arm | arm − control | floor, control − repeat (193) |
|---|---|---|---|---|
| judge, 0 to 10 | 8.858 | 8.784 | **−0.074 [−0.332, +0.174]** | −0.036 [−0.238, +0.155] |

The bar was a lower edge above 0.155. The lower edge is −0.33 and the point is below zero, so **the strip
does not pay**, and the closing came out as the odds said it would.

| guard, 190 or 200 pairs | arm against control | its bar | state |
|---|---|---|---|
| relevance | −0.068 [−0.326, +0.195] | 0.104 | undecided |
| completeness | **−0.226 [−0.479, +0.026]** | 0.192 | undecided |
| refusals | +0.015 [−0.005, +0.040], 9 against 6 | 0.015 | undecided |

No guard is decided, and completeness leans the way the mechanism predicts: a quarter less context, a
quarter of it from the gold section, and answers that cover a little less of the question. It is the one
cost the night can name, and only carefully: by point it sits above its bar, its lower edge is just under
zero, and its raw p of 0.043 is not significant under Holm over the six tests of the comparison. On the 181
judged rows where the two chunk passes kept the same set it reads −0.276 [−0.514, −0.044]; that reading
was chosen after the data and explains, it does not close.

**The price.** Context went from 3512 to 2639 characters a question, and the answer took 16.4 seconds
instead of 8.9, twelve more model calls on every question.

**One caveat on the pairing.** The arm repeats the chunk grader with its own verdicts before stripping.
The chunk pass repeats itself on 189 of 200 questions against the arm and on 193 against the control's
own repeat, a little under the 0.95 the promise named; that is the chunk grader's own noise, not
something the strip did. On the 181 judged rows where both passes kept the same set, faithfulness reads
−0.099 [−0.365, +0.144], the same verdict.

**A floor for the strip grader, first reading.** On identical inputs minutes apart, the arm and the bench
gave the same verdict on 97 of 98 strips in the first smoke (keys shifted by the source-tag defect) and on
98 of 98 in the re-smoke, and the same chunk verdict on 46 of 50 chunks. At temperature zero the strip
grader is about as stable as the chunk grader.

## What stays in the tree, and what does not

The strip goes: the flag on the run, the path that cut and rejoined chunks, and the bench form that
measured it. What stays is what the measurement needed and what the next preregistration will need: the
preregistration door now reads a recorded grader measurement and a judge score as well as outcomes, a
promise can name its question draw, a veto is read inside one arm before the expensive part, and the
grader bench takes named question ids. The columns that read the characters a strip removed stay too,
because this entry's promise names one of them and the door must still be able to read it back. The door
recorded this closing with the run names under a doubled key (`runs.runs`); the key is fixed in the code
and the recorded row is left as it was written.
