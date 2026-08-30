# 2026-08-30 - Dropping what repeats across a source, and the six sections it nearly took

The coverage report has always measured how much of a source is blocks repeated verbatim across its
files. This entry turns that measurement into a cut rule, and asks what removing such blocks buys.

## Setup

**Sets** `paraphrased_v2_ru` (n=823), `paraphrased_v2` (n=820) and `veto_v1` (n=270) · **corpus**
`noboiler_1024` against `clean_1024` · **judge** none, retrieval only

The rule: a chunk whose body repeats verbatim across half or more of its source's files is dropped,
unless it is the only carrier of its section. Both halves of that sentence are load-bearing, and the
second was added before the run rather than after it.

`noboiler_1024` was dropped from the table and from the config once it lost. Its policy, so this
entry can be rebuilt from itself:

```yaml
noboiler_1024: { chunker: rooted, max_chunk_size: 1024, ceiling_on: body, drop_boilerplate: true }
```

Exact search on every arm, pool 100, paired bootstrap intervals over per-question deltas of the
reciprocal rank, eight seeds. The rule lives in the source pipeline rather than as a pass over the
table, so `scripts/cut_digest.py` reproduces the variant from the sources; the preflight confirms it.

## What the setup already answered

Counted before writing the rule, and it changed what the entry could be:

```
source                  boilerplate chunks   sole carrier of its section   dropped   of total
system-design-primer            43                     6                    37        337
the other 176 sources            0                     -                     0          -
```

One source of 177, 37 chunks of 12,102, 0.3% of the corpus. No devinterview repository qualifies at
all, and the reason is structural: each is a single file, and "repeated across half the files of a
source" cannot apply to one file.

The six spared chunks are the point. They carry the root heading of the primer's solution READMEs,
whose intro is shared across all of them, and each of the six is the gold section of a veto
question. Dropping them would have made six questions unscorable by section and **nothing would
have reported it**: the preflight checks that a question's marked file is reachable, and the file
stays. Hygiene that removes the answer is not hygiene.

## Result

```
clean_1024 -> noboiler_1024, exact search, pool 100

criterion ru, section   +0.0003  CI [0.0000, 0.0006]    5 better,  0 worse   n=823
criterion en, section   +0.0007  CI [0.0000, 0.0019]    4 better,  0 worse   n=820
veto,         section   +0.0001  CI [0.0000, 0.0004]    1 better,  0 worse   n=270
veto,         file       0.0000  CI [0.0000, 0.0000]    0 better,  0 worse   n=270

system-design-primer    boilerplate 0.1276 -> 0.0200, verdict dirty -> ok, 337 -> 300 chunks
```

Not one question of 1,913 across three sets got worse. Every lower bound sits exactly on zero.

The residue of 0.0200 is not noise: six spared chunks over 300 is exactly 0.02, so the exception is
visible in the metric it was meant to leave alone.

## The prediction that was wrong, and why it is the interesting part

The pre-registration said the criterion would move by **exactly zero**, on the argument that every
criterion question is marked to a devinterview file and no devinterview chunk is dropped. That is
mechanically wrong, and the ten questions that moved say why: **retrieval competes over the whole
corpus, not within a source**. A devinterview question's candidate list contains primer chunks, and
removing 37 navigation blocks frees ranks for it.

Which is what dropping boilerplate is for. The prediction had argued away the effect it was
supposed to be testing.

The effect is also larger on devinterview (9 questions across the two criterion sets) than on the
primer's own questions in the veto set (1 question). Navigation competes globally rather than at
home.

## Decision

**Not adopted.** The lower bounds sit on zero, so this is not a retrieval win and must not be quoted
as one. What it did do is turn one source's verdict from dirty to ok, and a source that stopped
breaching a soft gate is not a reason to move the corpus that answers questions.

The rule stays in the code behind a policy key, off by default, because the finding it produced is
worth keeping and the variant is one job away.

## Caveats

- The row count came out at 12,067 against a predicted 12,065. The rule dropped exactly the 37 it
  was counted to drop; the extra two are the local `notes` directory, which is live and had grown
  since the corpus was last indexed. No veto question is marked to the file that grew, so the
  comparison is not contaminated, but the prediction was written by someone who had documented that
  drift the same morning.
- This says nothing about generation. Both arms are read by rank alone, and a navigation block
  crowding a generator's context without changing any rank would be invisible here.
- The rule is frequency-based and source-scoped by construction, so it cannot see boilerplate that
  is common to *different* sources, or a block that repeats in fewer than half a source's files.
