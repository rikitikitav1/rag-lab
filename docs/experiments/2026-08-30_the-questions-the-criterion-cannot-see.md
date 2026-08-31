# 2026-08-30 - The questions the criterion cannot see

The criterion set is made entirely from one family of sources. This entry builds a second set over
the four families it cannot see, measures the hygiene change on it, and says what such a set is
allowed to decide.

## Setup

**Set** `veto_v1` (n=270) · **corpus** `clean_1024` against `baseline` · **judge** none, retrieval only

- questions are generated from section headings by `gemma2:9b` under prompt `question.from_heading`
  v1, seed `hygiene_v1`, quotas of 80 / 80 / 80 / 30 over `cheatsheets`, `redis-doc/docs`, `notes`
  and `system-design-primer`
- exact search on both arms (`enable_indexscan = off`), pool 100, no reranking
- paired bootstrap intervals over per-question deltas of the reciprocal rank
- halves are a pure function of the question id

## Why the set exists

`paraphrased_v2_ru` and `paraphrased_v2` are built from devinterview: 173 repositories, each one
README whose headings are already questions. The corpus holds four more families with a different
shape, and until this entry they had no questions at all:

| family | what it is | files | chunks in `clean_1024` |
|---|---|---|---|
| cheatsheets | small files with a frontmatter title | 321 | 1269 |
| redis-doc/docs | product documentation | 93 | 1913 |
| notes | the owner's own notes, in Russian | 36 | 176 |
| system-design-primer | nine very large files | 9 | 337 |

That is 3,695 chunks of 12,102, **31% of the corpus the veto now asks about**. Another 840 chunks
(`redis-doc/commands`, 370 files) are outside the criterion too and are left out on purpose: their
headings are command names, not questions. Counting them, 37% of the corpus has no criterion
question, and the veto reaches four fifths of it. The exposure is concrete rather than theoretical: the chunk ceiling is chosen on devinterview and applies to
everything, so a ceiling that helps long uniform READMEs and shreds small cheatsheets would pass
the criterion without a mark.

## How the questions are made

A heading becomes a question through the generator rather than being used as one:

```
notes/databases/redis/use-cases.md | Rate limiting
   -> Как можно использовать Redis для ограничения частоты запросов?
```

Four rules, each closing one way the set could lie:

- **generated, not copied.** A raw heading sits verbatim in the chunk, so the keyword leg would find
  it every time and the set would measure string equality rather than retrieval.
- **only headings that name exactly one file within their family**, and no shorter than 12
  characters. `Usage` and `Reference` sit in a hundred cheatsheets and point at nothing.
- **only from files every compared variant holds.** `baseline` is frozen in time while `notes` is a
  live directory, so without this the delta on that family would be part cut and part drift.
- **the heading is stored as the matcher will look for it**, numeric prefix stripped, so gold and
  section agree by construction rather than by luck.

`redis-doc/commands` is excluded: files like `get.md` have no heading that is a question, and a
question made from a file stem is a label.

## Result

```
baseline -> clean_1024, exact search, pool 100

file     all  +0.0870  CI [+0.0609, +0.1170]   71 better, 12 worse, 187 unchanged   n=270
         A    +0.0974  CI [+0.0564, +0.1414]   n=124
         B    +0.0783  CI [+0.0392, +0.1198]   n=146
section  all  +0.0265  CI [-0.0053, +0.0675]   n=59
```

Corpus drift is printed beside the delta, because a delta over questions only means something
where both sides hold the file:

```
cheatsheets only in baseline: 15, only in clean_1024: 0, in both: 321
notes       only in baseline:  2, only in clean_1024: 2, in both:  34
redis-doc   only in baseline:  6, only in clean_1024: 0, in both: 463
```

The redis-doc line counts the whole source, `commands` included, which is why 463 stands against
the 93 files of the `docs` subtree in the table above.

devinterview does not drift and so is not printed.

## Two floors, and the data chose which

On `baseline` only 59 of 270 questions are scorable by section: `cheatsheets` and `redis-doc` have
**zero** chunks carrying a section there, because the old splitter did not record one. So against
the frozen variant the set reads **by file**, and by section only among hygienic variants. That was
planned as a two-floor design and the measurement is what says which floor applies where.

## Decision

The hygiene change measured on devinterview moves the four families it cannot see in the same
direction, +0.0870 by file with both halves agreeing. This is the first confirmation of the
`clean_1024` decision **outside the population it was taken on**; before this entry it was an
assumption.

The set's standing role is fixed here, and it is narrow on purpose:

- **it vetoes, it does not choose.** A winner picked on devinterview must fail to lose here, and the
  test is that the paired interval does not lie entirely below zero. It never selects a winner.
- **its paired deltas are readable, its absolute numbers are a floor.** Each question carries one
  marked file and some could be answered honestly from another family, so `hit@5` of 0.907 is a
  lower bound, not retrieval quality. The bias is constant between arms and cancels in the pair.
- **it resolves about 0.03 MRR at n=270.** Anything smaller it cannot see, and per family, at n=80,
  less still, which is why family numbers are descriptive and the veto is read on the whole set.

## Caveats

- The questions are synthetic and one of them is visibly poor ("Is this document still being
  actively worked on?", made from a heading about a document's status). Such noise is symmetric: it
  hurts both arms equally and cancels in the pair, while widening the interval, so it makes the veto
  harder to fire rather than easier. That is the safe direction for a veto.
- The set describes the sources and the variants that existed when it was built. Adding a source, a
  family, a ceiling or a splitter makes it stale, and a stale veto does not fail: it goes green. The
  triggers for rebuilding it are kept with the set (internal note `notes/reference/veto_set.md`),
  and a rebuild takes a new name rather than editing the old set in place.
- Nothing here touches generation. Both arms are read by rank alone.
