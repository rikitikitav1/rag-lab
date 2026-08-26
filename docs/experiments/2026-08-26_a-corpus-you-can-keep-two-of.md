# 2026-08-26 - A corpus you can keep two of, and the instrument that measures it

Every threshold in this lab was measured against one corpus: retrieval distance 0.55, the coverage
gate at 0.39, the topic axis at 0.50. Re-chunking moves the embeddings and every one of those
numbers with them, so the plan called re-indexing a border: cross it once, recalibrate everything,
and accept that runs from before and after cannot be compared.

This entry is about deciding not to cross it. The corpus now has a `variant` column. The old corpus
stays as `baseline`, frozen; a new chunking arrives as a neighbour; both live in the same table and
answer the same questions. Chunking stops being a border and becomes a swept parameter.

Nothing was re-chunked yet. This is the instrument, and the entry is mostly about what the
instrument got wrong three times before it was trusted.

## What a variant is

A row in `data_chunks` names the recipe that produced it. `baseline` is the corpus as it stood on
26 August: 13 068 chunks, 1 023 files, 177 sources, cut by `chunk_markdown` at 1 024 characters,
`chunker: legacy` recorded in its policy. It was frozen where it lay rather than re-indexed by the
new code with hygiene switched off, because "the old splitter with old settings" and "the new
splitter with old settings" are not the same thing, and only the first is what every previous run
actually measured.

Two columns were backfilled onto it without re-embedding. `content_hash` for deduplication, and
`section` for the heading path, recovered with a window function over `chunk_index`: 7 918 of the
8 091 interview chunks got one, and the 173 that did not are exactly the boilerplate chunk 0 of each
repository. `redis-doc` and `cheatsheets` have no `##` structure to recover, so their section is
unknown and counts as a miss at section level, which is honest, because that is what it was.

Every reader of the corpus now has to say which variant it reads - `hybrid_search`,
`nearest_distance`, `corpus_fingerprint`, `is_empty`, `list_categories`, `cleanup` - as a
keyword-only argument with no default, and a test asserts that shape so the next reader cannot
forget. The topic axis mattered most: had it kept reading the whole table, it would have measured
distance to somebody else's corpus without a word.

## The instrument, and the four ways it lied

Retrieval quality is measured by a script that takes the question embeddings already stored in the
question bank, runs the hybrid search, and reports hit@k and MRR@20 at two levels: the file, and the
section inside it. No generator, no judge. A question is scorable at section level only if some
chunk of its marked source actually carries that section; the rest are excluded and counted, not
scored as misses.

**It measured the index, not the corpus.** pgvector's hnsw is approximate, and rebuilding the same
index over the same data moves the output. Measured, on the same corpus, index rebuilt: on 100
questions, 2 flips and a bootstrap interval 0.0014 wide - effectively deterministic. On 30
questions, delta MRR +0.044 with an interval of [-0.033, +0.133]. That last number settled an
argument about set size faster than any reasoning had: a thirty-question set cannot resolve anything
smaller than 0.13, whatever its labels look like. The fix was not to live with the noise but to
remove it: the primary criterion runs with `enable_indexscan = off`, an exact scan with exact
distances. The arc measures the corpus. What the index costs is reported beside it, as recall@20 of
hnsw against exact search, and gated in preflight.

**And that fix did not work for a day**, which is the fourth lie and the worst of them. The setting
was applied to the connection the script had open; `hybrid_search` opens its own from the pool, so
it reached a different backend and did nothing. Everything labelled `search: "exact"` had gone
through the index at the server default `ef_search = 40`, and `--recall` had been comparing the
index against itself, which is why it returned exactly 1.0 at every setting and why the preflight
gate built on it could never fail. Found by an independent code review, not by us.

The correction is on the pool checkout, not on connect: a connection created before the script ran
returns to the pool without the setting, and the first version of the fix - a `connect` listener
plus `engine.dispose()` - still let those through, so half the questions were measured one way and
half the other. What the index actually costs, measured honestly:

| `hnsw.ef_search` | recall@20 against exact |
|---|---|
| 40 (the server default, what production ran) | **0.9148** |
| 100 | 1.0 |
| 200 | 1.0 |

So the interactive and agent paths had been losing roughly one section in twelve out of the exact
top-20, and nobody could have seen it, because the number that would have shown it was a comparison
of the index with itself. `hnsw.ef_search` is now a setting in `service.retrieval`, pinned to 100 on
both engines and recorded in every run's snapshot. This is a change to the measured system, made
before the controls were taken, on a number rather than a taste.

**Its candidate pool was a quarter of what it claimed.** The script asked for 100 candidates, but
the per-leg limits still came from the config at 20 and 20, so the final limit cut a union of at
most 40 rows and cut nothing. Fixing the limits only got the pool to 40, because the index was
still capping it at `ef_search`; it took the fourth lie above to actually reach 100, and the guard
that would have caught both - a warning when the pool comes back shorter than the one asked for -
was only written afterwards. This mattered less for the numbers than for their comparability: with
one chunk per section, 20 rows yield 20 sections; with `baseline`, the same 20 rows yield 8. The
"after" corpus would have won mechanically. The bias was removed before it could be collected.

**It left a probe index in the measured system.** While proving that a generic plan cannot use a
partial index, two throwaway indexes were created; one was dropped and one was not, and dbmate
dumped it into the committed schema. The measured system had an artifact of its own measurement in
it for about an hour.

## What a bound parameter does to a partial index

A variant needs its own vector index, or an approximate search over all variants post-filters the
foreign ones away and returns fewer rows than asked - a quiet loss that reads as bad chunking. So:
one partial hnsw index per variant, and no catch-all.

Both database drivers here bind parameters on the server, and a generic plan cannot prove
`WHERE variant = 'x'` from `variant = $1`. Measured on our own table:

```
forced generic plan  ->  Index Scan using data_chunks_source_id_idx   (partial index NOT used)
forced custom plan   ->  Index Only Scan using tmp_probe2             (used)
```

Under the default `auto` mode the planner kept the custom plan, because it was cheaper - so this
would not have broken for certain, it would have depended on cost estimates that drift with the
table. An instrument does not get to depend on that: `plan_cache_mode=force_custom_plan` on both
engines, next to the statement timeout that was already there. The catch-all index was dropped for a
second reason, too: without a fallback, forgetting the variant becomes slow instead of quietly
wrong.

## The keyword leg: a hypothesis that survived four rounds and died to a grid

The hybrid has two legs, dense and keyword, fused by RRF. The keyword leg builds its query with
`plainto_tsquery`, which joins every lexeme of the question with AND - the chunk must contain every
content word. On the Russian paraphrase set, the leg alone finds the marked source 0 times out of
100. Switch to OR, and it finds it 49 times. The conclusion looked obvious: half the hybrid is
switched off, turn it on.

The grid says otherwise. Sixteen points per set, three sets, exact search, MRR@20:

| set | level | AND | OR |
|---|---|---|---|
| `paraphrased_ru` | section | **0.6263** | 0.3916 |
| `paraphrased` (en) | section | **0.6598** | 0.5107 |
| `curated` | file | **0.7272** | 0.6983 |

The first two rows survived the instrument being fixed; the third did not. Measured through the
index, `curated` was the one set where OR appeared to win by a clear margin, and that margin was the
artifact. On an exact search OR loses on both paraphrase sets by 0.15-0.23 and lands inside
`curated`'s own noise floor of 0.13 either way, so nothing supports it anywhere.

Narrowing the keyword leg to the production 20 does not rescue it. `ts_rank_cd` and length
normalisation are indifferent under AND and harmful under OR. Querying under both language configs
adds matches and no correct hits.

What the earlier numbers measured was the leg's recall **alone**, not its contribution to the
fusion. Under AND the leg fires rarely and lands on the right chunk; under OR it fires always and
lands anywhere, and RRF dutifully promotes the noise. Stated precisely, because the general claim
would be wrong: **with an equal-weight RRF (`rrf_k = 60`, both legs at the same weight), a full
keyword leg interleaves with the dense one and costs 0.15-0.23 MRR at section level.** A weighted
RRF, or the keyword leg as a filter over the dense one, was not measured and is a candidate arm
after the hygiene, not before - it would otherwise be tuned on the very sets the chunking is later
read on.

One switch was kept, and the honest version of why is not the one first written here. `langdetect`
calls short Russian questions English ("Локи в PG?", "зачем нужен vacuum?"), and a wrong text-search
config kills the match outright, so a cyrillic-ratio rule replaced it. The number offered in support
- file MRR on `curated` moving from 0.6276 to 0.6440 - was measured through the index and does not
survive: on the corrected grid the two settings tie to the fourth decimal on every set. It stays as
the default because it is deterministic and because the misclassification is real and shown, not
because it measurably won anything. How much it touches, as a number rather than three examples.
Counted over every question in the bank, bucketed by length rather than by set - because which sets
count as "short" was a judgement, and the length is the thing that actually decides it:

| question length | rules disagree |
|---|---|
| under 40 characters | **11 of 340** |
| 40 to 80 | 0 of 2 272 |
| 80 and over | 0 of 1 170 |

Those buckets are still a choice, so the script also prints the lines that are not: **the longest
question the two rules disagree on is 36 characters, and all 3 525 questions longer than that
agree.**

There is now a third rule, and it is the default. Ask each text-search configuration whether it
recognises the question's own function words: the one that drops the most tokens as stopwords is
the language, and if none of them react, fall back to the alphabet share. No library and no model,
and it extends to any of the 29 configurations Postgres ships simply by naming the language in the
config. Bulgarian is not among those 29, which is the same wall our search hits anyway.

Measured before it was switched on (`datasets/measurements/query_lang/`): identical to the fourth
decimal against the alphabet rule on all three sets, because under `and` the keyword leg is nearly
empty and the language mode cannot move ranking. It disagrees with the alphabet rule on exactly one
question of 3 782, and it is right there: "что такое structured outputs в OpenAI API?" has more
Latin than Cyrillic, so the alphabet says English while «что», «такое», «в» say Russian. That class
- a Russian question full of English technical terms - is the house style of this corpus.

The same rule now also picks **the language the answer comes back in**, which was still on
`langdetect` and was the only place where the misdetection reached a person: "Локи в PG?" earned a
`Respond in English` directive. Like `hnsw.ef_search`, this is a change to the measured system made
before the controls are taken, chosen on correctness rather than on a metric, and recorded in the
snapshot; `query_lang` is a pinned field, so agent runs from before this commit are not comparable
on it. `langdetect` is reliable on a sentence and fails on a fragment; `curated` is
the set made of fragments, which is why it was the only one where the switch appeared to matter. That distinction - kept for a reason, not for a
number - is the whole point of keeping this journal.

Grid summary on disk: `datasets/measurements/2026-08-26_keyword_switch_grid.txt`. The 48 per-point
reports are 14 MB and stayed out of the repository; `scripts/keyword_switch_grid.sh` regenerates
them, and it runs in about a minute per set.

## The criterion set

`curated`, the hand-written set, turned out to have 20 of its 53 marks pointing at a source that is
not in the corpus at all, and 2 of its 30 questions unhittable by construction. With a noise floor
of 0.13 on thirty questions it could not have been a criterion anyway; it is descriptive now, and
its stale marks are visible in every preflight run until the marks table exists to clean them
properly.

The primary set is new. The old `paraphrased_ru` was drawn with `random()` and cannot be rebuilt, so
it was not extended and not replaced - it stays as the continuity set for agent runs, and it was
exported to a file first, being the one thing here that is irrecoverable. Beside it now:
`paraphrased_v2` (342) and `paraphrased_v2_ru` (344), drawn under seed `hygiene_v1`, stratified two
per repository across 173 of them, and exported to `datasets/questions/`. A seed is mandatory in
code for any set whose name is not a test name.

The seed fixes which originals are drawn. It does not fix the paraphrase text - the model will word
it differently next time - so a criterion set is reproducible by storage, not by regeneration. That
is why it lives in a file with the text of its original beside it, and why seeding restores the link
by hashing that text.

**The "before" for the hygiene arc**, exact search, pool 100:

| set | level | hit@1 | hit@5 | hit@10 | MRR@20 | n |
|---|---|---|---|---|---|---|
| `paraphrased_v2_ru` | file | 0.712 | 0.898 | 0.945 | 0.7907 | 344 |
| | **section** | 0.497 | **0.767** | 0.829 | **0.6096** | 344 |
| `paraphrased_v2` | file | 0.757 | 0.936 | 0.971 | 0.8345 | 342 |
| | section | 0.556 | 0.801 | 0.883 | 0.6607 | 342 |

These are the numbers after the fourth lie was fixed. The ones taken before it differ in the third
decimal and, on the primary criterion, not at all: section MRR on `paraphrased_v2_ru` was 0.6096
either way. The baseline survived; its label did not.

Every question is scorable at section level (344/344, 342/342), which is the check that the links
and headings are intact rather than the corpus being good. Files:
`datasets/measurements/before_paraphrased_v2*.json`.

Note where the headroom is. At file level the hybrid already reaches 0.945 at depth 10 - there is
almost nothing left to win, and the dense leg carries nearly all of it. At section level - the right
section inside a four-hundred-page README - hit@5 is 0.770. That is the number the hygiene is for, and it is the reason the primary
criterion is MRR at section level rather than any hit@k at file level.

## The decision rule, written before the numbers

- **Primary:** paired delta of MRR@20 at **section** level on `paraphrased_v2_ru`, between the
  `baseline` variant and the new one, over the stored question embeddings. The bootstrap interval
  over questions must exclude zero and the delta must be positive. Printed beside it: the share of
  questions better / worse / unchanged, and a sign test, because on a few hundred questions with a
  heavy tail the interval can fail to resolve while the picture is plain.
- **Second primary:** hit@5 at section level on the same set does not fall.
- **`interview` is a ceiling, not a criterion.** Its questions are literally the `##` headings cut
  out of the same READMEs, so any chunking that puts a heading in its chunk wins it trivially.
- **Ranks are compared without the retrieval threshold** (2.0). With it, the delta would mix "ranks
  moved" with "0.55 now cuts a different share of candidates".
- **The metric counts collapsed sections**, distinct in order of first appearance, over a candidate
  pool with slack. Otherwise "after" wins or loses mechanically by how many children a question has.
- **Paired on the intersection.** A question becomes scorable at section level only if some chunk
  carries its section, and hygiene will make more of them scorable; the primary delta is read on
  questions scorable in both corpora, and the growth in scorability is reported separately. Counting
  the previously unscorable as misses would credit chunking with the work of labelling.
- **The primary set measures the dense leg**, because the keyword leg is near-empty under AND. Said
  out loud so that a later win from reviving the leg is not attributed to the chunking.
- **Veto**, on agent runs against two controls on the new corpus: refusals on `off_domain` no lower
  than control minus noise; false refusals and false handoffs on `in_corpus` no higher than control
  plus noise; median latency on the corpus pool at most 1.15x.

## Traps, for whoever crosses this way next

- **A saved file spends the card.** The API runs under `uvicorn --reload`, so writing any file under
  the mounted tree restarts it, and the restart ran `bootstrap`, which enqueued jobs. With variants
  that stopped being noise and became dangerous: the first save of `config.yaml` naming a new, empty
  variant would have started a full 20-40 minute indexing run - past preflight, with whatever
  chunker happened to be in the code. Bootstrap no longer auto-indexes a named empty variant, and
  the embedding job deduplicates against the queue instead of trusting its own idempotence.
- **A mass update degrades an hnsw index until vacuum.** Two full-table updates on the frozen
  baseline left dead entries in the graph, and the first "before" was taken in that state. It did no
  harm only because autovacuum happened to fire in between. `n_dead_tup` is now a preflight check.
- **Artifacts, not intentions.** Three separate times a claim of the form "X is written to the
  report" turned out to be false when the file was opened: fields that never landed, a constant
  printed where a parameter belonged, a stale file disagreeing with a table. The rule earned here is
  literal: open the file before writing the sentence.

## What this bought, and what it did not

It bought a corpus that can be varied without losing its predecessor, an instrument that measures
the corpus rather than the index, a criterion set that can be rebuilt from a file, and twelve
preflight checks that now stand between a run and the traps above.

Worth separating, because in a month this entry is the only witness: preflight caught none of the
three lies listed here. The probe index and the residency check aimed at the wrong model were found
by the auditor reading the schema and the job options; the candidate pool and the exact-search
question were found by the auditor reading the script. Preflight is what keeps them from coming
back - it holds the plan check, the dead-tuple bound, the recall gate and the switch comparison -
but it was written after the fact, and every check in it exists because something got past first.

It bought no retrieval improvement at all. Not one chunk has been re-cut. The numbers above are the
line the hygiene has to beat, and the point of the whole exercise was to make sure that line is
worth something before drawing it.
