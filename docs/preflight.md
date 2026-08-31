# Preflight: what it refuses, and why each refusal exists

`scripts/preflight_grid.py` runs before a grid and after one. It is not a test suite. Tests ask
whether the code is right on fixed inputs; this asks whether the stand is still the stand the
numbers will claim it was.

```bash
python scripts/preflight_grid.py                       # sixteen checks, exit 1 on any failure
python scripts/preflight_grid.py --verify RUN [RUN..]  # a finished run instead of the stand
```

## Why a measuring rig needs one

Three properties of this project put it here, and none of them is unusual once a repository starts
producing numbers rather than features.

**The instrument lies through state, not through code.** Which model is resident, which worker
process is running, what is in the table, what cut produced it, which switches the config held when
the run started. None of that is in a diff, and all of it decides what a run measured. A green test
suite says nothing about any of it.

**A run costs from forty minutes to a day of GPU.** The mistake surfaces when the results are read,
which is a day later, and the fix is to run it again. The checks below cost about twenty seconds.

**A spoiled run does not look spoiled.** It completes, it writes plausible numbers, the numbers go
into the journal, and later work is built on them. There is no exception raised anywhere. This is
the property that makes a preflight worth its maintenance: everything it catches is silent.

Most of these checks exist because the corresponding trap had already cost a grid. Where that is
true, the incident is named below, because a check whose reason is forgotten is a check somebody
deletes.

## The sixteen checks

### Is the code that runs the code we think runs

**`tree_is_clean`** refuses an edited working tree. Every run records `code_version`, the current
commit, in its snapshot. With uncommitted edits that field names code which never ran, and the
snapshot becomes a confident lie about a run nobody can reproduce.

**`worker_newer_than_sources`** compares the worker container's start time against the newest file
under `app/` (and `config.yaml`, since a run reads its thresholds from there). The API runs with
`--reload` and the worker does not: it holds its code in memory from the moment it started. Edit a
file, and the worker keeps executing yesterday's version while the API serves today's. This cost
three separate incidents in two days: prompt keys that were never re-read, an experiment that
silently failed to aggregate, and a smoke test whose second arm died on an import.

**`worker_imports`** executes the orchestrator and agent imports inside the worker. An import error
in a rarely-loaded module surfaces as a failed job three hours into a run, not at startup.

### Is the machine that computes the machine we think computes

**`models_are_on_the_card`** asks what is actually resident, per role, and refuses when anything is
loaded with zero VRAM. It asks residency rather than free memory on purpose: the scheduler keeps
reporting free VRAM after the card has gone away, so "there is room" is not evidence that anything
is on it. A model that spilled to the CPU produces correct numbers about thirty times slower, which
turns a forty-minute arm into a day and looks like nothing but slowness.

It names the role beside the model, because a job runs one model and the others being resident
proves nothing about the one that matters.

The cross-encoder reranker is torch inside the API and worker processes rather than an ollama model,
so this check cannot see where it sits, and a probe from outside cannot either: `docker compose exec`
starts a new process where the model is never loaded. What the preflight reports about the card is
therefore the driver's own numbers, free and total, which are honest from any process. Where the
reranker actually sits is asked by the run, in the process that holds it, and a run refuses on a
spill exactly as it refuses when ollama drops a model to the CPU.

The incident that put it there: on 30.08 a paraphrasing model left resident with
`keep_alive: Forever` took 6.4 GB of an 8 GB card, the reranker fell back to the CPU with a warning,
and the run would have taken thirteen times longer with identical numbers. It was caught by eye.

**`window_matches_config`** compares the context window the config declares against the window the
server reports for the generator that is actually loaded. A stray environment variable in a running
container beat the config once, so the server is treated as the authority and the config as the
claim. It asks whichever generator is loaded rather than the configured one, because a run with a
model override leaves the configured name unloaded, and reading that as "the server says nothing"
made the check fail on every override.

**`queue_is_idle`** refuses when jobs are queued or running. Two jobs on one card do not fail; they
take turns, evict each other's models, and produce timings that belong to neither.

### Is the corpus the corpus we think

**`every_variant_cuts_into_its_own_rows`** is the strongest check here. It re-cuts every indexed
variant from the sources and compares the **text** of each chunk against what the table holds, not
the row count per source. The reason is exact: when the parser changed, fourteen of the sixteen
sources that changed kept their row counts unchanged. A count-based check would have passed while
the corpus underneath the numbers had moved.

It asks every indexed variant rather than the served one. `notes` is expected to drift, because it
is a live local directory the owner writes in, and that one family is allowed to differ.

**`corpus_variant_is_usable`** refuses when the served variant holds no chunks, and prints what the
other variants hold. A variant named in the config but never indexed searches an empty set and
returns nothing, per question, forever.

**`table_is_vacuumed`** refuses more than a thousand dead tuples in `data_chunks`. A mass update
leaves dead entries inside the hnsw graph, and retrieval quietly returns fewer neighbours than it
should. Nothing errors; recall just shrinks.

**`schema_holds_no_variant_indexes`** refuses when `db/schema.sql` carries a partial index belonging
to a corpus variant. A variant is a line in the config and its index is built at runtime, so an
index in the dump means the dump has become a function of whichever variants happened to exist on
the machine that produced it.

### Is the depth the depth we recorded

**`every_variant_walks_its_index`** asks the planner whether the configured `ef_search` still
produces a plan containing an `Index Scan`, on **every indexed variant**, and refuses when any of
them falls back to sorting the table.

This is the check that makes pinning a depth safe. Postgres chooses between walking the hnsw graph
and reading the table filtered by variant, priced per page. Past the crossover it silently prefers
the read, which is exact search: the results are perfect, the recall against exact search is 1.0 by
construction, and the record says the run used hnsw at some depth. Nothing anywhere reports the
substitution.

It asks every indexed variant because the crossover moves with what is in the table, and because it
turned out to be a property of the **variant**, not of the table alone: on 30.08 a variant with
7,102 rows had its planner walking its smaller partial index to 400 while two variants of 12,102 and
13,068 rows in the same table stopped at 200. Checking only the served variant is how the crossover
moved twice before anybody noticed.

**`tuned_numbers_still_describe_the_corpus`** reads every `# tuned: file=` line in `config.yaml`,
opens the measurement each one points at, and compares the corpus fingerprint recorded there with
the live one. A number that came out of a measurement is only as good as the corpus it was measured
on, and a comment claiming provenance cannot be checked while a file carrying a fingerprint can. It
refuses when a named file is missing or describes a different corpus, and says how many carry no
comparable fingerprint. On its first run it found the criterion's zero point taken on 12,108 chunks
against a table holding 12,102.

**`index_is_alive`** is a smoke test and says so in its own output: recall@20 against exact search on
a few dozen questions, with a deliberately low floor. It catches an empty or half-built index. It is
not the gate on depth; that gate is `max_mrr_loss`, measured on the full criterion set with
`retrieval_report.py --index-cost`.

### Are the questions the questions we think

**`marks_are_reachable`** finds questions whose marked source file exists in no chunk of the served
variant. Such a question can never be answered correctly: it counts as a miss on every arm, in every
comparison, forever. On the sets a verdict is read on (`retrieval.criterion_sets` and
`retrieval.veto_sets`) this refuses; elsewhere it prints a note. The list of decisive sets is asked
of the worker rather than written in the script, because the script's own copy said
`paraphrased_ru` for a day after every measurement had moved to `paraphrased_v2_ru`, and nothing
complained.

**`one_question_per_original`** refuses when an original question has more than one paraphrase. A
requeued paraphrasing job re-paraphrases what it already did, and a fresh paraphrase is new text
with a new hash, so no unique constraint catches it. Two paraphrases of one original quietly
double-weight that original in every paired comparison.

### Are the switches the switches the last run used

**`keyword_switches_match_the_worker`** compares the keyword-leg switches the config on the worker's
disk declares (`keyword_query`, `keyword_rank`, `keyword_norm`, `query_lang`, and the resolved depth)
against the switches recorded in the most recent answer log. It runs `python -c` inside the worker
container, which reads `config.yaml` fresh, so it sees the file rather than the memory of the
process that is actually serving: a worker running yesterday's code is caught by
`worker_started_after_newest_source`, not here. A switch flipped between two arms is invisible in
the numbers, and both arms look like valid measurements of different things.

It compares the **resolved** depth rather than the declared one, and resolves it for the variant the
logged row was taken on.

### The note that never refuses

**`halves_of_pairs_are_counted`** prints how many originals are missing half of their pair. Every
original produces two rows, an English paraphrase and its Russian translation, and a job that died
between them leaves a half. That is a fact worth seeing and not a reason to stop, so it prints as a
note. Standing among the sixteen checks made it look like a gate with a permanently green
light, which is why it was moved out.

## `--verify`: the same idea, after the fact

`--verify` reads finished runs instead of the stand, and asks what a comparison needs in order to
mean anything:

- one row per question, no duplicates, and the same question set across the runs being compared;
- fewer than a tenth of rows are errors;
- for agent runs, the snapshot carries the orchestrator and the context window, and there is exactly
  one of each across the run. An options payload without an orchestrator silently runs the
  hand-rolled loop, which is a different system;
- every setting in `PINNED` is identical across the arms: corpus fingerprint, variant and its cut
  policy, keyword switches, depth, `k`, hops, gates, context window, reranking, distance threshold,
  MCP configuration, code version. Two arms may differ in the axis under study and in nothing else.

It also refuses a setting that changed **halfway** through a run, which reading only the first row
would miss.

## Reading the output

Each line is `ok`, `FAIL` or `note`, followed by what was actually observed rather than a verdict
alone, so a failure can be acted on without re-running anything by hand:

```
ok   working tree: clean
ok   worker started 2026-08-30T17:48:47Z, newest source config.yaml
ok   depth (100, 25170 rows estimated): baseline@100, clean_1024@100
ok   tuned numbers: 4 of 8 files compared; 4 carry no comparable fingerprint
ok   index alive: recall@20 against exact 1.0 on 40 questions (liveness, floor 0.9; the gate is max_mrr_loss)
note originals missing half of their pair: paraphrased_v2=3 (sets with one row per original
     are not counted: veto)
```

Exit code is 1 if any check failed, 0 otherwise, so it drops into a script before a long run.

## Adding a check

The bar is deliberately high: a check earns its place by naming the run it would have saved. Two
rules follow from that.

**A check that gates nothing tells the next reader that the problem is handled.** One that reads
something out without ever refusing belongs in `NOTES`, not in `CHECKS`.

**A check states what it saw, not that it passed.** `index_is_alive` prints its floor and calls
itself a liveness test in its own message, because a line reading `ok` next to a recall number is
otherwise read as the gate on retrieval quality, which it is not.
