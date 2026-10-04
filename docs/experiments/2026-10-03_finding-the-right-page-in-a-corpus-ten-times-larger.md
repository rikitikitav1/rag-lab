# 2026-10-03 - Finding the right page in a corpus ten times larger

The corpus grew from about 30 thousand chunks of interview banks, a few books and three documentation sets to about 785 thousand: the interview banks, 28 books, 70-odd documentation repositories and sites, the man pages, in English and Russian. The question this run asks: does search still put the page or chapter file that holds the answer among its first five results, now that every topic has close neighbours from other sources and versions? The exact section is read beside it. These runs measure the level on the new corpus; there is no `clean_1024` arm for this set, so the cost of the larger corpus itself is read only from the interview banks, run on both corpora below. The bar was set by the executor before the run, at 0.75 on the lower edge of the interval; a larger corpus with close copies is expected to cost something, and the cost itself is not a defect.

## Setup

**Set** `big_corpus_v1` (552 en, 552 ru questions: accepted pairs whose gold section an active source holds) · **corpus** `clean_big_1024` (784,869 chunks in 259 active sources) · **arm** the serving single-shot pipeline, runs `big_corpus_v1_serving_en` and `_ru` (jobs 6144, 6145, code `31ce483`, `loaded_differs: null`) · **preregistration** `arc7_close_en`, `arc7_close_ru` (written 00:05:21 UTC, the runs queued nine seconds later, the addendum of everything read beside the bar in the arc log at 23:32 the evening before)

- **Questions.** Ten pairs per source, one fact asked in English and in Russian, written by Sonnet agents from blocks of each source's own sections under the stand's generator rules, the first batch of every round read by hand. The set is `sources/questions/big_corpus_v1.jsonl`. Every pair then went through the stand's own sieve: a local reader (`accepting`, llama3.1 8b) answered each question blind from its gold section by a quote, and the judge (`judging`, qwen 2.5 7b on vLLM) settled the pairs the reader left open. Only accepted pairs are asked: of 796 pairs over 87 sources, 559 accepted, 170 refused (the reader 113, the judge 57), 67 left open. The sieve refused over half of four sources: elasticsearch-guides (2 of 7 kept), object-pascal-handbook (3 of 10), python-docs (3 of 10), redis-in-action (4 of 9). Seven accepted pairs are out of the population because their sources (object-pascal-handbook, redis-in-action) were left inactive to keep the served search of the night unchanged; dedup took no gold.
- **Column.** `hit_at_5`: the gold file is among the first five distinct (source, leaf heading) pairs the arm returned, read on the chunks the gates kept; a section's copy in another version is one pair, two sections of one file with the same leaf are one pair, a copy of the text in another source is a miss. Half the questions' gold files hold seven sections or more, so this is the page, not the section: `section_hit_at_k` of the same runs is quoted beside it with no bar.
- **Corpus rules new in this build**, each through its gate before it became a default: trust-ordered dedup across sources (official > book > notes, then freshness), frontmatter read away, table cell padding squeezed, `.mdx` read as markdown, folders skipped by path (`skip_paths`), pictures beside documents left out.
- **Thresholds** re-read on this corpus with `scripts/threshold_calibration.py` on the bank sets only (`threshold_calibration_clean_big_1024_20261002.json`): topic en 0.4287 ru 0.4707, weak 0.3861, pool cut 0.56. The serving arm reads only the pool cut; 0.18% of the English set sits above it and none of the Russian. The topic and weak gates are the agent's: on these thresholds the agent arm would refuse 4.47% of the English and 1.25% of the Russian questions as off-topic.

## Result

| half | n | hit@5 | 95% interval | bar (edge) | verdict |
|---|---|---|---|---|---|
| en | 552 | 0.817 | 0.784 - 0.848 | 0.784 against 0.75 | cleared |
| ru | 552 | 0.741 | 0.705 - 0.777 | 0.705 against 0.75 | missed |

Read beside the bar, declared before the runs:

| | en | ru |
|---|---|---|
| exact section among the kept chunks (`section_hit_at_k`) | 0.773 | 0.699 |
| interval over sources (a source's questions drawn together) | 0.777 - 0.854 | 0.698 - 0.780 |
| mean of per-source rates (85 sources) | 0.816 | 0.737 |
| misses: pool cut / not found | 1 / 100 | 0 / 143 |
| misses whose answer evidence sits in a top-five chunk of another file | 10 | 11 |
| question and source in the same language | 0.833 (n 520) | 0.906 (n 32) |
| question and source in different languages | 0.563 (n 32) | 0.731 (n 520) |
| anchored by a rare identifier / not | 0.762 (80) / 0.826 (472) | 0.762 (80) / 0.737 (472) |
| shares a word with its heading / not | 0.859 (220) / 0.789 (332) | 0.807 (114) / 0.724 (438) |
| refused with the context in hand (the generator, not retrieval) | 31 | 25 |

The two sources that are one text under two names (redis-doc and redis-docs-new, nginx-org-en and nginx-org-ru) hold 27 pairs a half: 4 misses in each half, 2 and 3 of them with the evidence in a top-five chunk.

The Russian miss sits on the crossing of languages, and that is a paired number: the same 520 facts over English sources read 0.833 asked in English and 0.731 asked in Russian, ten points. 520 of the 552 Russian questions cross. The two cells of 32 questions (Russian over Russian sources 0.906, English over Russian sources 0.563, three Russian sources) carry bands of about ±0.1 and are no evidence on their own. The Russian bar was argued from same-language sets, so its miss says the embedder and the keyword leg do not carry a question across languages well enough on this corpus; the size of the corpus is not what these runs measure. The worst sources: English postgresql-internals-18 0.14, altinity-kb 0.33, python-docs 0.33; Russian altinity-kb 0.17, protobuf-docs 0.17, open-data-structures-python 0.25.

## Judges

Not measured. The closing column is a retrieval column and needs no judge; the owner dropped the judge pass on these runs and with it the stand-against-guest correlation («нам не нужен судья, если такой hit@5»). A hand read of eight answers: where search found the page the answers were right and cited it; where it missed, the generator either refused honestly or answered with confidence and wrongly (the PostgreSQL selectivity question), and one right answer scored as a miss came from a neighbouring man page.

## Depth: hit@5, @7, @10

A retrieval-only pass over the same 552 + 552 questions, stored question vectors, the serving arm's limits, ranks over distinct files (so a little above the closing column, which counts source and section pairs). Counted after the result, beside it, not instead of it.

| | hit@5 | hit@7 | hit@10 | gold nowhere in the 40-candidate pool |
|---|---|---|---|---|
| en | 0.848 (0.817 - 0.877) | 0.859 | 0.868 (0.839 - 0.895) | 71 (12.9%) |
| ru | 0.775 (0.739 - 0.810) | 0.794 | 0.806 (0.774 - 0.839) | 99 (17.9%) |

Depth buys two to three points. What search finds it mostly puts in the first five; the loss is the pool itself, where the gold never arrives.

## The interview banks on the new corpus

The like-for-like price of the larger corpus: the same bank questions, the same column (hit@k over the banks' file marks), the old corpus against the new one.

| | `clean_1024`, 2026-08-28 | `clean_big_1024`, 2026-10-03 |
|---|---|---|
| en (820) | 0.918 | 0.766 |
| ru (823) | 0.892 | 0.683 |

Fifteen and twenty-one points. A bank's gold is its README; on the big corpus the official docs and books hold the same topics and outrank it, and dedup ranks a bank below every other source, so a bank text another source also holds is gone from the bank. The old runs are from another code and other thresholds (pool cut 0.55, not 0.56); no same-night control was run on `clean_1024`, for the card. For a user the doc answering instead of the README is no loss, which is what scoring by the answer's twin would show.

## What loses the gold

Read from the misses and from the corpus, read-only:

- **Garbage from another source in the five.** Of the nine misses of altinity-kb, seven had ClickHouse changelogs in the five. Changelogs, release notes, NEWS files and blogs sit in 22 sources, about 37 thousand chunks (ClickHouse 16 thousand, 20% of the source; pgbouncer half of its source; fastapi 44%; go 42%).
- **Old versions of the same pages.** redis-docs-new kept 33 versions of one library's docs plus older operator manuals: 39 thousand chunks, 54% of the source.
- **The page title lost.** A page whose title lives in its frontmatter and whose body opens with sub-headings was rooted at its first sub-heading: «Redis hashes» read as «Basic commands». 22 thousand files in 21 sources.
- **Site markup left as text.** Hugo shortcodes in 55% of the Kubernetes chunks, 14% of redis-docs-new, 12% of docker, 18% of grpc; MDN's macros on 14.7 thousand pages, three quarters of them links that hold the page's words.
- **Twins.** The same topic in two languages or two places: the English PostgreSQL docs outrank the Russian PostgreSQL book (English questions on the book 0.14), ClickHouse carries a near-complete Russian mirror, protobuf's API text sits in several pages, `min` collides across four databases. Cleanup does not fix these; a twin-aware column or a translated query might.
- **Copies inside a source and heading-only stubs.** mdn 32 thousand surplus copies, man pages 24 thousand (section stubs, colophons), sqlserver 9 thousand; 52 thousand chunks have a body under 60 characters, some of them real one-line answers.

## Cleanup after this measurement

Done the same day, each through its gate, before the second measurement:

- changelogs, release notes and old versions left out by path in 22 sources (a door sets a source's skipped paths in place);
- a page roots at its frontmatter title unless its body opens with its own top heading outside code (22301 of 61748 files changed, no double top heading);
- Hugo shortcodes rendered by a name table (`markup: hugo`: a tab keeps its label, a term its word, a heading its title), MDN macros likewise (`markup: mdn`: a reference keeps its word, a badge its note); no shortcode or macro left, no reference word lost;
- the five per-source Python readers replaced by declared fields and the markup families: the same chunks byte for byte, or the same text with the frontmatter lead added;
- the 67 pairs the judge split (English YES, Russian NO in 49) re-judged by a second judge: a blind read found the Russian question sound in 40 of 40 splits.

Left for the cycle after: heading-only stubs merged into their neighbour, copies inside a source, the PDF cover text that roots every heading of ten books.

## The second measurement

A new preregistration on the cleaned corpus, the same bar (0.75 on the lower edge for both halves): strict `hit_at_5`, beside it a hit on the answer's twin (the answer's evidence inside a top-five chunk) and a control with search filtered to the gold's own source, which tells a cleanup that broke a source from a source outranked by others; hit@5/7/10. The query translation for the keyword leg (a small local translator, opus-mt-ru-en) is its own A/B on the Russian questions. The reranker is not part of it.

## Decision

## Decision

The English half clears the executor's bar, the Russian half misses it. The bar was the executor's, not the owner's, and the chain ran past a preflight with seven FAILs on the executor's reading (the list below); search moves to `clean_big_1024` only on the owner's word. No second promise was opened the same night after the miss; the second measurement above follows the cleanup.

## Caveats

- The questions were written by one model family (Sonnet) and checked by the stand's reader and judge, not by a person; a question built on a premise its section never states can still pass (the known limit of the sieve).
- Known defects left in `clean_big_1024`: code crumbs cut off from their block (about 1.5 thousand chunks), link-only service rows (about 760 confident), empty and redirect stub files (707), a heading echo in short sections. Fixing them changes the cut and waits for the next index.
- A copy dropped by dedup is gone from the losing source until it is indexed again; the losing row names its keeper (`raw.copies_kept_by`).
- Sources left out on purpose: the Kubernetes API reference, Altinity release notes, the React blog (generated or changelog text); mongodb-manual (its onboard through Docling would have held the card through the closing runs; its pages are fetched for the next build); the two scanned copies of books already in the corpus.
- The chain ran past a preflight with seven FAILs, output in `datasets/measurements/preflight_20261003_0250.txt`: declaration digests moved by a new empty field (25 of 25 recomputed), route digests moved by default-off knobs added or removed (49 of 52; nginx-org-en, nginx-org-ru and rfc9110, 24 pairs, unattributed, their questions written from the same raw text the index cut), lost tuned files, rows cut before the day's index rules, a reachability query that timed out (replaced by the population's own check), the working tree (since committed), an idle card read as no generator.
- The column reads the page file; half the gold files hold seven sections or more. The section number is beside it, without a bar.
- `hit_at_5` is retrieval; the generator (llama3.1 8b) still refused 32 English and 25 Russian questions with the context in hand, often with the gold in the five. `answered` is not `found`.
- Copies of one text inside a source take slots in the five (the man pages' aliases of one page, redis-docs-new, sre-workbook): by `content_hash` of the served chunks, 8 English and 16 Russian misses carry a copy; if every such miss flipped, the halves would read about 1.5 and 3 points higher, and the cases where the copies are the gold's own source under another file name are under one point. Dedup read across sources; copies within a source, which the raw report flagged, stayed in the index.
- The thresholds were fitted on the bank sets and are shared, so the served search reads them too from the night's restart.
- Tries under the promises: exactly the two closing runs, 6144 and 6145.
- Every number here dies with `clean_big_1024`; the measurement files and this entry hold it.
