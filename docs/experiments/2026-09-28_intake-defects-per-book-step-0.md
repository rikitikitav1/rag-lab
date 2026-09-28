# 2026-09-28 - Intake defects per book: the step 0 baseline

Before fixing the intake any further, every book in the store was read on the current defaults and the second chapter of each book was checked by hand, page by page, against the page image and the PDF's own text. A first look at 5 to 10 pages of each whole book started the list of defect kinds; as a per-book measure it was noise, so it is not reported here. This entry keeps, per book, the number of pages that came out clean on the second chapter. Each later fix adds its own column on the same chapters, so a book that gets better or worse stays visible, instead of an average hiding it.

## Setup

**Books** 25 from the store plus 2 Russian books with a text layer and their image-only copies · **defaults** of 27 September 2026 on branch `intake` (docling-parse 7.20.0, code from the layer, fences for code with no box, joins over page breaks, outline levels, the second reading under layer F1 0.95; row rules, seam shifting and HTML title levels off) · **chapters** the chapter set `datasets/converter_gold/chapter_set.yaml` (the second chapter of each PDF document, whole; 45 documents plus the 4 Russian sources), runs `chapter_set_en_20260927f` (job 3505) and `chapter_set_ru_step0base_off` (job 3541, today's fixes switched off by the run's knobs).

Reviewers: Claude agents, two on FreePascal and GoalKicker (Opus), the rest Sonnet, each going page by page against the PDF's text layer and a render where the layer could not settle it, with the instruction in `~/working_docs/projects/rag-lab/notes/reference/step0/CHAPTER_INSTRUCTION.md`; the merged list of defect kinds is in `CHAPTERS_SUMMARY.md` in the same folder. Where two reviews of one document differ, the more pessimistic is kept.

## Result

A page is clean when its review names no defect on it. The defects columns count rows, not pages: one page can hold several. Severity is by what retrieval loses: high means a question about that content would fail or land on the wrong chunk.

| source | lang | pages | clean pages | defects h / m / l | ch0 | ch1 | ch2 | ch3 | ch4 | ch5 | ch6 | clean pages, run 6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| freepascal | en | 128 | 17 | 79 / 80 / 39 | 397 | 384 | 37 | 37 | 25 | 25 | 25 | 126 of 133 |
| goalkicker | en | 66 | 9 | 1 / 45 / 50 | 108 | 74 | 10 | 10 | 5 | 8 | 8 | 74 of 80 |
| erickson-algorithms | en | 50 | 29 | 12 / 8 / 6 | 78 | 50 | 50 | 50 | 49 | 49 | 49 | 30 of 51 |
| ostep | en | 19 | 4 | 1 / 3 / 13 | 12 | 8 | 5 | 5 | 1 | 1 | 1 | 18 of 19 |
| think-python-2e | en | 8 | 3 | 4 / 3 / 5 | 14 | 11 | 6 | 6 | 1 | 1 | 1 | 8 of 9 |
| eloquent-javascript | en | 12 | 3 | 2 / 9 / 2 | 24 | 1 | 0 | 0 | 0 | 0 | 0 | 13 of 13 |
| van-steen-distributed-systems-4e | en | 56 | 42 | 4 / 3 / 5 | 44 | 22 | 20 | 20 | 8 | 8 | 11 | 49 of 57 |
| van-steen-graph-theory | en | 38 | 20 | 12 / 1 / 3 | 45 | 32 | 29 | 29 | 29 | 29 | 29 | 24 of 39 |
| coulouris-distributed-systems-5e | en | 44 | 24 | 8 / 9 / 8 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 45 of 45 |
| security-engineering-3e | en | 21 | 10 | 10 / 5 / 1 | 10 | 9 | 9 | 9 | 9 | 9 | 9 | 15 of 21 |
| van-steen-computer-network-organization | en | 11 | 1 | 18 / 7 / 1 | 18 | 18 | 18 | 18 | 15 | 15 | 15 | 4 of 11 |
| data-oriented-design | en | 11 | 6 | 5 / 1 / 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 of 11 |
| kafka-the-definitive-guide-2e | en | 43 | 29 | 1 / 9 / 4 | 42 | 34 | 7 | 7 | 6 | 6 | 6 | 40 of 44 |
| object-pascal-handbook | en | 36 | 20 | 6 / 5 / 0 | 8 | 2 | 2 | 2 | 2 | 2 | 2 | 35 of 37 |
| redis-in-action | en | 13 | 1 | 4 / 2 / 2 | 27 | 22 | 1 | 1 | 0 | 0 | 0 | 14 of 14 |
| open-data-structures-python | en | 30 | 19 | 9 / 2 / 0 | 44 | 47 | 38 | 38 | 35 | 35 | 35 | 20 of 31 |
| gfs-and-dynamo | en | 31 | 23 | 5 / 2 / 5 | 46 | 39 | 39 | 39 | 36 | 36 | 39 | 20 of 31 |
| readings-in-database-systems-5e | en | 2 | 1 | 1 / 0 / 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 of 3 |
| the-linux-command-line | en | 26 | 23 | 2 / 0 / 1 | 12 | 2 | 1 | 1 | 1 | 1 | 2 | 26 of 28 |
| papers | en | 9 | 6 | 2 / 0 / 1 | 22 | 11 | 10 | 10 | 7 | 7 | 7 | 6 of 10 |
| nginx-cookbook | en | 16 | 6 | 3 / 4 / 5 | 30 | 20 | 1 | 1 | 0 | 0 | 0 | 17 of 17 |
| designing-distributed-systems | en | 10 | 6 | 0 / 2 / 3 | 7 | 4 | 0 | 0 | 0 | 0 | 0 | 11 of 11 |
| designing-event-driven-systems | en | 4 | 3 | 0 / 1 / 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 of 5 |
| unix-4-3bsd-reference | en | 11 | 5 | 3 / 3 / 5 | 15 | 15 | 14 | 4 | 4 | 4 | 4 | 10 of 11 |
| postgresql-internals-18 | ru | 29 | 14 | 4 / 6 / 3 | 17 | 5 | 1 | 1 | 1 | 1 | 1 | 28 of 29 |
| cloud-native-docker-k8s | ru | 16 | 10 | 0 / 4 / 2 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 12 of 16 |
| analiz-dannyh-genai-python | ru | 11 | 2 | 3 / 2 / 2 | 23 | 11 | 1 | 1 | 1 | 1 | 1 | 10 of 11 |
| cloud-native-docker-k8s-scan | ru | 11 | 5 | 2 / 3 / 1 | 34 | 28 | 20 | 19 | 19 | 19 | 19 | 5 of 11 |
| analiz-dannyh-genai-python-scan | ru | 11 | 3 | 6 / 2 / 0 | 36 | 25 | 17 | 7 | 7 | 7 | 7 | 8 of 11 |

| scope | pages | clean pages | clean share | ch0 | ch1 | ch2 | ch3 | ch4 | ch5 | ch6 | clean pages, run 6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| whole set | 773 | 347 | 44.9% | 1120 | 879 | 341 | 320 | 266 | 269 | 276 | 687 of 809 (84.9%) |
| English half | 695 | 313 | 45.0% | 1005 | 805 | 297 | 287 | 233 | 236 | 243 | 624 of 731 (85.4%) |
| Russian half | 78 | 34 | 43.6% | 115 | 74 | 44 | 33 | 33 | 33 | 33 | 63 of 78 (80.8%) |

The checker columns, ch0 to ch6, need no reviewer. A checker (its second version, counting page by page) compares each chapter's markdown with the PDF's own text layer (the text copy's layer for a scan copy) and counts the defects it can see by itself, listed below; words missing from the layer altogether and image placeholders are left out of the sum, as too noisy to count as defects. ch0 is run 0, the defaults of the night (jobs 3557, 3558); ch1 is run 1, today's first fixes (3555, 3556); ch2 is run 2, with underscores unescaped outside code and the reread decided on the raw layer again (3559, 3560); ch3 is run 3, where a picture's placeholder or inlined base64 becomes its address with its caption and the MinerU path decodes entities (3565, 3566); ch4 is run 4, where a word the converter split with a space (a ligature, a first letter apart) is joined when the layer has it whole (3571, 3572); ch5 is run 5, where a code row is given again until every glyph of it is drawn, a pipe inside a table cell stays `\|`, only whole entities are decoded and a hyphen inside a line is left alone (3587, 3588); ch6 is run 6, where a spaced hyphen is joined only between letters with one half no layer word, a caption's brackets are escaped in its picture address and a code block made only of rows drawn earlier is dropped (3605, 3606). All of them read the same chapter pages. A converter repeats byte for byte, so every change between the columns is the work of the code.

| check | what it counts | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| glued | two words glued where the layer has a dash between them | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| split | one layer word split by a space, one half not a layer word | 108 | 95 | 90 | 90 | 39 | 39 | 42 |
| broken | one layer word broken as `foo- bar` | 20 | 0 | 0 | 0 | 0 | 0 | 2 |
| run_together | a word of 10+ letters made of layer words with the spaces lost | 10 | 16 | 10 | 10 | 10 | 10 | 10 |
| wrong_char | a word of 4+ letters one letter off a layer word | 26 | 26 | 26 | 26 | 24 | 24 | 24 |
| entities | HTML entities left undecoded in prose | 189 | 10 | 10 | 0 | 0 | 0 | 0 |
| lone_pipes | a line holding a lone pipe | 13 | 0 | 0 | 0 | 0 | 0 | 0 |
| escapes | a backslash before ASCII punctuation in prose | 564 | 544 | 24 | 24 | 24 | 27 | 29 |
| private_use | private-use glyphs | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| formulas | formula placeholders | 137 | 136 | 137 | 137 | 137 | 137 | 137 |
| inline_pictures | pictures inlined as base64 | 11 | 11 | 11 | 0 | 0 | 0 | 0 |
| html_tags | wrapper tags (`span`, `sup`, any tag with `class=` or `style=`) in prose | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| text_lost | pages with under 90% of their layer words in the markdown | 34 | 37 | 29 | 29 | 28 | 28 | 28 |
| **all** | | 1120 | 879 | 341 | 320 | 266 | 269 | 276 |

From 5 to 6 three sources move, none by a text defect but one. GFS goes from 36 to 39 in `split`: its source knob turns joining split words off, because its own layer glues words after `k` and the join was false there. van Steen's Distributed Systems goes from 8 to 11: two are caption brackets now written `\[2010\]` as the page prints them, which the checker takes for escapes, and one is `N - 2`, no longer glued into `N2`. The Linux Command Line goes from 1 to 2, and this one is a loser: `user_name` broken over a line comes out `user_- name`, because the join of a broken word now wants both halves to be letters and `user_` is not. On the gold cuts the four sets read as in run 4 at 98, 95 and 90 (the Russian cuts gain one at 95); three cuts are lower than in run 4: `postgres_en_table_03` (-1.04) and `postgres_en_code_04` (-0.08), whose psql output tables now write a pipe in a cell as `\|`, and `python_en_code_00` (-0.05), whose index gets back the row `--check-hash-based-pycs` the row rule used to drop.

From 4 to 5 GoalKicker goes from 5 to 8, all three in `escapes`: the pipes `\|` and `\|\|` inside the cells of its Ruby operator table, which is how markdown writes a pipe in a cell and not a defect; the checker's next version should leave table rows out of that count. From 0 to 1 one source got worse, open-data-structures-python (44 to 47): with the layer's hyphenated words joined, its layer F1 rose past 0.95 and the second reading was no longer taken; the default reading loses spaces and splits ligatures. Run 2 decides the reread on the raw layer the floor was set on, and the three documents that had switched (with van Steen's graph theory book and one paper) switched back, 38. No source is worse in ch2 than in ch0 or ch1. On the gold cuts the same run moved every cut up but one: `postgres_ru_table_04` fell from 99.77 to 97.96, a cut whose raw layer F1 sits at 0.9495 against the 0.95 floor and now takes the second reading, cleaner in prose and worse in its table (a wrapped cell read as a second row). Joining split words also joins about 19 formula indices the layer holds glued (`K n`, `L j`), which leaves them as the layer has them; one join is false (`ask for` in the GFS paper, whose own layer glues words after `k`). One gold cut, `haskell_kholomiev_ru_prose_03`, moves in every run with no rule firing (93.6, 96.1, 96.6, 94.1): Docling does not repeat itself on it, so its moves are not read as losses.

## The route against Docling's defaults on the gold

The same 111 gold cuts of the converter entry, read by Docling `default` (runs `docling_default_*_cuts_v720`) and by the intake route on run 6 (`intake_*_cuts_gate6b`), scored by the same scorer:

| cuts | recovered at 98 / 95 / 90 | similarity | headings exact | code exact | table cells |
|---|---|---|---|---|---|
| en, 54 | 17 / 25 / 34 → 17 / 28 / 35 | 85.2 → 85.7 | 115 → 114 of 151 | 11 → 24 of 32 | 154 → 157 of 401 |
| ru, 38 | 6 / 14 / 23 → 11 / 21 / 28 | 87.3 → 88.5 | 58 → 62 of 66 | 1 → 7 of 17 | 25 → 25 of 56 |
| Postgres Pro ru, 14 | 12 / 14 / 14 → 12 / 14 / 14 | 99.4 → 99.5 | 28 → 27 of 46 | 6 → 11 of 20 | 112 → 105 of 218 |
| ARES en, 5 | 1 / 2 / 3 → 2 / 4 / 4 | 88.5 → 93.3 | 3 → 3 of 12 | none | 131 → 131 of 162 |

The route gains most on code (exact blocks 18 → 42 of 69 over the four sets) and on Russian prose. It loses on two columns: Postgres Pro's table cells (the second reading of `postgres_ru_table_03` and `_04`, whose raw layer sits at the 0.95 floor, wraps a cell into a second row) and one English heading.

## Third chapters

The third chapter of each document (`datasets/converter_gold/chapter_set_3.yaml`, 699 pages) was never used for tuning. It was read once on the defaults of the night ("before", jobs 3583, 3584) and once on the code of run 6 ("after", 3607, 3608), and counted by the same checker:

| source | before | after |
|---|---|---|
| ostep | 7 | 2 |
| goalkicker | 171 | 13 |
| open-data-structures-python | 46 | 31 |
| erickson-algorithms | 41 | 20 |
| think-python-2e | 41 | 0 |
| eloquent-javascript | 5 | 0 |
| van-steen-distributed-systems-4e | 10 | 3 |
| van-steen-graph-theory | 16 | 7 |
| van-steen-computer-network-organization | 26 | 8 |
| coulouris-distributed-systems-5e | 1003 | 517 |
| security-engineering-3e | 5 | 1 |
| the-linux-command-line | 14 | 2 |
| readings-in-database-systems-5e | 1 | 1 |
| data-oriented-design | 0 | 0 |
| kafka-the-definitive-guide-2e | 18 | 7 |
| designing-event-driven-systems | 0 | 0 |
| redis-in-action | 50 | 7 |
| nginx-cookbook | 112 | 0 |
| designing-distributed-systems | 6 | 0 |
| object-pascal-handbook | 2 | 0 |
| freepascal | 122 | 6 |
| unix-4-3bsd-reference | 15 | 6 |
| papers | 2 | 2 |
| postgresql-internals-18 | 38 | 4 |
| cloud-native-docker-k8s | 9 | 7 |
| analiz-dannyh-genai-python | 43 | 0 |
| cloud-native-docker-k8s-scan | 14 | 8 |
| analiz-dannyh-genai-python-scan | 57 | 1 |
| **all** | 1874 | 653 |

No source is worse. Coulouris keeps most of what is left: 394 split words (`m obility`, `forwardin g`) and 100 one-letter-off words with the look of OCR errors (`sinele`, `nossible`, `entrv`), which its second chapter does not show; why this chapter reads that way is not yet known.

The review of one document by two agents differed by 4 of 56 pages (van Steen, Distributed Systems: 42 and 46 clean); read a single source's count with that spread in mind.

## Clean pages and reviewers on run 6

The column "clean pages, run 6" counts the pages of run 6 where the checker finds no defect of its kinds; it reads the chapter set as it stands, one page longer than the reviewers' range, and it cannot see what only a reader sees (a lost heading, a broken table), so it is no match for the reviewers' clean pages of the night. One reviewer per group of sources also read run 0 against run 6 by the diff, every changed place checked against the PDF (six Sonnet reviewers, reports in `~/working_docs/projects/rag-lab/notes/reference/step0/run6/`): no source got worse on the whole, and four places did. Three are one mechanism: a compound whose hyphen falls at a line end (`problem-oriented`, `receive-omission`, `WordPress-based`) is marked by PDFium as a hyphenation break, so the layer the word rules read holds it glued and the join of a broken word takes the glued form (3 of its 20 firings on the chapters; the other 17 are right). They are kept as named losers: fixing them is polish the corpus does not need now. The fourth is a caption Docling reads with two words glued (`G1from`), shown now that the picture has an address.

Clean share per source, over the 29 sources:

| | min | max | mean | median |
|---|---|---|---|---|
| before: reviewers, the night's run | 7.7% (Redis in Action) | 88.5% (The Linux Command Line) | 46.7% | 50.0% |
| after: checker, run 6 | 36.4% (van Steen, Computer Networks) | 100% (8 sources) | 83.6% | 90.9% |

## Decision

The threshold for the first merge request is set in numbers the stand computes, not in the clean share: similarity of at least 98% on the gold cuts with no loser caused by our own rules, and on the chapter set the mechanical count of each planned kind brought to zero or to a declared tolerance. Agents review only the pages a fix changed. Tuning runs only on the second chapters; the third chapters and the whole books come after, and two held-back books are read once at the end.

## Caveats

- The chapter runs end one page earlier than the chapter set does now (their jobs were built before the chapter end was moved to include the next chapter's first page); the reviewed pages are exactly the converted ones.
- Mapping about 425 defect rows to kinds was done by one agent reading the reviews; spot-check it before a kind's rank decides a fix.
- Two books were reviewed by a stronger model than the rest; their defect counts per page may be higher for that reason alone.
- The reviewers read the night's run (jobs 3505 and 3541), the checker read runs 0 and 1 made later on the chapter set as it stands; the two kinds of column are not comparable with each other.
- Every checker column is counted by the checker's second version; a later version recounts every column rather than adding one beside numbers of another version.
