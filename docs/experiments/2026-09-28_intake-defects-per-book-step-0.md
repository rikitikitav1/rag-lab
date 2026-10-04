# 2026-09-28 - Intake defects per book: the step 0 baseline

Before fixing the intake any further, every book in the store was read on the current defaults and the second chapter of each book was checked by hand, page by page, against the page image and the PDF's own text. A first look at 5 to 10 pages of each whole book started the list of defect kinds; as a per-book measure it was noise, so it is not reported here. This entry keeps, per book, the number of pages that came out clean on the second chapter. Each later fix adds its own column on the same chapters, so a book that gets better or worse stays visible, instead of an average hiding it.

## Setup

**Books** 25 from the store plus 2 Russian books with a text layer and their image-only copies · **defaults** of 27 September 2026 on branch `intake` (docling-parse 7.20.0, code from the layer, fences for code with no box, joins over page breaks, outline levels, the second reading under layer F1 0.95; row rules, seam shifting and HTML title levels off) · **chapters** the chapter set `datasets/converter_gold/chapter_set.yaml` (the second chapter of each PDF document, whole; 45 documents plus the 4 Russian sources), runs `chapter_set_en_20260927f` (job 3505) and `chapter_set_ru_step0base_off` (job 3541, today's fixes switched off by the run's knobs).

Reviewers: Claude agents, two on FreePascal and GoalKicker (Opus), the rest Sonnet, each going page by page against the PDF's text layer and a render where the layer could not settle it, with the instruction in `~/working_docs/projects/rag-lab/notes/reference/step0/CHAPTER_INSTRUCTION.md`; the merged list of defect kinds is in `CHAPTERS_SUMMARY.md` in the same folder. Where two reviews of one document differ, the more pessimistic is kept.

## Result

pages, clean and clean ch12 count pages; ch0 to ch12 count the checker's defects. A page is clean when its review names no defect on it (clean: the reviewers on the night's run) or when the checker finds none of its kinds on it (clean ch12: run 13, recounted only where ch12 differs from ch11). pages is the chapter as run 10 reads it; the reviewers read the chapters as they ended on the night, some of them shorter (see Caveats).

<details>
<summary>Per source: pages, clean pages and the checker's defects by chapter</summary>

| source | lang | pages | clean | clean ch12 | ch0 | ch1 | ch2 | ch3 | ch4 | ch5 | ch6 | ch7 | ch8 | ch9 | ch10 | ch11 | ch12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| freepascal | en | 133 | 17 | 132 | 397 | 384 | 37 | 37 | 25 | 25 | 25 | 2 | 2 | 2 | 2 | 2 | 2 |
| goalkicker | en | 80 | 9 | 75 | 108 | 74 | 10 | 10 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| erickson-algorithms | en | 51 | 29 | 46 | 78 | 50 | 50 | 50 | 49 | 49 | 49 | 9 | 9 | 9 | 9 | 9 | 9 |
| ostep | en | 19 | 4 | 18 | 12 | 8 | 5 | 5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| think-python-2e | en | 9 | 3 | 8 | 14 | 11 | 6 | 6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| eloquent-javascript | en | 13 | 3 | 13 | 24 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| van-steen-distributed-systems-4e | en | 57 | 42 | 51 | 44 | 22 | 20 | 20 | 8 | 8 | 11 | 7 | 7 | 7 | 7 | 7 | 7 |
| van-steen-graph-theory | en | 39 | 20 | 36 | 45 | 32 | 29 | 29 | 29 | 29 | 29 | 5 | 5 | 5 | 5 | 5 | 4 |
| coulouris-distributed-systems-5e | en | 45 | 24 | 45 | 8 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| security-engineering-3e | en | 21 | 10 | 21 | 10 | 9 | 9 | 9 | 9 | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| van-steen-computer-network-organization | en | 11 | 1 | 6 | 19 | 19 | 19 | 19 | 16 | 16 | 16 | 6 | 6 | 5 | 8 | 6 | 5 |
| data-oriented-design | en | 11 | 6 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kafka-the-definitive-guide-2e | en | 44 | 29 | 40 | 42 | 34 | 7 | 7 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| object-pascal-handbook | en | 37 | 20 | 35 | 17 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 2 | 2 | 2 | 2 |
| redis-in-action | en | 14 | 1 | 14 | 31 | 26 | 5 | 5 | 4 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| open-data-structures-python | en | 31 | 19 | 25 | 44 | 47 | 38 | 38 | 35 | 35 | 35 | 8 | 8 | 8 | 8 | 8 | 8 |
| gfs-and-dynamo | en | 31 | 23 | 31 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| readings-in-database-systems-5e | en | 3 | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| the-linux-command-line | en | 28 | 23 | 26 | 12 | 2 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| papers | en | 10 | 6 | 7 | 22 | 11 | 10 | 10 | 7 | 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 |
| nginx-cookbook | en | 17 | 6 | 17 | 30 | 20 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| designing-distributed-systems | en | 11 | 6 | 11 | 7 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| designing-event-driven-systems | en | 5 | 3 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unix-4-3bsd-reference | en | 11 | 5 | 10 | 15 | 15 | 14 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| postgresql-internals-18 | ru | 29 | 14 | 28 | 17 | 5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| cloud-native-docker-k8s | ru | 16 | 10 | 12 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| analiz-dannyh-genai-python | ru | 11 | 2 | 10 | 23 | 11 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| cloud-native-docker-k8s-scan | ru | 11 | 5 | 5 | 35 | 29 | 21 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| analiz-dannyh-genai-python-scan | ru | 11 | 3 | 7 | 38 | 27 | 19 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 |

</details>

| scope | pages | clean | clean ch12 | clean ch12 share | ch0 | ch1 | ch2 | ch3 | ch4 | ch5 | ch6 | ch7 | ch8 | ch9 | ch10 | ch11 | ch12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whole set | 809 | 347 | 748 | 92.5% | 1104 | 863 | 325 | 304 | 253 | 253 | 257 | 117 | 107 | 97 | 100 | 98 | 96 |
| English half | 731 | 313 | 686 | 93.8% | 986 | 786 | 278 | 268 | 217 | 217 | 221 | 81 | 71 | 61 | 64 | 62 | 60 |
| Russian half | 78 | 34 | 62 | 79.5% | 118 | 77 | 47 | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 |

The checker columns, ch0 to ch12, need no reviewer. A checker (its fourth version, counting page by page) compares each chapter's markdown with the PDF's own text layer (the text copy's layer for a scan copy) and counts the defects it can see by itself, listed below; words missing from the layer altogether and image placeholders are left out of the sum, as too noisy to count as defects. ch0 is run 0, the defaults of the night (jobs 3557, 3558); ch1 is run 1, today's first fixes (3555, 3556); ch2 is run 2, with underscores unescaped outside code and the reread decided on the raw layer again (3559, 3560); ch3 is run 3, where a picture's placeholder or inlined base64 becomes its address with its caption and the MinerU path decodes entities (3565, 3566); ch4 is run 4, where a word the converter split with a space (a ligature, a first letter apart) is joined when the layer has it whole (3571, 3572); ch5 is run 5, where a code row is given again until every glyph of it is drawn, a pipe inside a table cell stays `\|`, only whole entities are decoded and a hyphen inside a line is left alone (3587, 3588); ch6 is run 6, where a spaced hyphen is joined only between letters with one half no layer word, a caption's brackets are escaped in its picture address and a code block made only of rows drawn earlier is dropped (3605, 3606); ch7 is run 8, where a formula Docling could not decode gets the text the layer holds under it (3623, 3624); ch8 is run 9, where a figure, listing or table caption made a heading goes back to a line (3629, 3630); ch9 is run 10, where a running head made a heading is dropped from its second time on (3635, 3636); ch10 is run 11, where a second reading refused for losing table cells alone keeps its prose and takes the first reading's tables, matched by page, where they keep more cells (3856, 3857); ch11 is run 12, where the same splice also takes back a formula whose operators the second reading lost, (3865, 3866; van Steen's computer networks and Coulouris in 3862, van Steen's read with the splice on, 6 against 5 on its first reading, which is why it now turns the splice off as its own knob). ch12 is run 13, the fixes of the second review: a heading takes its level from the file's whole outline and one the outline lacks moves down with the outline's heading above it, never up; a running head is read over the whole file and has at least two words; a word hyphenated over a page break is joined; the formula splice is fixed (3888 for the English chapters, 3874 for the Russian; the checker's eighth version, where a one-word line is never a running head). All of them read the same chapter pages. ch12 joins two runs: the English chapters from 3888, the Russian from 3874, whose markdown did not move in the later rounds. The layout of a page repeats run to run; one glyph of the Rails notes and one table of Coulouris do not (see run 13), so a change between the columns is the work of the code everywhere else.

| check | what it counts | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glued | two words glued where the layer has a dash between them | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| split | one layer word split by a space, one half not a layer word | 81 | 68 | 63 | 63 | 15 | 15 | 15 | 18 | 18 | 18 | 21 |
| broken | one layer word broken as `foo- bar` | 20 | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 2 | 2 | 6 |
| run_together | a word of 10+ letters made of layer words with the spaces lost | 10 | 16 | 10 | 10 | 10 | 10 | 10 | 13 | 13 | 13 | 11 |
| wrong_char | a word of 4+ letters one letter off a layer word | 14 | 14 | 14 | 14 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| entities | HTML entities left undecoded in prose | 189 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lone_pipes | a line holding a lone pipe | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| escapes | a backslash before ASCII punctuation in prose | 564 | 544 | 24 | 24 | 24 | 24 | 26 | 26 | 26 | 26 | 26 |
| private_use | private-use glyphs | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| formulas | formula placeholders | 137 | 136 | 137 | 137 | 137 | 137 | 137 | 0 | 0 | 0 | 0 |
| inline_pictures | pictures inlined as base64 | 11 | 11 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| html_tags | wrapper tags (`span`, `sup`, any tag with `class=` or `style=`) in prose | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| text_lost | pages with under 90% of their layer words in the markdown | 34 | 37 | 29 | 29 | 28 | 28 | 28 | 19 | 19 | 19 | 17 |
| caption_headings | a heading that is a figure, listing or table caption | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 0 | 0 | 0 |
| running_headings | a heading that is the book's running head, from its second time on | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 3 | 3 |
| **all** | | 1104 | 863 | 325 | 304 | 253 | 253 | 257 | 117 | 107 | 97 | 100 |

Run 11 fires on one chapter only, van Steen's computer networks (5 to 8), and is a loser there; the chapters of the books it was made for hold no refused second reading. Its effect is on the pieces whose second reading was refused for table cells alone (jobs 3852 to 3855, each piece read with the switch off and on): it fired on five, all better by the checker, Coulouris pages 751-800 from 738 to 42 defects with table cells 215 to 236 and text lost 18 to 0, pages 801-850 from 982 to 25, the SQL Server notes 101-150 from 12 to 6, van Steen's graph theory 95-144 from 20 to 11; 1944 to 276 over the thirteen pieces. The gold cuts do not move. As one source loses, the switch stays a source's knob, off by default.

Run 13 moves two sources, both down: van Steen's graph theory 5 to 4 (a formula whose operators came back) and van Steen's computer networks 6 to 5; its clean pages go 7 to 6, the defects left sit on more pages. The run took three rounds. The first read the running head over the whole book, so a one-word heading at the top of three pages went as one: nginx lost its 23 `Solution` and `Discussion` headings and five chapters their `Chapter N` label; a running head now has two words, and the one-word `Computers` of van Steen's computer networks is back as it was in run 12. The second moved a heading the outline lacks with the outline's heading above it both ways, and the SQL Server notes and Readings in Database Systems had a sub-heading raised to a chapter's level; it now moves down only, which keeps Coulouris's `17.4.1` under `17.4`. The gold cuts do not move in any round: they have no outline, so they cannot judge the levels at all. Two things change between runs of the same code with no rule firing: one glyph of the Rails notes (`e` and `¢` in turn) and one table of Coulouris, read whole in one run and in scraps in the next.

From 5 to 6 two sources count more, and one of them holds a loser: The Linux Command Line (1 to 2) breaks `user_name` over a line into `user_- name`, since both halves are words of its layer and the join of a broken word needs one that is not. van Steen's Distributed Systems goes from 8 to 11: two are caption brackets written `\[2010\]` as the page prints them, which this checker still takes for escapes, and one is `N - 2`, no longer glued into `N2`. From 6 to 7 formulas go from 137 to 0 and pages with text lost from 27 to 18, on the eight sources that have formulas; from 7 to 8 the ten caption headings (Coulouris, Redis) and from 8 to 9 ten of the thirteen running heads (Object Pascal, van Steen's computer networks) leave; the three left are in the two scans, which have no layer for the running-head rule to read. On the gold cuts the sets read as in run 4 at 98, 95 and 90 (the Russian cuts gain one at 95); three cuts are lower than in run 4: `postgres_en_table_03` (-1.04) and `postgres_en_code_04` (-0.08), whose psql output tables write a pipe in a cell as `\|`, and `python_en_code_00` (-0.05), whose index gets back the row `--check-hash-based-pycs` the row rule used to drop; runs 8 to 10 move no cut down.

From 0 to 1 one source got worse, open-data-structures-python (44 to 47): with the layer's hyphenated words joined, its layer F1 rose past 0.95 and the second reading was no longer taken; the default reading loses spaces and splits ligatures. Run 2 decides the reread on the raw layer the floor was set on, and the three documents that had switched (with van Steen's graph theory book and one paper) switched back, 38. No source is worse in ch2 than in ch0 or ch1. On the gold cuts the same run moved every cut up but one: `postgres_ru_table_04` fell from 99.77 to 97.96, a cut whose raw layer F1 sits at 0.9495 against the 0.95 floor and now takes the second reading, cleaner in prose and worse in its table (a wrapped cell read as a second row). Joining split words also joins about 19 formula indices the layer holds glued (`K n`, `L j`), which leaves them as the layer has them; one join is false (`ask for` in the GFS paper, whose own layer glues words after `k`). One gold cut, `haskell_kholomiev_ru_prose_03`, moved in the early runs with no rule firing (93.6, 96.1, 96.6, 94.1); from run 6 on, and in two runs of the same code, it reads 96.58 byte for byte, so those moves came from the stand of that time (runs before PDFium pages were closed in their own thread), not from Docling.

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

<details>
<summary>Third chapters per source: defects before and after</summary>

| source | before | after |
|---|---|---|
| ostep | 7 | 2 |
| goalkicker | 171 | 10 |
| open-data-structures-python | 46 | 31 |
| erickson-algorithms | 41 | 20 |
| think-python-2e | 41 | 0 |
| eloquent-javascript | 5 | 0 |
| van-steen-distributed-systems-4e | 10 | 3 |
| van-steen-graph-theory | 16 | 7 |
| van-steen-computer-network-organization | 27 | 9 |
| coulouris-distributed-systems-5e | 1020 | 534 |
| security-engineering-3e | 5 | 1 |
| the-linux-command-line | 14 | 2 |
| readings-in-database-systems-5e | 1 | 1 |
| data-oriented-design | 0 | 0 |
| kafka-the-definitive-guide-2e | 18 | 7 |
| designing-event-driven-systems | 0 | 0 |
| redis-in-action | 50 | 2 |
| nginx-cookbook | 118 | 6 |
| designing-distributed-systems | 6 | 0 |
| object-pascal-handbook | 5 | 3 |
| freepascal | 122 | 6 |
| unix-4-3bsd-reference | 15 | 6 |
| papers | 2 | 2 |
| postgresql-internals-18 | 38 | 4 |
| cloud-native-docker-k8s | 9 | 7 |
| analiz-dannyh-genai-python | 46 | 1 |
| cloud-native-docker-k8s-scan | 14 | 8 |
| analiz-dannyh-genai-python-scan | 62 | 6 |
| **all** | 1909 | 678 |

</details>

No source is worse. Coulouris keeps most of what is left (534): its pages 149 to 163 agree with their layer at F1 0.69 and the second reading, at 0.97, was refused for losing two of 44 table cells; a source knob (`reread_cells_slack`) lets that source take it, and is set when Coulouris is taken into the corpus.

The review of one document by two agents differed by 4 of 56 pages (van Steen, Distributed Systems: 42 and 46 clean); read a single source's count with that spread in mind.

## Clean pages and reviewers

The column "clean ch9" counts the pages of run 10 where the checker finds no defect of its kinds; it reads the chapter set as it stands, one page longer than the reviewers' range, and it cannot see what only a reader sees (a lost heading, a broken table), so it is no match for the reviewers' clean pages of the night. One reviewer per group of sources also read run 0 against run 6 by the diff, every changed place checked against the PDF (six Sonnet reviewers, reports in `~/working_docs/projects/rag-lab/notes/reference/step0/run6/`): no source got worse on the whole, and four places did. Three are one mechanism: a compound whose hyphen falls at a line end (`problem-oriented`, `receive-omission`, `WordPress-based`) is marked by PDFium as a hyphenation break, so the layer the word rules read holds it glued and the join of a broken word takes the glued form (3 of its 20 firings on the chapters; the other 17 are right). They are kept as named losers: fixing them is polish the corpus does not need now. The fourth is a caption Docling reads with two words glued (`G1from`), shown now that the picture has an address.

Clean share per source, over the 29 sources:

| | min | max | mean | median |
|---|---|---|---|---|
| reviewers, the night's run | 7.7% (Redis in Action) | 88.5% (The Linux Command Line) | 46.7% | 50.0% |
| checker, run 0 | 17.6% (NGINX Cookbook) | 100% (3 sources) | 61.1% | 61.3% |
| checker, ch9 (run 10) | 45.5% (Docker and Kubernetes, scan) | 100% (10 sources) | 89.4% | 93.8% |

The two checker rows are one ruler, before and after; the reviewers' row is another and is kept for what it saw that the checker cannot.

## Decision

The threshold for the first merge request is set in numbers the stand computes, not in the clean share: similarity of at least 98% on the gold cuts with no loser caused by our own rules, and on the chapter set the mechanical count of each planned kind brought to zero or to a declared tolerance. Agents review only the pages a fix changed. Tuning runs only on the second chapters; the third chapters and the whole books come after, and two held-back books are read once at the end.

## Caveats

- The chapter runs end one page earlier than the chapter set does now (their jobs were built before the chapter end was moved to include the next chapter's first page); the reviewed pages are exactly the converted ones.
- Mapping about 425 defect rows to kinds was done by one agent reading the reviews; spot-check it before a kind's rank decides a fix.
- Two books were reviewed by a stronger model than the rest; their defect counts per page may be higher for that reason alone.
- The reviewers read the night's run (jobs 3505 and 3541), the checker read runs 0 and 1 made later on the chapter set as it stands; the two kinds of column are not comparable with each other.
- Every checker column, and the third chapters, is counted by the checker's fourth version (`temp_files/chapter_check.py`, the stand's module `app/use_cases/layer_check.py` is its fifth); it leaves a pipe escaped in a table cell out of the escapes, does not judge words on GFS, whose layer glues them itself, and counts caption and running-head headings.
