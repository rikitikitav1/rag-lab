# 2026-09-25 - Which converter turns each kind of document into markdown

The corpus of the next arc comes from books as PDF, from scanned pages and from documentation sites. Each
of these has to become plain markdown before the stand can cut it into chunks. This entry records which
of three open converters does that best on each kind of input, measured against a gold made from each
document's own source. The text layer was measured on 25 September and again on 27 September, after the
first run turned out to sit on a broken release of Docling's PDF parser; the numbers below are the second
run's for the text layer and the first run's for scans and HTML.

## What was declared before the runs

The yardstick is the owner's: raw material good enough for retrieval, not a byte-exact copy. A section
counts as well parsed when its plain text is at least 98 alike to the gold (rapidfuzz ratio, 0-100), and
the output must be cut by the stand's own chunker without breaking its gates.

**The bar.** Written on 24.09, before the second and third tools ran, and declared in
`datasets/converter_gold/manifest.yaml` (`score.bar`). A tool is chosen for an input kind when:

1. it is no worse than the others on the share of outputs the stand can cut;
2. it is no worse than the others on headings found;
3. it is better than the others on at least one of: code blocks kept whole, table cells, body
   similarity.

If no tool meets all three, the tools are indistinguishable for that input kind. That counts as a
result, and speed then decides.

**Change to the bar, 25.09.** In rule 2, "headings found" replaced "headings at the exact level". One
tool sets every heading at one level, and the level is kept as a separate, descriptive column. The
change was made before any verdict turned on that rule, so no verdict was taken under the old wording.

**The floor.** Every difference is read against a repeat of the same tool on the same inputs.

## Setup

**Tools** Docling (docling-serve 1.35.0, docling 2.130.0, docling-parse 7.20.0), MinerU 4.0.7, Marker 2.0.0,
each in its own image with its models checked by hash at build and no network at run time.

**Gold** `datasets/converter_gold/`. Seven PDF documents with a public source (Pro Git en and ru,
PostgreSQL en, Python, Django, Baldin's LaTeX book, Kholomiev's Haskell book) and the Postgres Pro ru
translation, whose gold is its own site, a converter's reading and named as such. Fifteen sections a
document, drawn by seed in three kinds (prose, code, table). The gold markdown is made by pandoc from
the section's own source. ARES, a two-column paper, adds 5 sections with its arXiv HTML as gold.

**Populations**
- **Text-layer PDF.** The drawn sections cut out of the book as short PDFs of whole pages: en 54 sections
  over four books, ru 38 over three, Postgres Pro ru 14, ARES 5.
- **Probe.** Seven pieces of real books without gold, 125 pages (OSTEP chapters, the GFS and Dynamo papers, a
  Python notes book), compared with each PDF's own text layer.
- **Raster.** The Russian sections rendered as images at 300 dpi ("baseline"), and at a "scanner"
  point (200 dpi, JPEG quality 75, 1° skew, noise sigma 5), 38 sections each.
- **HTML.** 115 site pages drawn from the source files of Pro Git (git-scm.com), kubernetes, fastapi, vue,
  and Postgres Pro: en 50, ru 65. Each page is paired with its file by status and title.

**Arms**
- Docling, three settings files under `converters/docling/settings/`: `default` (the docling_parse backend, no code
  enrichment), `pypdfium2` (the pypdfium2 backend) and `code_enrichment`. Tesseract OCR (`rus`, `eng`) on raster.
- MinerU with `ocr_mode: txt` on text-layer PDFs and with forced OCR on raster.
- Marker in its balanced mode.

Scoring is `scripts/converter_gold.py` (`score`, `gates`). Every similarity cell is read on the same pairs:
sections found by both arms, not overrun in either, and not doubtful in either. Paired deltas use the stand's
bootstrap (10000 draws, seed 42). Jobs of the text-layer run: `convert_source` 3429 to 3443, runs `*_v720`.

## Result

### Text layer

Gold sections recovered at similarity 98 / 95 / 90:

| cuts | Docling default | Docling pypdfium2 | Docling code enrichment | MinerU |
|---|---|---|---|---|
| en, 54 | 17 / 25 / 34 | 13 / 24 / 34 | 17 / 23 / 32 | 16 / 25 / 30 |
| ru, 38 | 6 / 14 / 23 | 11 / 19 / 26 | 2 / 13 / 20 | 13 / 22 / 28 |
| Postgres Pro ru, 14 | 12 / 14 / 14 | 12 / 14 / 14 | 9 / 12 / 13 | 10 / 10 / 10 |
| ARES en, 5 | 1 / 2 / 3 | 1 / 3 / 4 | 1 / 2 / 3 | 3 / 3 / 4 |

The same rows split by the kind each cut is labelled with:

| cuts | kind | Docling default r98/95/90 | pypdfium2 | MinerU | headings exact, default / MinerU |
|---|---|---|---|---|---|
| en | table, 14 | 4 / 6 / 8, cells 153 | 2 / 5 / 8, cells 141 | 3 / 7 / 7, cells 103 | 34 / 14 of 43 |
| en | code, 20 | 2 / 6 / 12 | 2 / 6 / 12 | 2 / 5 / 9 | 29 / 17 of 53 |
| en | prose, 20 | 11 / 13 / 14 | 9 / 13 / 14 | 11 / 13 / 14 | 52 / 18 of 55 |
| ru | prose, 15 | 2 / 5 / 8 | 5 / 8 / 9 | 8 / 11 / 11 | 23 / 14 of 24 |
| Postgres Pro ru | code, 4 | 4 / 4 / 4 | 4 / 4 / 4 | 1 / 1 / 1 | |

Table cells recovered: en 154 of 401 (Docling default and code enrichment), 141 (pypdfium2), 103 of 386 (MinerU); ARES
131 of 162, 7, 152. Gold code blocks matched exactly / after collapsing whitespace: en 11 / 21 (default), 10 / 19
(pypdfium2), 0 / 11 (code enrichment), 12 / 13 (MinerU); Postgres Pro ru 6 / 16, 5 / 15, 0 / 4, 5 / 6.

Word F1 of real books against their own text layers: GFS 0.937 default, 0.919 pypdfium2, 0.937 MinerU; Python notes
0.934, 0.974, 0.936; OSTEP `threads-sema` 0.951, 0.955, 0.940. Each Docling run took about 8 minutes for all five
populations (405 pages).

### Raster and HTML

| Input kind | n | Docling | MinerU | MinerU minus Docling | Winner |
|---|---|---|---|---|---|
| Raster ru, baseline (38) | 34 | 82.8, at 98: 0; cut 38/38; 378 s | 89.8, at 98: 12; cut 38/38; 248 s | +6.9 [+5.4, +8.5] | MinerU |
| Raster ru, scanner (38) | 33 | 83.0, at 98: 0; cut 37/38; 492 s | 88.4, at 98: 11; cut 38/38; 171 s | +5.5 [+3.6, +7.4] | MinerU |
| HTML, en / ru (50 / 65 pages) | 50 / 64 | 97.1 / 97.9, at 98: 30 / 45; cut 45/50 and 59/64 | not run: it does not read HTML | | Docling |

On raster MinerU also finds 60 of 65 headings against Docling's 47 of 66, and keeps 12 of 17 code blocks whole
against 1. Words mixing Latin and Cyrillic letters: MinerU 10-12, Docling 12-18, the gold 3.

**Floors.** Docling repeats byte for byte on the text layer and the raster. MinerU repeats with at most 0.01 of
similarity moved on the text layer, and at most 0.19 on the raster.

**Marker** fails the bar on the text layer before speed is read: on en cuts 42/54 cut against Docling's 51, headings
found 87 of 147 against 116, and 0 of 27 code blocks kept whole. On raster it was stopped after 9 of 38 sections in 36
minutes, about 2.4 hours for the whole point. It was taken out of the stand.

## What each setting does to a page

- **Without code enrichment, Docling puts every code block on one line** (en cuts 162 of 162 fenced blocks; Python
  notes 146 of 146). The content is there: after whitespace is collapsed it matches 21 of 32 English blocks, against 13
  for MinerU.
- **Code enrichment restores line breaks but reads Russian comments in code as Latin letters**: Postgres Pro ru code
  matches fall from 6 / 16 to 0 / 4. Granite-Docling (`pipeline: vlm`) showed the same in a three-page smoke run.
- **pypdfium2 does best on Russian prose among the Docling settings, but misses most ARES table cells** (7 of 162
  against 131). The Russian prose gap comes from one book, Baldin's LaTeX.
- **On book pages Docling wins or ties every kind, and MinerU loses headings on every kind**, which the chunker cuts on.

## What a site page needs before any converter

HTML was read twice: once as served, and once reduced to the site's own text. The reduction keeps the
element a per-site setting names, drops the site's furniture inside it (child-page lists, feedback
widgets, a translation banner, variant code blocks), and flattens syntax highlighting in `<pre>`. As
served, a kubernetes page is mostly its navigation tree: similarity 12.7 en and 16.7 ru, and every page
broken for the chunker. Reduced, it reads 97.3 and 99.3. Highlighted code on vue came out with a space
between every token (`console. log (`), 2 of 96 blocks kept whole; flattened, 96 of 96. This is a
property of the input, not a choice between converters: the corpus pipeline owns the reduction.

## Decision

- **A route by file** (owner, 27 September): a PDF with a text layer and an HTML page go to Docling `default`, a scan or
  image goes to MinerU with forced OCR. The kind is read from the input itself (a text layer present or not).
- Code blocks are rebuilt from the PDF's own text layer inside Docling's code items: exact code blocks rise from 11 to 17
  of 32 on en, 1 to 5 of 17 on ru, 6 to 8 of 20 on Postgres Pro ru (MinerU 12, 5, 5), and Cyrillic comments survive.
- A piece that agrees badly with its own text layer is read again with the `pypdfium2` backend.
- Piece boundaries must not split code blocks or tables; joined pieces close an open fence or table header.
- Marker is removed from the stand. Its readings stay in the arc log and its run folders.
- A site source declares where its text is, what inside it is furniture, and which pages the site builds itself.
- The Dockerfile pins docling-parse 7.20.0 and checks the version during the build.

## Caveats

- The first text-layer run, on 25 September, sat on docling-parse 7.21.0, which drops spaces between words
  (docling-parse issue 355): GFS word F1 0.695 against 0.937, ARES similarity 67.2 against 88.5. It named Docling the
  text-layer winner outright; on 7.20.0, with code shape and table cells counted, the route became per file.
- A per-page map first read from whole populations put tables on MinerU on the strength of one two-column paper
  (ARES); split by the kind each cut is labelled with, it did not hold (found by the auditor, round 130).
- The raster was read on Russian only. English raster pages are rendered and have gold but were not run.
- The text-layer sections are short cuts of whole pages without the book's outline. On whole books the
  heading levels read differently.
- The Postgres Pro gold is pandoc's reading of its own site, so its numbers compare a converter with a
  converter.
- Code shape counts fenced blocks rather than comparing line breaks with the gold.
- Every number was rescored after gold and scorer defects were fixed (a comment line in code read as a heading, pipes
  in code read as table cells, a `#` inside a fence ending a section); no winner changed.
- The route by file on whole books is read in the step 0 entry (2026-09-28).
