# 2026-09-25 - Which converter turns each kind of document into markdown

The corpus of the next arc comes from books as PDF, from scanned pages and from documentation sites. Each
of these has to become plain markdown before the stand can cut it into chunks. This entry records which
of three open converters does that best on each kind of input, measured against a gold made from each
document's own source. The stand keeps two of them, one per kind of input.

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

**Tools** Docling (docling-serve 1.35.0, docling 2.130.0), MinerU 4.0.7, Marker 2.0.0, each in its own
image with its models checked by hash at build and no network at run time.

**Gold** `datasets/converter_gold/`. Seven PDF documents with a public source (Pro Git en and ru,
PostgreSQL en, Python, Django, Baldin's LaTeX book, Kholomiev's Haskell book) and the Postgres Pro ru
translation, whose gold is its own site, a converter's reading and named as such. Fifteen sections a
document, drawn by seed in three kinds (prose, code, table). The gold markdown is made by pandoc from
the section's own source.

**Populations**
- **Text-layer PDF.** The drawn sections cut out of the book as short PDFs of whole pages: en 54 sections
  over four books, ru 53 over four.
- **Raster.** The same Russian sections rendered as images at 300 dpi ("baseline"), and at a "scanner"
  point (200 dpi, JPEG quality 75, 1° skew, noise sigma 5), 38 sections each.
- **HTML.** 115 site pages drawn from the source files of Pro Git (git-scm.com), kubernetes, fastapi, vue,
  and Postgres Pro: en 50, ru 65. Each page is paired with its file by status and title.

**Arms** Docling's default is Tesseract OCR (`rus`, `eng`), no code enrichment. That default was itself
chosen on these populations over EasyOCR, the pypdfium2 backend and code enrichment. MinerU runs with
`ocr_mode: txt` on text-layer PDFs and with forced OCR on raster. Marker runs in its balanced mode.

Scoring is `scripts/converter_gold.py` (`score`, `gates`). Every similarity cell below is read on the
same pairs: sections found by both arms, not overrun in either, and not doubtful in either (a row whose
section two cuts claim alike is left out, as the score's own means leave it). Paired deltas use the stand's
bootstrap (10000 draws, seed 42).

## Result

Similarity is the mean over the n pairs of the row, and "at 98" counts those pairs. "Cut" is the share of
all outputs of the population the stand's chunker can cut. Time is the sum over the population on one 8 GB GPU.

| Input kind | n | Docling | MinerU | MinerU minus Docling | Winner |
|---|---|---|---|---|---|
| Text-layer PDF, en (4 books, 54 cuts) | 48 | 84.3, at 98: 14; cut 51/54; 113 s | 78.0, at 98: 17; cut 54/54; 153 s | −6.3 [−13.0, −0.9] | Docling |
| Text-layer PDF, ru (3 books, 38 cuts) | 30 | 87.1, at 98: 5; cut 35/38; 58 s | 86.8, at 98: 9; cut 36/38; 126 s | −0.3 [−4.8, +3.4] | Docling, by speed: indistinguishable |
| Text-layer PDF, ru Postgres Pro (15 cuts, gold from its site) | 14 | 98.9, at 98: 11; cut 15/15; 42 s | 88.0, at 98: 8; cut 15/15; 127 s | −10.9 [−22.0, −1.9] | Docling |
| Raster ru, baseline (38) | 34 | 82.8, at 98: 0; cut 38/38; 378 s | 89.8, at 98: 12; cut 38/38; 248 s | +6.9 [+5.4, +8.5] | MinerU |
| Raster ru, scanner (38) | 33 | 83.0, at 98: 0; cut 37/38; 492 s | 88.4, at 98: 11; cut 38/38; 171 s | +5.5 [+3.6, +7.4] | MinerU |
| HTML, en / ru (50 / 65 pages) | 50 / 64 | 97.1 / 97.9, at 98: 30 / 45; cut 45/50 and 59/64 | not run: it does not read HTML | | Docling |

With doubtful rows kept the text-layer rows read en −6.6 [−12.9, −1.4] n=51, and ru with Postgres Pro pooled
−3.6 [−8.5, +0.5] n=44; the pooled ru band hides the clear Postgres Pro cell, which is why the rows are apart.

On raster MinerU also finds 60 of 65 headings against Docling's 47 of 66, and keeps 12 of 17 code blocks
whole against 1. Words mixing Latin and Cyrillic letters, the mark of a bad Cyrillic reading: MinerU 10-12,
Docling 12-18, the gold 3.

**Floors.** Docling repeats byte for byte on the text layer and the raster. MinerU repeats with at most
0.01 of similarity moved on the text layer, and at most 0.19 on the raster: its OCR is not byte-stable.
Every winning difference above sits well outside its floor.

**Marker** fails the bar on the text layer in both languages before speed is read: on en cuts 42/54
against Docling's 51, headings found 87 of 147 against 116, and 0 of 27 code blocks kept whole, because
code comes out as escaped prose with no fence. On raster it was stopped after 9 of 38 sections in 36
minutes. Its OCR model writes each page token by token, one request at a time, so the whole point would
have taken about 2.4 hours. It was taken out of the stand.

## What a site page needs before any converter

HTML was read twice: once as served, and once reduced to the site's own text. The reduction keeps the
element a per-site setting names, drops the site's furniture inside it (child-page lists, feedback
widgets, a translation banner, variant code blocks), and flattens syntax highlighting in `<pre>`. As
served, a kubernetes page is mostly its navigation tree: similarity 12.7 en and 16.7 ru, and every page
broken for the chunker. Reduced, it reads 97.3 and 99.3. Highlighted code on vue came out with a space
between every token (`console. log (`), 2 of 96 blocks kept whole; flattened, 96 of 96. This is a
property of the input, not a choice between converters: the corpus pipeline owns the reduction.

## Decision

- **Two converters by input kind.** A PDF with a text layer and an HTML page go to Docling, a scan or
  image goes to MinerU with forced OCR. The kind is read from the input itself (a text layer present or
  not), not chosen per document.
- Docling's default: Tesseract OCR, no code enrichment. MinerU's: `mineru/ocr` for raster.
- Marker is removed from the stand. Its readings stay in the arc log and its run folders.
- A site source declares where its text is, what inside it is furniture, and which pages the site builds
  itself.

## Caveats

- The raster was read on Russian only. English raster pages are rendered and have gold but were not run.
- The text-layer sections are short cuts of whole pages without the book's outline. On whole books the
  heading levels read differently.
- The Postgres Pro gold is pandoc's reading of its own site, so its numbers compare a converter with a
  converter.
- Several gold defects were found and fixed while reading: a comment line in code read as a heading, pipes
  in code read as table cells. All runs were rescored after each fix, and no winner changed.
- Scoring still lives in a script. Its move into the stand as a job and an experiment kind is the next
  step, and it must reproduce these numbers.
