"""Draw the chapter set: the second chapter (or the one `--nth` names) of every PDF document in the store, whole."""

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
INBOX = ROOT / "datasets" / "inbox"
GOLD = ROOT / "datasets" / "converter_gold"
# a book split into chapter files names its chapter by file; ostep's third is a short dialogue, its fourth stands in
CHAPTER_FILES = {
    2: {"ostep": "intro.pdf", "security-engineering-3e": "SE-02.pdf"},
    3: {"ostep": "cpu-intro.pdf", "security-engineering-3e": "SE-03.pdf"},
}
# outline entries that are not chapters: front and back matter, and a part that only holds chapters
MATTER = re.compile(
    r"^(cover|title|copyright|contents|content list|table of contents|preface|foreword|about|acknowledg|dedication|"
    r"index|bibliography|references|glossary|appendix|colophon|summary|begin$|"
    r"о книге|предисловие|содержание|оглавление|список литературы|предметный указатель|приложение)",
    re.I,
)
PART = re.compile(r"^(part|часть)\b", re.I)
# a document with no outline gives a fixed stretch per set, declared rather than guessed; a short one is read whole
NO_OUTLINE = {2: (100, 110), 3: (120, 130)}
WHOLE_UNDER = 40


def _outline(pdf):
    entries = []
    for item in pdf.get_toc(max_depth=15):
        dest = item.get_dest()
        if dest is not None and dest.get_index() is not None:
            entries.append((item.level, item.get_title().strip(), dest.get_index() + 1))
    return entries


# the chapters in reading order: top entries past the matter, a part's children standing for the part
def _chapters(entries):
    top = min((level for level, _, _ in entries), default=0)
    chapters, in_part = [], False
    for level, title, page in entries:
        if level == top:
            in_part = bool(PART.match(title))
            if not in_part and not MATTER.match(title):
                chapters.append((level, title, page))
        elif level == top + 1 and in_part and not MATTER.match(title):
            chapters.append((level, title, page))
    return chapters


def _numbered(nth: int):
    return re.compile(rf"^(chapter|глава)\s*0?{nth}\b|^0?{nth}(\s|\.\s|:)", re.I)


def _nth_chapter(path: Path, pages: int, nth: int) -> dict:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        entries = _outline(pdf)
    finally:
        pdf.close()
    if not entries and pages <= WHOLE_UNDER:
        return {"pages": [1, pages], "chapter": None, "by": "no outline, short: whole"}
    if not entries:
        return {"pages": list(NO_OUTLINE[nth]), "chapter": None, "by": "no outline"}
    numbered = next((e for e in entries if _numbered(nth).match(e[1])), None)
    chapters = _chapters(entries)
    chosen = numbered or (chapters[nth - 1] if len(chapters) >= nth else None)
    if chosen is None:
        return {"pages": list(NO_OUTLINE[nth]), "chapter": None, "by": f"no chapter {nth} in the outline"}
    level, title, first = chosen
    after = [page for lv, _, page in entries if lv <= level and page > first]
    # the next chapter often starts on the page this one ends on, so that page is read too and nothing is cut
    last = min(after) if after else pages
    return {"pages": [first, max(first, last)], "chapter": title, "by": "numbered outline" if numbered else "outline"}


def main() -> None:
    import pypdfium2 as pdfium

    nth = int(sys.argv[sys.argv.index("--nth") + 1]) if "--nth" in sys.argv else 2
    out = GOLD / ("chapter_set.yaml" if nth == 2 else f"chapter_set_{nth}.yaml")

    books = yaml.safe_load((INBOX / "books" / "books.yaml").read_text())["books"]
    documents = []
    # a book on hold, as the control group read only after the fixes, never enters a gate set
    for book in (b for b in books if not b.get("hold")):
        folder = INBOX / "books" / book["name"]
        loose = [INBOX / "books" / f for f in book.get("files", []) if (INBOX / "books" / f).is_file()]
        if not folder.is_dir() and not loose:
            continue
        if loose and not folder.is_dir():
            files, split = [p for p in loose if p.suffix.lower() == ".pdf"], False
        elif book["name"] in CHAPTER_FILES[nth]:
            files, split = [folder / CHAPTER_FILES[nth][book["name"]]], True
        else:
            files, split = sorted(folder.glob("*.pdf")), False
        for path in files:
            pages = len(pdfium.PdfDocument(str(path)))
            whole = {"pages": [1, pages], "chapter": path.stem, "by": "chapter file"}
            drawn = whole if split else _nth_chapter(path, pages, nth)
            # a document read whole in the second set was tuned on, so it is no unseen check in a later one
            if nth != 2 and drawn["by"] == "no outline, short: whole":
                continue
            file = str(path.relative_to(INBOX))
            documents.append(
                {"source": book["name"], "file": file, "language": book["language"], "file_pages": pages, **drawn}
            )
    for path in sorted((INBOX / "papers").glob("*.pdf")):
        pages = len(pdfium.PdfDocument(str(path)))
        drawn = _nth_chapter(path, pages, nth)
        if nth != 2 and drawn["by"] == "no outline, short: whole":
            continue
        documents.append(
            {"source": "papers", "file": str(path.relative_to(INBOX)), "language": "en", "file_pages": pages, **drawn}
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    head = (
        f"# chapter {nth} of every PDF document in the store, whole; drawn by scripts/chapter_set.py, "
        "read by convert_source with intake and root inbox\n"
    )
    out.write_text(head + yaml.safe_dump({"documents": documents}, allow_unicode=True, sort_keys=False, width=200))
    for d in documents:
        print(f"{d['file'][:55]:55} {d['by']:32} {d['pages']} {str(d['chapter'])[:40]}")
    total = sum(d["pages"][1] - d["pages"][0] + 1 for d in documents)
    print(len(documents), "documents,", total, "pages", file=sys.stderr)


if __name__ == "__main__":
    main()
