"""Fetch the converter gold by its manifest, and check that each PDF is the same document as its source."""

import argparse
import hashlib
import json
import random
import re
import subprocess
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import formats
import requests
import yaml
from use_cases import site_page

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "datasets" / "converter_gold"
FILES = GOLD / "files"
LEDGER = GOLD / "fetched.json"
# how alike two headings must be to be one: WRatio on folded text, chosen once on the pair set
_SCORE = yaml.safe_load((GOLD / "manifest.yaml").read_text())["score"]
HEADING_SIMILARITY = _SCORE["heading_similarity"]
# a section came through when its plain text is at least this alike to the gold's
RECOVERED_SIMILARITY = _SCORE["recovered_similarity"]


# the site page's reading lives in the stand, and the bench reads pages the same way
_main_element, _element, _spans, _dropped, _flat_pre = (
    site_page.main_element,
    site_page.element,
    site_page.spans,
    site_page.dropped,
    site_page.flat_pre,
)


def _documents():
    manifest = yaml.safe_load((GOLD / "manifest.yaml").read_text())
    for doc in manifest["documents"]:
        for lang in manifest["languages"]:
            if lang in doc:
                yield doc["id"], lang, doc[lang]


def _sha256(path):
    from engines.converter_tools import sha256

    return sha256(path)


def _download(url, folder):
    from use_cases import fetch

    target = folder / url.rsplit("/", 1)[-1]
    fetch.download(url, target)
    if target.suffix == ".zip":
        with zipfile.ZipFile(target) as archive:
            archive.extractall(folder)
    return {
        "url": url,
        "file": str(target.relative_to(FILES)),
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }


def _clone(source, folder):
    from use_cases import fetch

    state = fetch.clone(source["repo"], folder, source.get("ref"), source.get("path"), source.get("also", []))
    return {"repo": source["repo"], "path": source.get("path"), "ref": source.get("ref"), **state}


# a source published only as an archive, as arXiv serves a paper's LaTeX: the url has no name, so the file is named here
def _unpack(url, folder):
    from use_cases import fetch

    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "source.tar.gz"
    fetch.download(url, target)
    with tarfile.open(target) as archive:
        archive.extractall(folder, filter="data")
    return {"archive": url, "file": str(target.relative_to(FILES)), "sha256": _sha256(target)}


# a paper's HTML as arXiv renders it from the LaTeX, macros and all: a gold where pandoc cannot read the source
def _page(url, folder):
    from use_cases import fetch

    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "page.html"
    fetch.download(url, target)
    return {"page": url, "file": str(target.relative_to(FILES)), "sha256": _sha256(target)}


def fetch(only):
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    for doc_id, lang, entry in _documents():
        if only and doc_id not in only:
            continue
        key = f"{doc_id}/{lang}"
        record = {"fetched": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if "repo" in entry["source"]:
            record["source"] = _clone(entry["source"], FILES / doc_id / lang / "source")
        if "archive" in entry["source"]:
            record["source"] = _unpack(entry["source"]["archive"], FILES / doc_id / lang / "source")
        if "page" in entry["source"]:
            record["page"] = _page(entry["source"]["page"], FILES / doc_id / lang / "page")
        if "text_pdf" in entry:
            record["text_pdf"] = _download(entry["text_pdf"], FILES / doc_id / lang / "pdf")
        ledger[key] = record
        LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
        print(key, "ok")


# reST lets any repeated punctuation character underline a title
_RST_UNDERLINE = re.compile(r"^([^\w\s])\1{2,}\s*$")
_SUFFIXES = {"asciidoc": ".asc", "markdown": ".md", "latex": ".tex", "docbook_sgml": ".sgml", "rst": ".rst"}


# markup and punctuation off, a dotted number kept as one token; case is kept for the roman numeral test
_RST_ROLE = re.compile(r":[\w:+-]+:`~?([^`<]*)[^`]*`")
_MARKUP = re.compile(r"<[^>]+>|[`*_\\{}]")
_PUNCTUATION = re.compile(r"[^\w]+")
_MD_HEADING = re.compile(r"^(#{1,6}) (.+)$", re.M)
_MD_HEADING_MARK = re.compile(r"^(#{1,6}) ", re.M)
_SOURCE_HEADING = {"asciidoc": re.compile(r"^(={1,6}) (.+)$"), "markdown": _MD_HEADING}
_TABLE_RULE = re.compile(r"\s*:?-+:?\s*")
_HTML_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_TAG = re.compile(r"<[^>]+>")
_INDEX_SPAN = re.compile(r'<span class="index(term)?"[^>]*></span>\s?')
_SGML_ENTITY = re.compile(r"&[A-Za-z][\w.-]*;")
_UNRESOLVED_XREF = re.compile(r"\[\?\?\?\]\(#([^)]+)\)")
_ASCIIDOC_INCLUDE = re.compile(r"^include::([^\[]+)\[\]\s*$", re.M)
_ASCIIDOC_ANCHOR = re.compile(r"^\[\[([^\],]+)(?:,[^\]]*)?\]\]\s*$|^\[#([^\],.]+)[^\]]*\]\s*$")
_ASCIIDOC_ANCHOR_LINE = re.compile(r"^\[\[[^\]]*\]\]\s*$|^\[#[^\]]*\]\s*$", re.M)
_ASCIIDOC_TITLE = re.compile(r"^=+ (.+)$")
_LATEX_LEVELS = formats.format_of("latex").levels
_LATEX_HEADING = re.compile(rf"^\s*\\({'|'.join(map(re.escape, _LATEX_LEVELS))})\*?\{{(.+)\}}")
# not measured: the slowest gold section so far took seconds; an ARES fragment ran past fourteen minutes
PANDOC_SECONDS = 120
_LATEX_COMMENT_LINE = re.compile(r"^\s*%.*$", re.M)


# markup and punctuation off, one case
def _folded(title):
    import html
    import unicodedata

    # the source spells a character as an entity where the page prints it, and the page hides break hints
    title = "".join(c for c in html.unescape(title) if unicodedata.category(c) != "Cf")
    title = _MARKUP.sub("", _RST_ROLE.sub(r"\1", title))
    return " ".join(_PUNCTUATION.sub(" ", title).split()).casefold()


def _heading_similarity(a, b) -> float:
    from rapidfuzz import fuzz

    return fuzz.WRatio(_folded(a), _folded(b))


# the most similar of the folded candidates, if it is similar enough to be the same heading
def _best_heading(title, folded_candidates: list[str]) -> tuple[int, float] | None:
    from rapidfuzz import fuzz, process

    found = process.extractOne(
        _folded(title), folded_candidates, scorer=fuzz.WRatio, processor=None, score_cutoff=HEADING_SIMILARITY
    )
    return (found[2], found[1]) if found else None


def _source_titles(folder, source):
    fmt = source["format"]
    titles = set()
    for path in folder.rglob(f"*{source.get('suffix', _SUFFIXES[fmt])}"):
        if not path.is_file() or ".git" in path.parts:
            continue
        text = path.read_text(encoding=source.get("encoding", "utf-8"), errors="ignore")
        if fmt == "latex":
            text = _LATEX_COMMENT_LINE.sub("", text)
        titles.update(_folded(title) for _, _, title in _headings(text.splitlines(), fmt))
    return titles - {""}


def _outline(pdf, max_level: int):
    from pypdf import PdfReader

    reader = PdfReader(pdf)
    titles = []

    def walk(items, depth):
        for item in items:
            if isinstance(item, list):
                if depth < max_level - 1:
                    walk(item, depth + 1)
            else:
                titles.append(item.title)

    walk(reader.outline, 0)
    meta = reader.metadata or {}
    return titles, {
        "pages": len(reader.pages),
        "title": meta.get("/Title"),
        "producer": meta.get("/Producer"),
        "created": meta.get("/CreationDate"),
    }


# a basic identity check: the PDF's own outline titles should be headings of the source it is paired with
def pairs(only):
    declared = yaml.safe_load((GOLD / "manifest.yaml").read_text())
    max_level = declared["raster"]["sections"]["detect"]["max_toc_level"]
    for doc_id, lang, entry in _documents():
        if only and doc_id not in only:
            continue
        pdfs = sorted((FILES / doc_id / lang / "pdf").rglob("*.pdf"))
        source = FILES / doc_id / lang / "source"
        if not pdfs or not source.exists():
            continue
        known = _source_titles(source, entry["source"])
        outline, meta = [], {}
        for pdf in pdfs:
            titles, meta = _outline(pdf, max_level)
            outline += titles
        normal = [t for t in outline if _folded(t)]
        candidates = sorted(known)
        hits = [t for t in normal if _best_heading(t, candidates)]
        missing = [t for t in normal if t not in hits]
        share = len(hits) / len(normal) if normal else 0.0
        # a printed share is not a record: the pair's identity lives beside what was fetched
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
        ledger.setdefault(f"{doc_id}/{lang}", {})["pair"] = {
            "outline_depth": max_level,
            "outline_titles": len(normal),
            "found_in_source": len(hits),
            "share": round(share, 4),
            "threshold": declared["pair_threshold"],
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
        print(
            f"{doc_id}/{lang}: pdf outline {len(normal)}, found in source {len(hits)} ({share:.0%}), "
            f"source headings {len(known)}, pdfs {len(pdfs)}, last meta {meta}"
        )
        for title in missing[:8]:
            print("   not in source:", title)


def smoke(only):
    from pypdf import PdfReader, PdfWriter

    manifest = yaml.safe_load((GOLD / "manifest.yaml").read_text())
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    for cut in manifest.get("smoke", []):
        if only and cut["id"] not in only:
            continue
        source = FILES / cut["from"]
        target = FILES / "smoke" / f"{cut['id']}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        reader, writer = PdfReader(source), PdfWriter()
        first, last = cut["pages"]
        for number in range(first - 1, last):
            writer.add_page(reader.pages[number])
        with target.open("wb") as f:
            writer.write(f)
        ledger[f"smoke/{cut['id']}"] = {
            "from": cut["from"],
            "from_sha256": _sha256(source),
            "pages": cut["pages"],
            "file": str(target.relative_to(FILES)),
            "sha256": _sha256(target),
        }
        LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
        print(cut["id"], "ok")


SECTIONS = GOLD / "sections.json"
CUTS = FILES / "sections"


# a span's glyphs as (glyph, width relative to the font size)
def _glyphs(span: dict) -> list[tuple[str, float]]:
    return [(c["c"], (c["bbox"][2] - c["bbox"][0]) / span["size"]) for c in span["chars"] if c["c"].strip()]


# glyphs of one width make a monospace font whatever its name; too few distinct glyphs say nothing
def _uniform_width(glyphs: list[tuple[str, float]], detect: dict) -> bool | None:
    if len({g for g, _ in glyphs}) < detect["mono_min_distinct"]:
        return None
    widths = sorted(w for _, w in glyphs)
    low, high = widths[len(widths) // 20], widths[-len(widths) // 20 - 1]
    return low > 0 and high / low <= detect["mono_width_spread"]


# bit 3 of a pymupdf span's flags is the font's own monospace mark
def _is_mono(span: dict, detect: dict) -> bool | None:
    return True if span["flags"] & 8 else _uniform_width(_glyphs(span), detect)


# the fonts a document uses, with the ones read as monospace marked: a zero of code is checked against it
def _fonts(doc, pages: set[int], detect: dict) -> dict:
    glyphs, flagged = {}, set()
    for number in sorted(pages):
        for block in doc[number - 1].get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                for s in line["spans"]:
                    glyphs.setdefault(s["font"], []).extend(_glyphs(s))
                    if s["flags"] & 8:
                        flagged.add(s["font"])
    return {font: font in flagged or bool(_uniform_width(seen, detect)) for font, seen in glyphs.items()}


# a figure of boxes and arrows is a ruled grid too, so a table needs rows, columns and text in its cells
def _is_table(table, detect: dict) -> bool:
    if table.row_count < detect["table_min_rows"] or table.col_count < detect["table_min_cols"]:
        return False
    cells = [c for row in table.extract() for c in row]
    filled = sum(1 for c in cells if c and c.strip())
    return bool(cells) and filled / len(cells) >= detect["table_min_filled"]


# code by a monospace run of lines, a table by the tool that finds ruled grids
def _kind(doc, boxes: list[dict], detect: dict) -> str:
    import pymupdf

    clips = [(doc[b["page"] - 1], pymupdf.Rect(*b["clip"])) for b in boxes]
    for page, clip in clips:
        if any(_is_table(t, detect) for t in page.find_tables(clip=clip).tables):
            return "table"
    for page, clip in clips:
        mono_lines = 0
        for block in page.get_text("rawdict", clip=clip)["blocks"]:
            for line in block.get("lines", []):
                read = [_is_mono(s, detect) for s in line["spans"] if any(c["c"].strip() for c in s["chars"])]
                if any(read) and all(r is not False for r in read):
                    mono_lines += 1
        if mono_lines >= detect["code_min_lines"]:
            return "code"
    return "prose"


# a section runs from its heading's page to the page of the next heading at its level or above
def _candidates(pdf: Path, known: set | None, detect: dict) -> list[dict]:
    import pymupdf

    doc = pymupdf.open(pdf)
    toc = doc.get_toc()
    out = []
    for i, (level, title, page) in enumerate(toc):
        if level > detect["max_toc_level"] or page < 1:
            continue
        # the last section at its level runs to the document's end, not to its own first page
        end = next((p for lv, _, p in toc[i + 1 :] if lv <= level and p >= 1), doc.page_count)
        last = max(page, end)
        if last - page + 1 > detect["max_pages"] or not _folded(title):
            continue
        if known is not None and not _best_heading(title, known):
            continue
        out.append({"title": title, "level": level, "pages": [page, last]})
    return out


def draw(only):
    import pymupdf

    manifest = yaml.safe_load((GOLD / "manifest.yaml").read_text())
    raster = manifest["raster"]
    plan = raster["sections"]
    detect = plan["detect"]
    drawn, deficits, census, skipped = [], [], [], []
    for doc_id, lang, entry in _documents():
        if doc_id not in raster["documents"].get(lang, []) or (only and doc_id not in only):
            continue
        source = FILES / doc_id / lang / "source"
        known = sorted(_source_titles(source, entry["source"])) if source.exists() else None
        rng = random.Random(f"{plan['seed']}:{doc_id}:{lang}")
        candidates = [
            (pdf, c)
            for pdf in sorted((FILES / doc_id / lang / "pdf").rglob("*.pdf"))
            for c in _candidates(pdf, known, detect)
        ]
        rng.shuffle(candidates)
        buckets = {kind: [] for kind in plan["kinds"]}
        opened, tocs = {}, {}
        for pdf, c in candidates:
            if all(len(b) >= plan["per_kind"] for b in buckets.values()):
                break
            doc = opened.setdefault(pdf, pymupdf.open(pdf))
            boxes, notes, heading = _crop_boxes(doc, c, tocs.setdefault(pdf, doc.get_toc()), detect)
            if not boxes:
                skipped.append({"doc": doc_id, "lang": lang, "title": c["title"], "why": notes})
                continue
            kind = _kind(doc, boxes, detect)
            if len(buckets[kind]) < plan["per_kind"]:
                buckets[kind].append(
                    {
                        **c,
                        "kind": kind,
                        "pdf": str(pdf.relative_to(FILES)),
                        "boxes": boxes,
                        "heading": heading,
                        "notes": notes,
                    }
                )
        for kind, got in buckets.items():
            if len(got) < plan["per_kind"]:
                deficits.append({"doc": doc_id, "lang": lang, "kind": kind, "short_by": plan["per_kind"] - len(got)})
            for n, c in enumerate(got):
                drawn.append(
                    {
                        "id": f"{doc_id}_{lang}_{kind}_{n:02}",
                        "doc": doc_id,
                        "lang": lang,
                        "gold_titles_checked": known is not None,
                        **c,
                    }
                )
        # the census reads the pages that are cut, so a book whose code starts late is not read as having none
        fonts = {}
        for c in (c for got in buckets.values() for c in got):
            fonts.update(_fonts(opened[FILES / c["pdf"]], {b["page"] for b in c["boxes"]}, detect))
        mono = sorted(font for font, is_mono in fonts.items() if is_mono)
        census.append({"doc": doc_id, "lang": lang, "mono_fonts": mono})
        print(doc_id, lang, {k: len(v) for k, v in buckets.items()}, "of", len(candidates), "candidates, mono", mono)
        if not buckets.get("code"):
            print("   no code found; fonts seen:", sorted(fonts))
    # a draw of some documents keeps the others' rows, so a partial redraw does not wipe the declaration
    if only and SECTIONS.exists():
        kept = json.loads(SECTIONS.read_text())
        drawn = [x for x in kept["sections"] if x["doc"] not in only] + drawn
        deficits = [x for x in kept["deficits"] if x["doc"] not in only] + deficits
        census = [x for x in kept["fonts"] if x["doc"] not in only] + census
        skipped = [x for x in kept.get("skipped", []) if x["doc"] not in only] + skipped
    SECTIONS.write_text(
        json.dumps(
            {
                "seed": plan["seed"],
                "detect": detect,
                "fonts": census,
                "sections": drawn,
                "deficits": deficits,
                "skipped": skipped,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


RASTER = FILES / "raster"
RASTER_LEDGER = GOLD / "raster.json"


# the section's vertical span: from its heading on the first page to the next heading at its level or above
def _crop_boxes(doc, section, toc, detect: dict) -> tuple[list[dict], list[str], dict | None]:
    first, last = section["pages"]
    notes = []
    at = next((i for i, (lv, t, p) in enumerate(toc) if t == section["title"] and p == first), None)
    after = next(((t, p) for lv, t, p in toc[at + 1 :] if lv <= section["level"]), None) if at is not None else None
    heading = _heading_hit(doc[first - 1], section["title"])
    # a wrong top is a wrong gold, so a section whose heading is not on its page is not drawn at all
    if not heading:
        return [], ["heading not found on its page"], None
    top = heading["y"]
    bottom_page = doc[last - 1]
    bottom = bottom_page.rect.y1
    if after and after[1] == last:
        end = _heading_hit(bottom_page, after[0], below=top if last == first else None)
        if end:
            bottom = end["y"]
        else:
            notes.append("next heading not found on the last page, the crop runs to the page end")
    boxes = []
    for number in range(first, last + 1):
        page = doc[number - 1]
        rect = page.rect
        head, foot = _margins(page, detect)
        y0 = top if number == first else head
        y1 = min(bottom, foot) if number == last else foot
        if y1 - y0 > 1:
            boxes.append({"page": number, "clip": [rect.x0, round(y0, 2), rect.x1, round(y1, 2)]})
    return boxes, notes, heading


# a running header or footer is the document's own repetition: a text block at one height on most pages of a parity
def _running_heights(doc, detect: dict) -> set[tuple[int, int, int]]:
    if doc.name not in _RUNNING:
        seen: dict[tuple[int, int, int], int] = {}
        for number in range(len(doc)):
            for key in {(number % 2, round(b[1]), round(b[3])) for b in doc[number].get_text("blocks") if b[4].strip()}:
                seen[key] = seen.get(key, 0) + 1
        pages = {parity: len(range(parity, len(doc), 2)) for parity in (0, 1)}
        _RUNNING[doc.name] = {k for k, n in seen.items() if n > pages[k[0]] * detect["running_block_share"]}
    return _RUNNING[doc.name]


_RUNNING: dict = {}


# the crop stops short of the running blocks at the page's top and bottom
def _margins(page, detect: dict) -> tuple[float, float]:
    rect = page.rect
    running = _running_heights(page.parent, detect)
    blocks = [
        b for b in page.get_text("blocks") if b[4].strip() and (page.number % 2, round(b[1]), round(b[3])) in running
    ]
    middle = (rect.y0 + rect.y1) / 2
    head = max([b[3] for b in blocks if b[3] <= middle], default=rect.y0)
    foot = min([b[1] for b in blocks if b[1] >= middle], default=rect.y1)
    return head, foot


# the heading is the line most like the title, set largest among the alike (a running header repeats it small)
def _heading_hit(page, title: str, below: float | None = None) -> dict | None:
    hits = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [sp for sp in line["spans"] if sp["text"].strip()]
            text = " ".join(sp["text"].strip() for sp in spans)
            y = line["bbox"][1]
            if spans and (below is None or y > below) and _heading_similarity(text, title) >= HEADING_SIMILARITY:
                hits.append({"y": round(y, 2), "size": round(max(sp["size"] for sp in spans), 2)})
    return max(hits, key=lambda h: (h["size"], -h["y"])) if hits else None


def _seeded(seed: int, *parts: str):
    import numpy

    digest = hashlib.sha256(":".join([str(seed), *parts]).encode()).digest()
    return numpy.random.default_rng(int.from_bytes(digest[:8], "big"))


def _degrade(image, axis: str, step, rng):
    import io

    import numpy
    from PIL import Image, ImageFilter

    if axis == "jpeg_quality":
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=int(step))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if axis == "noise_sigma":
        arr = numpy.asarray(image).astype(numpy.float32)
        arr += rng.normal(0, float(step), (arr.shape[0], arr.shape[1], 1))
        return Image.fromarray(numpy.clip(arr, 0, 255).astype(numpy.uint8))
    if axis == "skew_degrees":
        return image.rotate(float(step), expand=True, fillcolor="white", resample=Image.BICUBIC)
    if axis == "blur_radius_px":
        return image.filter(ImageFilter.GaussianBlur(float(step)))
    raise ValueError(axis)


def _points(raster: dict) -> list[tuple[str, dict]]:
    base = raster["baseline"]["dpi"]
    points = [("baseline", {"dpi": base})]
    for axis, steps in raster["ladder"].items():
        for step in steps:
            points.append((f"{axis}_{step}", {"dpi": step} if axis == "dpi" else {"dpi": base, axis: step}))
    points += [(name, dict(spec)) for name, spec in raster.get("composite", {}).items()]
    return points


# a point's degradations in a fixed order after the render at its dpi: skew, blur, noise, compression
ORDER = ("skew_degrees", "blur_radius_px", "noise_sigma", "jpeg_quality")


def raster(only):
    import pymupdf
    from PIL import Image

    manifest = yaml.safe_load((GOLD / "manifest.yaml").read_text())
    plan = manifest["raster"]
    seed = plan["sections"]["seed"]
    langs = [x for x in plan["language_order"] if not only or x in only] or plan["language_order"]
    ledger = json.loads(RASTER_LEDGER.read_text()) if RASTER_LEDGER.exists() else {"sections": {}}
    points = _points(plan)
    opened = {}
    for section in json.loads(SECTIONS.read_text())["sections"]:
        if section["lang"] not in langs:
            continue
        doc = opened.setdefault(section["pdf"], pymupdf.open(FILES / section["pdf"]))
        boxes = section["boxes"]
        ledger["sections"][section["id"]] = {
            "pdf": section["pdf"],
            "boxes": boxes,
            "heading": section["heading"],
            "notes": section["notes"],
        }
        for name, spec in points:
            folder = RASTER / name
            folder.mkdir(parents=True, exist_ok=True)
            rng = _seeded(seed, section["id"], name)
            pages = []
            for box in boxes:
                pix = doc[box["page"] - 1].get_pixmap(dpi=int(spec["dpi"]), clip=pymupdf.Rect(*box["clip"]))
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                for axis in ORDER:
                    if axis in spec:
                        image = _degrade(image, axis, spec[axis], rng)
                pages.append(image)
            pages[0].save(
                folder / f"{section['id']}.pdf",
                "PDF",
                save_all=True,
                append_images=pages[1:],
                resolution=float(spec["dpi"]),
            )
            if name in plan["jpeg_pages_points"]:
                for n, page in enumerate(pages):
                    page.save(folder / f"{section['id']}_p{n:02}.jpg", "JPEG", quality=plan["jpeg_pages_quality"])
        print(section["id"], len(boxes), "pages", section["notes"] or "")
    ledger["points"] = {name: spec for name, spec in points}
    ledger["jpeg_twins"] = (
        f"the loose pages are saved as JPEG at quality {plan['jpeg_pages_quality']} after the point's own degradation"
    )
    ledger["seed"] = seed
    RASTER_LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))


# the text layer stays: the text_pdf regime reads these, the raster regime renders them
def cut(only):
    import pymupdf

    CUTS.mkdir(parents=True, exist_ok=True)
    drawn = json.loads(SECTIONS.read_text())["sections"]
    # the folder is the draw and nothing else: a cut of an earlier draw would be picked up by a glob
    for stale in set(p.stem for p in CUTS.glob("*.pdf")) - {x["id"] for x in drawn}:
        (CUTS / f"{stale}.pdf").unlink()
    for section in drawn:
        if only and section["doc"] not in only:
            continue
        first, last = section["pages"]
        out = pymupdf.open()
        out.insert_pdf(pymupdf.open(FILES / section["pdf"]), from_page=first - 1, to_page=last - 1)
        out.save(CUTS / f"{section['id']}.pdf")
    print(len(list(CUTS.glob("*.pdf"))), "cuts")


GOLD_MD = FILES / "gold_md"
GOLD_LEDGER = GOLD / "gold.json"
PANDOC_READERS = {
    "asciidoc": "asciidoc",
    "rst": "rst",
    "latex": "latex",
    "markdown": "markdown",
    "docbook_sgml": "docbook",
    "html": "html",
}


# every heading of a source file as (line index, level, title), in the file's own convention
def _headings(lines: list[str], fmt: str) -> list[tuple[int, int, str]]:
    found = []
    if fmt in ("asciidoc", "markdown"):
        for i, line in enumerate(lines):
            m = _SOURCE_HEADING[fmt].match(line)
            if m:
                found.append((i, len(m.group(1)), m.group(2)))
    elif fmt == "rst":
        styles = []
        for i in range(len(lines) - 1):
            under = lines[i + 1].strip()
            if lines[i].strip() and _RST_UNDERLINE.match(lines[i + 1]) and len(under) >= len(lines[i].strip()):
                over = i > 0 and lines[i - 1].strip() == under
                style = (under[0], over)
                if style not in styles:
                    styles.append(style)
                found.append((i - 1 if over else i, styles.index(style) + 1, lines[i].strip()))
    elif fmt == "latex":
        for i, line in enumerate(lines):
            m = _LATEX_HEADING.match(line)
            if m:
                found.append((i, _LATEX_LEVELS[m.group(1)], m.group(2)))
    elif fmt == "docbook_sgml":
        found = _docbook_headings("\n".join(lines))
    return found


# DocBook 4's sectioning elements, and the other elements its schema lets carry a title of their own
_DOCBOOK_SECTIONS = site_page.DOCBOOK_SECTIONS
_DOCBOOK_TITLED = "|".join(map(re.escape, formats.format_of("docbook").titled))
_DOCBOOK_SECTION = re.compile(rf"<(/?)({_DOCBOOK_SECTIONS})\b[^>]*>")
_DOCBOOK_OWNER = re.compile(rf"<({_DOCBOOK_SECTIONS}|{_DOCBOOK_TITLED})\b[^>]*>")
# a reference page has no title element: its name is the refentrytitle
_DOCBOOK_TITLE = re.compile(r"<(title|refentrytitle)>(.+?)</\1>", re.S)
_DOCBOOK_SECTION_NAME = re.compile(_DOCBOOK_SECTIONS)
_DOCBOOK_OPENING = re.compile(rf"<({_DOCBOOK_SECTIONS})\b")
_DOCBOOK_ID = re.compile(r'id="([^"]+)"(?:[^>]*xreflabel="([^"]+)")?[^>]*>\s*(?:<title>(.+?)</title>)?')


# a title is a heading when the element it opens is a section; its line is where it starts, its level the depth
def _docbook_headings(text: str) -> list[tuple[int, int, str]]:
    found, depth, owner = [], 0, None
    events = sorted(
        [(m.start(), "owner", m) for m in _DOCBOOK_OWNER.finditer(text)]
        + [(m.start(), "section", m) for m in _DOCBOOK_SECTION.finditer(text)]
        + [(m.start(), "title", m) for m in _DOCBOOK_TITLE.finditer(text)],
        key=lambda e: e[0],
    )
    for at, kind, m in events:
        if kind == "owner":
            owner = m.group(1)
        elif kind == "section":
            depth += -1 if m.group(1) else 1
        elif owner and _DOCBOOK_SECTION_NAME.fullmatch(owner):
            found.append((text.count("\n", 0, at), depth, " ".join(m.group(2).split())))
    return found


def _inline_includes(text: str, base: Path, root: Path) -> str:
    def swap(m):
        target = (base / m.group(1)) if (base / m.group(1)).exists() else (root / m.group(1))
        return _inline_includes(target.read_text(errors="ignore"), target.parent, root) if target.exists() else ""

    return _ASCIIDOC_INCLUDE.sub(swap, text)


# the section's own source: from its heading to the next heading at its level or above, in the same file
def _source_fragment(source_dir: Path, source: dict, title: str) -> tuple[str, str] | None:
    fmt = source["format"]
    best = None
    for path in sorted(source_dir.rglob(f"*{source.get('suffix', _SUFFIXES[fmt])}")):
        if not path.is_file() or ".git" in path.parts:
            continue
        text = path.read_text(encoding=source.get("encoding", "utf-8"), errors="ignore")
        if fmt == "asciidoc":
            text = _inline_includes(text, path.parent, source_dir)
        lines = text.splitlines()
        heads = _headings(lines, fmt)
        found = _best_heading(title, [_folded(h) for _, _, h in heads])
        if found and (best is None or found[1] > best[0]):
            best = (found[1], path, lines, heads, found[0])
    if best is None:
        return None
    _, path, lines, heads, n = best
    at, level, _ = heads[n]
    if fmt == "docbook_sgml":
        return _docbook_element(lines, at), str(path.relative_to(source_dir))
    end = next((i for i, lv, _ in heads[n + 1 :] if lv <= level), len(lines))
    fragment = "\n".join(lines[at:end])
    if fmt == "asciidoc":
        # block anchors carry no text, and pandoc's reader stops on some of them
        fragment = _ASCIIDOC_ANCHOR_LINE.sub("", fragment)
    return fragment, str(path.relative_to(source_dir))


_LATEX_DEFINITION = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\*?\s*"
    r"\{?\\([A-Za-z]+)\}?\s*(?:\[(\d)\])?|\\def\s*\\([A-Za-z]+)((?:#\d)*)\s*(?=\{)"
)
_LATEX_COMMENT = re.compile(r"(?<!\\)%.*$", re.M)
_LATEX_USE = re.compile(r"\\([A-Za-z]+)")


# the body in braces that starts at `at`, braces balanced
def _braced(text: str, at: int) -> str | None:
    if at >= len(text) or text[at] != "{":
        return None
    depth = 0
    for i in range(at, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0) if text[i - 1] != "\\" or i == at else 0
        if depth == 0:
            return text[at + 1 : i]
    return None


# every macro the source defines for itself; the book's own files near the root win over example documents below
def _latex_definitions(source_dir: Path, encoding: str) -> dict[str, str]:
    if source_dir not in _DEFINITIONS:
        found = {}
        for path in sorted(source_dir.rglob("*"), key=lambda p: (len(p.parts), str(p))):
            if path.suffix not in (".tex", ".sty", ".cls") or not path.is_file() or ".git" in path.parts:
                continue
            text = _LATEX_COMMENT.sub("", path.read_text(encoding=encoding, errors="ignore"))
            for m in _LATEX_DEFINITION.finditer(text):
                name = m.group(1) or m.group(3)
                args = int(m.group(2) or 0) if m.group(1) else len(m.group(4)) // 2
                start = m.end()
                if m.group(1) and text[start : start + 1] == "[":
                    start = text.index("]", start) + 1
                body = _braced(text, start + len(text[start:]) - len(text[start:].lstrip()))
                if body is not None and name not in found:
                    found[name] = f"\\newcommand{{\\{name}}}" + (f"[{args}]" if args else "") + f"{{{body}}}"
        _DEFINITIONS[source_dir] = found
    return _DEFINITIONS[source_dir]


_DEFINITIONS: dict = {}


# babel's Russian shorthands, and the source's own macros defined ahead of the fragment so pandoc expands them
def _latex_readable(fragment: str, definitions: dict[str, str]) -> str:
    for shorthand, glyph in (('"---', "\u2014"), ('"--~', "\u2014"), ("<<", "\u00ab"), (">>", "\u00bb")):
        fragment = fragment.replace(shorthand, glyph)
    wanted, queue = [], list(_LATEX_USE.findall(fragment))
    while queue:
        name = queue.pop()
        if name in definitions and name not in wanted:
            wanted.append(name)
            queue += _LATEX_USE.findall(definitions[name])
    return "".join(definitions[n] + "\n" for n in reversed(wanted)) + fragment


# a cross-reference out of the section prints its target's label on the page; pandoc knows only the fragment
def _docbook_labels(source_dir: Path) -> dict[str, str]:
    if source_dir not in _LABELS:
        labels = {}
        for path in source_dir.rglob("*.sgml"):
            text = path.read_text(errors="ignore")
            for m in _DOCBOOK_ID.finditer(text):
                label = m.group(2) or (_TAG.sub("", m.group(3)) if m.group(3) else None)
                if label:
                    labels[m.group(1)] = label
        _LABELS[source_dir] = labels
    return _LABELS[source_dir]


_LABELS: dict = {}


# asciidoctor prints a bare cross-reference as its target's title; the anchor stands on the line above the heading
def _asciidoc_labels(source_dir: Path) -> dict[str, str]:
    if source_dir not in _LABELS:
        labels = {}
        for path in source_dir.rglob("*.asc"):
            lines = path.read_text(errors="ignore").splitlines()
            for n, line in enumerate(lines[:-1]):
                anchor = _ASCIIDOC_ANCHOR.match(line)
                heading = _ASCIIDOC_TITLE.match(lines[n + 1])
                if anchor and heading:
                    labels[anchor.group(1) or anchor.group(2)] = heading.group(1).strip()
        _LABELS[source_dir] = labels
    return _LABELS[source_dir]


_ASCIIDOC_XREF = re.compile(r'<a href="#([^"]+)" class="cross-reference">([^<]*)</a>')


# a reference whose text is its own target was written bare, so the page prints the target's title
def _resolve_asciidoc_xrefs(markdown: str, labels: dict[str, str]) -> str:
    def title(m):
        target = m.group(1).split("#")[-1]
        return labels.get(target, m.group(2)) if m.group(2) == m.group(1) else m.group(2)

    return _ASCIIDOC_XREF.sub(title, markdown)


def _resolve_xrefs(markdown: str, labels: dict[str, str]) -> str:
    return _UNRESOLVED_XREF.sub(lambda m: labels.get(m.group(1), m.group(0)), markdown)


# a DocBook section is its element whole: from the opening tag above its title to the matching close
def _docbook_element(lines: list[str], title_at: int) -> str:
    start = next(i for i in range(title_at, -1, -1) if _DOCBOOK_OPENING.search(lines[i]))
    tag = _DOCBOOK_OPENING.search(lines[start]).group(1)
    depth = 0
    for i in range(start, len(lines)):
        depth += len(re.findall(rf"<{tag}\b", lines[i])) - len(re.findall(rf"</{tag}>", lines[i]))
        if depth == 0:
            return "\n".join(lines[start : i + 1])
    return "\n".join(lines[start:])


# the section is `##` and each deeper level one more, by rank: a source that skips a level does not count as a loss
def _rebased(markdown: str) -> str:
    # a comment line in code starts with "# " too, and it is neither a heading nor a level
    parts = re.split(r"(^```[^\n]*\n.*?^```)", markdown, flags=re.M | re.S)
    prose = parts[::2]
    ranks = {level: n for n, level in enumerate(sorted({len(m[0]) for p in prose for m in _MD_HEADING.findall(p)}))}
    for i in range(0, len(parts), 2):
        parts[i] = _MD_HEADING_MARK.sub(lambda m: "#" * min(6, 2 + ranks[len(m.group(1))]) + " ", parts[i])
    return "".join(parts)


# fastapi's include macro prints a file, or its ln[a:b] lines, as a code block; hl[...] is a display hint
_MD_CODE_INCLUDE = re.compile(r"^\{\*\s*(\S+)((?:\s+\w+\[[^\]]*\])*)\s*\*\}\s*$", re.M)
_LINE_RANGE = re.compile(r"ln\[(\d+):(\d+)\]")


# the path is relative to the file or to a folder above it, whichever holds it, as the site's build resolves it
def _markdown_code_includes(text: str, base: Path, root: Path) -> str:
    def swap(m):
        target = next(
            (d / m.group(1) for d in [base, *base.parents] if (d / m.group(1)).is_file() and root in [d, *d.parents]),
            None,
        )
        if target is None:
            return m.group(0)
        lines = target.read_text(errors="ignore").splitlines()
        span = _LINE_RANGE.search(m.group(2))
        if span:
            lines = lines[int(span.group(1)) - 1 : int(span.group(2))]
        return f"```{target.suffix.lstrip('.')}\n" + "\n".join(lines) + "\n```"

    return _MD_CODE_INCLUDE.sub(swap, text)


# a source fragment as the gold's markdown, the same way for a drawn PDF section and for a site's page
def _gold_markdown(fragment: str, source: dict, source_dir: Path, base: Path | None = None) -> tuple[str, str]:
    import pypandoc

    fmt = source["format"]
    if fmt == "markdown" and base is not None:
        fragment = _markdown_code_includes(fragment, base, source_dir)
    if fmt == "latex":
        fragment = _latex_readable(fragment, _latex_definitions(source_dir, source.get("encoding", "utf-8")))
    if fmt == "docbook_sgml":
        fragment = _SGML_ENTITY.sub(
            lambda m: m.group(0) if m.group(0) in ("&amp;", "&lt;", "&gt;", "&quot;", "&apos;") else "", fragment
        )
    # a fragment pandoc cannot finish is a section without gold, not a gold step that never ends
    markdown = subprocess.run(
        [pypandoc.get_pandoc_path(), f"--from={PANDOC_READERS[fmt]}", "--to=gfm", "--wrap=none"],
        input=fragment,
        capture_output=True,
        text=True,
        timeout=PANDOC_SECONDS,
        check=True,
    ).stdout
    # index terms come through as empty spans that the page never shows
    markdown = _INDEX_SPAN.sub("", markdown)
    if fmt == "docbook_sgml":
        markdown = _resolve_xrefs(markdown, _docbook_labels(source_dir))
    if fmt == "asciidoc":
        markdown = _resolve_asciidoc_xrefs(markdown, _asciidoc_labels(source_dir))
    return _rebased(markdown), fragment


# the drawn site pages' golds again from the source files and saved pages, without fetching anything
def _rebuild_page_golds(only):
    if not HTML_LEDGER.exists():
        return
    entries = {(doc_id, lang): entry for doc_id, lang, entry in _documents()}
    ledger = json.loads(HTML_LEDGER.read_text())
    for page in ledger["sections"]:
        if only and page["doc"] not in only:
            continue
        source = entries[(page["doc"], page["lang"])]["source"]
        if "source_file" in page:
            repo_dir = FILES / page["doc"] / page["lang"] / "source"
            source_dir = repo_dir / source["path"] if source.get("path") else repo_dir
            made = _file_gold(source_dir / page["source_file"], source, repo_dir)
            if made is None:
                continue
            markdown, readable, _ = made
            page["fragment_sha256"] = hashlib.sha256(readable.encode()).hexdigest()
        else:
            markdown = _html_gold((FILES / page["html"]).read_text(errors="ignore"), source.get("drop", []))
            if markdown is None:
                continue
        (FILES / page["gold"]).write_text(markdown)
    HTML_LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))


def gold(only):
    import pypandoc

    ledger = {
        "tool": f"pandoc {pypandoc.get_pandoc_version()}",
        "levels": "relative by rank: the section is ##, each deeper distinct level one more; a skipped level "
        "is forgiven, a merged level is caught, an inversion would be forgiven",
        "sections": {},
    }
    GOLD_MD.mkdir(parents=True, exist_ok=True)
    entries = {(doc_id, lang): entry for doc_id, lang, entry in _documents()}
    for section in json.loads(SECTIONS.read_text())["sections"]:
        if only and section["doc"] not in only:
            continue
        entry = entries[(section["doc"], section["lang"])]
        source_dir = FILES / section["doc"] / section["lang"] / "source"
        if "html" in entry["source"]:
            index_path = FILES / section["doc"] / section["lang"] / "site" / "index.json"
            url = (
                json.loads(index_path.read_text()).get("pdf_sections", {}).get(section["id"])
                if index_path.exists()
                else None
            )
            page = (
                FILES / section["doc"] / section["lang"] / "site" / (url.rsplit("/", 1)[-1] + ".html") if url else None
            )
            markdown = (
                _html_gold(page.read_text(errors="ignore"), entry["source"].get("drop", []))
                if page and page.exists()
                else None
            )
            if markdown is None:
                ledger["sections"][section["id"]] = {"gold": None, "why": "no site page for the title"}
                continue
            target = GOLD_MD / f"{section['id']}.md"
            target.write_text(markdown)
            ledger["sections"][section["id"]] = {
                "gold": str(target.relative_to(FILES)),
                "source_page": url,
                "gold_from": "the page's own DocBook element by pandoc",
                "sha256": _sha256(target),
            }
            continue
        if "page" in entry["source"]:
            page = FILES / section["doc"] / section["lang"] / "page" / "page.html"
            found = _page_section(page.read_text(), section["title"]) if page.exists() else None
            if found is None:
                ledger["sections"][section["id"]] = {"gold": None, "why": "heading not found in the page"}
                continue
            element_id, element = found
            markdown = pypandoc.convert_text(
                element, "gfm", format="html-native_divs-native_spans", extra_args=["--wrap=none"]
            )
            target = GOLD_MD / f"{section['id']}.md"
            target.write_text(_rebased(_ANCHOR_TAG.sub("", markdown)))
            ledger["sections"][section["id"]] = {
                "gold": str(target.relative_to(FILES)),
                "source_page": f"{entry['source']['page']}#{element_id}",
                "gold_from": "the paper's section in arXiv's HTML by pandoc",
                "sha256": _sha256(target),
            }
            continue
        if not ({"repo", "archive"} & set(entry["source"])) or not source_dir.exists():
            ledger["sections"][section["id"]] = {"gold": None, "why": "no source repository"}
            continue
        found = _source_fragment(source_dir, entry["source"], section["title"])
        if found is None:
            ledger["sections"][section["id"]] = {"gold": None, "why": "heading not found in the source"}
            continue
        fragment, file = found
        try:
            markdown, fragment = _gold_markdown(fragment, entry["source"], source_dir)
        except (RuntimeError, subprocess.SubprocessError) as e:
            ledger["sections"][section["id"]] = {"gold": None, "why": f"pandoc: {str(e)[:200]}"}
            continue
        target = GOLD_MD / f"{section['id']}.md"
        target.write_text(markdown)
        ledger["sections"][section["id"]] = {
            "gold": str(target.relative_to(FILES)),
            "source_file": file,
            "fragment_sha256": hashlib.sha256(fragment.encode()).hexdigest(),
            "sha256": _sha256(target),
        }
    _rebuild_page_golds(only)
    # a gold of some documents keeps the others' sections, as a partial draw does
    if only and GOLD_LEDGER.exists():
        kept = json.loads(GOLD_LEDGER.read_text())["sections"]
        ledger["sections"] = {k: v for k, v in kept.items() if k not in ledger["sections"]} | ledger["sections"]
    GOLD_LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
    made = sum(1 for v in ledger["sections"].values() if v["gold"])
    print(
        made,
        "of",
        len(ledger["sections"]),
        "gold sections;",
        {
            w: sum(1 for v in ledger["sections"].values() if v.get("why") == w)
            for w in {v.get("why") for v in ledger["sections"].values() if not v["gold"]}
        },
    )


_PAGE_SECTION = re.compile(r'<section id="([^"]+)" class="ltx_\w+"[^>]*>\s*<h\d[^>]*>(.*?)</h\d>', re.S)
_PAGE_SECTION_TAG = re.compile(r"<(/?)section\b[^>]*>")
_NUMBER_TAG = re.compile(r'<span class="ltx_tag\b[^"]*">.*?</span>', re.S)


# the section element whose heading is the title, subsections and all, without its number, as the LaTeX gold has none
def _page_section(page: str, title: str) -> tuple[str, str] | None:
    heads = list(_PAGE_SECTION.finditer(page))
    found = _best_heading(title, [_folded(_TAG.sub("", _NUMBER_TAG.sub("", m.group(2)))) for m in heads])
    if found is None:
        return None
    start, depth = heads[found[0]].start(), 0
    for m in _PAGE_SECTION_TAG.finditer(page, start):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return heads[found[0]].group(1), _NUMBER_TAG.sub("", page[start : m.end()])
    return None


HTML_LEDGER = GOLD / "html.json"
HTML_FILES = FILES / "html"
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_HTML_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)


# a title kept in front matter, which the page shows and the body does not
def _front_title(text: str) -> str | None:
    front = _FRONT_MATTER.match(text)
    if not front:
        return None
    try:
        title = (yaml.safe_load(front.group(1)) or {}).get("title")
    except yaml.YAMLError:
        return None
    return str(title) if title else None


def _page_path(relative: Path) -> str:
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] in ("index", "_index"):
        parts.pop()
    return "/".join(parts)


def _dashed(title: str) -> str:
    return "-".join(title.split())


# the heading of the file that includes this one, the chapter a section page sits under
def _includer_title(source_dir: Path, relative: Path, fmt: str) -> str | None:
    needle = f"include::{relative.as_posix()}"
    for path in sorted(source_dir.glob(f"*{_SUFFIXES[fmt]}")):
        text = path.read_text(errors="ignore")
        if needle in text:
            heads = _headings(text.splitlines(), fmt)
            return heads[0][2] if heads else None
    return None


def _page_kind(markdown: str) -> str:
    return "table" if _cells(markdown) else "code" if _FENCE.search(markdown) else "prose"


# a source file whole as a page's gold, its title the front matter's or its first heading; None when it has none
def _file_gold(path: Path, source: dict, repo_dir: Path) -> tuple[str, str, str] | None:
    fmt = source["format"]
    text = path.read_text(errors="ignore")
    fragment = _inline_includes(text, path.parent, repo_dir) if fmt == "asciidoc" else text
    if fmt == "asciidoc":
        fragment = _ASCIIDOC_ANCHOR_LINE.sub("", fragment)
    try:
        markdown, readable = _gold_markdown(fragment, source, repo_dir, path.parent)
    except RuntimeError:
        return None
    first = _MD_HEADING.search(markdown)
    title = _front_title(text) or (first.group(2) if first else None)
    if not title:
        return None
    # a page whose source keeps its title in front matter still shows it as its first heading
    if not any(_heading_similarity(title, t) >= HEADING_SIMILARITY for _, t in _heading_list(markdown)):
        markdown = _rebased(f"# {title}\n\n" + markdown)
    return markdown, readable, title


# a site's pages drawn by seed from its source files, each fetched whole with its gold made from the file
def html(only):
    manifest = yaml.safe_load((GOLD / "manifest.yaml").read_text())
    plan = manifest["html"]
    fetched = json.loads(LEDGER.read_text())
    ledger = (
        json.loads(HTML_LEDGER.read_text()) if HTML_LEDGER.exists() else {"sections": [], "deficits": [], "skipped": []}
    )
    drawn = {doc_id for doc_id, _, entry in _documents() if isinstance(entry.get("html"), dict)}
    for key in ("sections", "deficits", "skipped"):
        ledger[key] = [x for x in ledger[key] if x["doc"] not in drawn or (only and x["doc"] not in only)]
    ledger["seed"] = plan["sections"]["seed"]
    HTML_FILES.mkdir(parents=True, exist_ok=True)
    GOLD_MD.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "rag-lab converter benchmark"
    for doc_id, lang, entry in _documents():
        if only and doc_id not in only or not isinstance(entry.get("html"), dict):
            continue
        source = entry["source"]
        repo_dir = FILES / doc_id / lang / "source"
        source_dir = repo_dir / source["path"] if source.get("path") else repo_dir
        fmt = source["format"]
        files = sorted({p for pattern in entry["html"]["files"] for p in source_dir.glob(pattern) if p.is_file()})
        files = [p for p in files if "[" not in p.name]
        random.Random(f"{plan['sections']['seed']}:{doc_id}:{lang}").shuffle(files)
        quota = {
            k: plan["sections"]["per_document"] // len(plan["sections"]["kinds"]) for k in plan["sections"]["kinds"]
        }
        taken = {k: 0 for k in quota}
        for path in files:
            if all(taken[k] >= quota[k] for k in quota):
                break
            relative = path.relative_to(source_dir)
            generated = next(
                (g for g in entry["html"].get("generated", []) if re.search(g, path.read_text(errors="ignore"), re.M)),
                None,
            )
            if generated:
                ledger["skipped"].append(
                    {
                        "doc": doc_id,
                        "lang": lang,
                        "file": str(relative),
                        "why": f"the site builds this page: {generated}",
                    }
                )
                continue
            made = _file_gold(path, source, repo_dir)
            if made is None:
                continue
            markdown, readable, title = made
            kind = _page_kind(markdown)
            if taken[kind] >= quota[kind]:
                continue
            chapter = (
                _includer_title(repo_dir, path.relative_to(repo_dir), fmt)
                if "{chapter}" in entry["html"]["page"]
                else None
            )
            url = entry["html"]["page"].format(
                path=_page_path(relative), title=_dashed(title), chapter=_dashed(chapter or "")
            )
            time.sleep(plan["pause_seconds"])
            try:
                page = session.get(url, timeout=30)
            except requests.RequestException as e:
                ledger["skipped"].append(
                    {"doc": doc_id, "lang": lang, "file": str(relative), "url": url, "why": str(e)[:200]}
                )
                continue
            shown = _HTML_TITLE.search(page.text)
            shown = " ".join(shown.group(1).split()) if shown else ""
            if page.status_code != 200 or _heading_similarity(title, shown) < HEADING_SIMILARITY:
                ledger["skipped"].append(
                    {
                        "doc": doc_id,
                        "lang": lang,
                        "file": str(relative),
                        "url": url,
                        "status": page.status_code,
                        "shown": shown,
                        "why": f"status {page.status_code}" if page.status_code != 200 else "title does not match",
                    }
                )
                continue
            sid = f"{doc_id}_{lang}_html_{kind}_{taken[kind]:02d}"
            taken[kind] += 1
            (HTML_FILES / f"{sid}.html").write_bytes(page.content)
            (GOLD_MD / f"{sid}.md").write_text(markdown)
            ledger["sections"].append(
                {
                    "id": sid,
                    "doc": doc_id,
                    "lang": lang,
                    "kind": kind,
                    "title": title,
                    "url": url,
                    "final_url": page.url,
                    "shown_title": shown,
                    "source_file": str(relative),
                    "source_revision": fetched[f"{doc_id}/{lang}"]["source"]["revision"],
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "html": str((HTML_FILES / f"{sid}.html").relative_to(FILES)),
                    "html_sha256": _sha256(HTML_FILES / f"{sid}.html"),
                    "gold": str((GOLD_MD / f"{sid}.md").relative_to(FILES)),
                    "fragment_sha256": hashlib.sha256(readable.encode()).hexdigest(),
                }
            )
            HTML_LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
        for kind in quota:
            if taken[kind] < quota[kind]:
                ledger["deficits"].append(
                    {"doc": doc_id, "lang": lang, "kind": kind, "short": quota[kind] - taken[kind]}
                )
        HTML_LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
        print(doc_id, lang, taken)


_ANCHOR_TAG = re.compile(r"</?a\b[^>]*>|<span\b[^>]*></span>\s?", re.I)
_SITE_LINK = re.compile(r'<a\b[^>]*href="([^"#?]+)[^"]*"[^>]*>(.*?)</a>', re.S | re.I)


def _html_gold(page: str, drop: list[str]) -> str | None:
    import pypandoc

    main = _main_element(page)
    if main is None:
        return None
    main = _dropped(main, drop)
    # wrappers, empty index spans and link tags go; a table markdown cannot hold stays HTML, as cells read it
    markdown = pypandoc.convert_text(main, "gfm", format="html-native_divs-native_spans", extra_args=["--wrap=none"])
    return _rebased(_ANCHOR_TAG.sub("", markdown))


def _get(session, url: str, pause: float):
    time.sleep(pause)
    return session.get(url, timeout=30)


# a site's pages as the index and each chapter's contents list them, with the link text as the title
def _site_index(session, base: str, pause: float) -> list[dict]:
    from urllib.parse import urljoin

    def links(url):
        page = _get(session, url, pause)
        found = []
        for href, text in _SITE_LINK.findall(page.text):
            target = urljoin(url, href).rstrip("/")
            title = " ".join(_TAG.sub(" ", text).split())
            if target.startswith(base.rstrip("/") + "/") and title:
                found.append({"url": target, "title": title})
        return found

    top = links(base)
    seen = {x["url"]: x for x in top}
    for x in top:
        for y in links(x["url"]):
            seen.setdefault(y["url"], y)
    return list(seen.values())


def _site_page(session, site_dir: Path, url: str, pause: float) -> Path:
    target = site_dir / (url.rstrip("/").rsplit("/", 1)[-1] + ".html")
    if not target.exists():
        page = _get(session, url, pause)
        page.raise_for_status()
        target.write_bytes(page.content)
    return target


# a site that is its own source: PDF sections find pages by title, and the gold is a converter's output, said so
def site(only):
    manifest = yaml.safe_load((GOLD / "manifest.yaml").read_text())
    plan = manifest["html"]
    pause = plan["pause_seconds"]
    drawn_sections = json.loads(SECTIONS.read_text())["sections"]
    ledger = json.loads(HTML_LEDGER.read_text())
    session = requests.Session()
    session.headers["User-Agent"] = "rag-lab converter benchmark"
    for doc_id, lang, entry in _documents():
        if only and doc_id not in only or "html" not in entry["source"]:
            continue
        for key in ("sections", "deficits", "skipped"):
            ledger[key] = [x for x in ledger[key] if (x["doc"], x["lang"]) != (doc_id, lang)]
        site_dir = FILES / doc_id / lang / "site"
        site_dir.mkdir(parents=True, exist_ok=True)
        index_path = site_dir / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        if "pages" not in index:
            index["pages"] = _site_index(session, entry["source"]["html"], pause)
        titles = [_folded(x["title"]) for x in index["pages"]]
        index["pdf_sections"] = {}
        for section in drawn_sections:
            if (section["doc"], section["lang"]) != (doc_id, lang):
                continue
            found = _best_heading(section["title"], titles)
            if found is None:
                continue
            page = index["pages"][found[0]]
            _site_page(session, site_dir, page["url"], pause)
            index["pdf_sections"][section["id"]] = page["url"]
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
        order = list(index["pages"])
        random.Random(f"{plan['sections']['seed']}:{doc_id}:{lang}").shuffle(order)
        quota = {
            k: plan["sections"]["per_document"] // len(plan["sections"]["kinds"]) for k in plan["sections"]["kinds"]
        }
        taken = {k: 0 for k in quota}
        for page in order:
            if all(taken[k] >= quota[k] for k in quota):
                break
            try:
                path = _site_page(session, site_dir, page["url"], pause)
            except requests.RequestException as e:
                ledger["skipped"].append({"doc": doc_id, "lang": lang, "url": page["url"], "why": str(e)[:200]})
                continue
            markdown = _html_gold(path.read_text(errors="ignore"), entry["source"].get("drop", []))
            if markdown is None:
                ledger["skipped"].append({"doc": doc_id, "lang": lang, "url": page["url"], "why": "no DocBook element"})
                continue
            kind = _page_kind(markdown)
            if taken[kind] >= quota[kind]:
                continue
            sid = f"{doc_id}_{lang}_html_{kind}_{taken[kind]:02d}"
            taken[kind] += 1
            (HTML_FILES / f"{sid}.html").write_bytes(path.read_bytes())
            (GOLD_MD / f"{sid}.md").write_text(markdown)
            ledger["sections"].append(
                {
                    "id": sid,
                    "doc": doc_id,
                    "lang": lang,
                    "kind": kind,
                    "title": page["title"],
                    "url": page["url"],
                    "html": str((HTML_FILES / f"{sid}.html").relative_to(FILES)),
                    "html_sha256": _sha256(HTML_FILES / f"{sid}.html"),
                    "gold": str((GOLD_MD / f"{sid}.md").relative_to(FILES)),
                    "gold_from": "the page's own DocBook element by pandoc",
                }
            )
        for kind in quota:
            if taken[kind] < quota[kind]:
                ledger["deficits"].append(
                    {"doc": doc_id, "lang": lang, "kind": kind, "short": quota[kind] - taken[kind]}
                )
        HTML_LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
        print(doc_id, lang, "pages", len(index["pages"]), "pdf sections", len(index["pdf_sections"]), taken)


HTML_MAIN_FILES = FILES / "html_main"


# a site's declarations: where the text is, what inside it is the site's own, which files the site builds itself
def _site_settings(entry: dict) -> dict:
    return entry["html"] if isinstance(entry.get("html"), dict) else entry["source"]


# the prepared arm: each drawn page's own text by its site's selector, highlighting flattened, as a page again
def prepare(only):
    settings = {(doc_id, lang): _site_settings(entry) for doc_id, lang, entry in _documents()}
    ledger = json.loads(HTML_LEDGER.read_text())
    HTML_MAIN_FILES.mkdir(parents=True, exist_ok=True)
    missing = []
    for section in ledger["sections"]:
        if only and section["doc"] not in only:
            continue
        page = (FILES / section["html"]).read_text(errors="ignore")
        site = settings[(section["doc"], section["lang"])]
        main = _element(page, site["main"])
        if main is None:
            missing.append(section["id"])
            section.pop("main_html", None)
            continue
        target = HTML_MAIN_FILES / f"{section['id']}.html"
        body = _flat_pre(_dropped(main, site.get("drop", [])))
        target.write_text(f'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>{body}</body></html>')
        section["main_html"] = str(target.relative_to(FILES))
    HTML_LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
    print(len(ledger["sections"]) - len(missing), "prepared; no main element:", missing)


_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.M | re.S)
_WORD = re.compile(r"\w+")


def _mixed_script(word: str) -> bool:
    from use_cases.route import mixed_script

    return mixed_script(word)


# the similarity reads characters, so typography is folded; its backslashes stay as they were scored
def _plain(markdown: str) -> str:
    from use_cases.raw_quality import plain

    return plain(markdown, fold_typography=True, keep_escapes=True)


# the stand's own reader: `##` and `###` outside fences are headings, anything deeper is content
def _heading_list(markdown: str) -> list[tuple[str, str]]:
    import ingest

    return [(level, ingest._printable(text)) for _, level, text in ingest._heading_marks(markdown)]


def _fence_unbalanced(markdown: str) -> bool:
    import ingest

    return ingest._fence_scan(markdown.split("\n"))[1] is not None


def _cells(markdown: str) -> list[str]:
    # a pipe inside code is the code's, fenced or indented four spaces; a table nested that deep reads as code alike
    markdown = _FENCE.sub("", markdown)
    cells = [
        c
        for line in markdown.splitlines()
        if line.strip().startswith("|") and len(line) - len(line.lstrip()) < 4
        for c in line.strip().strip("|").split("|")
        if not _TABLE_RULE.fullmatch(c)
    ]
    cells += _HTML_CELL.findall(markdown)
    return [" ".join(_TAG.sub(" ", c).split()) for c in cells if c.strip()]


# a whole-book output's section: among the headings like its title, the cut whose text is most like the gold
def _section_of(markdown: str, title: str, gold_md: str, check: dict) -> tuple[str | None, str | None]:
    from rapidfuzz import fuzz

    lines = markdown.splitlines()
    heads = [(i, len(m.group(1)), m.group(2)) for i, line in enumerate(lines) if (m := _MD_HEADING.match(line))]
    # a section ends only at a foreign heading of its level, so a tool with flat levels does not cut it short
    own = [m.group(2) for m in _MD_HEADING.finditer(gold_md)]

    def foreign(text):
        return not any(_heading_similarity(text, t) >= HEADING_SIMILARITY for t in own)

    cuts = []
    for n, (at, level, text) in enumerate(heads):
        if _heading_similarity(text, title) >= HEADING_SIMILARITY:
            end = next((i for i, lv, t in heads[n + 1 :] if lv <= level and foreign(t)), len(lines))
            cuts.append(_rebased("\n".join(lines[at:end])))
    if not cuts:
        return None, None
    gold_text = _plain(gold_md)
    ranked = sorted(((fuzz.ratio(gold_text, _plain(cut)), cut) for cut in cuts), key=lambda x: -x[0])
    note = None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < check["identification_band"]:
        note = f"two cuts within {check['identification_band']} of the gold: {ranked[0][0]:.1f} and {ranked[1][0]:.1f}"
    return ranked[0][1], note


def _score(gold_md: str, out_md: str) -> dict:
    from collections import Counter

    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein

    gh, oh = _heading_list(gold_md), _heading_list(out_md)
    # each gold heading takes the most alike free output heading, and each is taken once
    free = list(range(len(oh)))
    exact = level_wrong = 0
    for lv, t in gh:
        scored = [(_heading_similarity(oh[i][1], t), i) for i in free]
        best = max(scored, default=(0, None))
        if best[0] < HEADING_SIMILARITY:
            continue
        free.remove(best[1])
        exact += oh[best[1]][0] == lv
        level_wrong += oh[best[1]][0] != lv
    lost = len(gh) - exact - level_wrong
    phantoms = len(free)
    gold_code, out_code = _FENCE.findall(gold_md), _FENCE.findall(out_md)

    def squash(block):
        return " ".join(block.split())

    code_exact = sum(1 for b in gold_code if b in out_code)
    code_normal = sum(1 for b in gold_code if squash(b) in {squash(o) for o in out_code})
    # a block split or merged at a page break is still there when read against all fences in order
    joined = squash(" ".join(out_code))
    code_contained = sum(1 for b in gold_code if squash(b) in joined)
    gold_cells, out_cells = Counter(_cells(gold_md)), Counter(_cells(out_md))
    gold_text, out_text = _plain(gold_md), _plain(out_md)
    gold_words, out_words = Counter(gold_text.lower().split()), Counter(out_text.lower().split())
    common = sum((gold_words & out_words).values())
    precision = common / max(1, sum(out_words.values()))
    recall = common / max(1, sum(gold_words.values()))
    words = [w for w in _WORD.findall(out_text) if any(c.isalpha() for c in w)]
    gold_words_list = [w for w in _WORD.findall(gold_text) if any(c.isalpha() for c in w)]
    similarity = round(fuzz.ratio(gold_text, out_text), 2)
    return {
        "headings": len(gh),
        "heading_exact": exact,
        "heading_level_wrong": level_wrong,
        "heading_lost": lost,
        "phantoms": phantoms,
        "fence_unbalanced": _fence_unbalanced(out_md),
        "code_blocks": len(gold_code),
        "code_exact": code_exact,
        "code_whitespace": code_normal,
        "code_contained": code_contained,
        "table_cells": sum(gold_cells.values()),
        "table_cells_found": sum((gold_cells & out_cells).values()),
        "body_f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
        "cer": round(Levenshtein.distance(gold_text, out_text) / max(1, len(gold_text)), 4),
        "mixed_script_words": sum(1 for w in words if _mixed_script(w)),
        "words": len(words),
        "gold_mixed_script_words": sum(1 for w in gold_words_list if _mixed_script(w)),
        "similarity": similarity,
        "recovered": similarity >= RECOVERED_SIMILARITY,
        "length_ratio": round(len(out_text) / max(1, len(gold_text)), 2),
    }


HEADING_READER = (
    "ingest._heading_marks on both sides: ## and ### outside fences, deeper is content; each gold "
    f"heading takes the most alike free output heading at WRatio {HEADING_SIMILARITY} or more"
)


# every drawn section a run converted, scored against its gold; the run's folder gets score.json
def score(only):
    sections = {x["id"]: x for x in json.loads(SECTIONS.read_text())["sections"]}
    golds = json.loads(GOLD_LEDGER.read_text())["sections"]
    # a site's page is whole on both sides, so its gold stands beside the drawn sections' under its own id
    if HTML_LEDGER.exists():
        pages = json.loads(HTML_LEDGER.read_text())["sections"]
        sections |= {x["id"]: x for x in pages}
        golds |= {x["id"]: {"gold": x["gold"]} for x in pages}
    declared = yaml.safe_load((GOLD / "manifest.yaml").read_text())
    for run in sorted((FILES / "runs").iterdir()):
        if only and run.name not in only or not (run / "record.json").exists():
            continue
        rows = []
        for key in json.loads((run / "record.json").read_text())["converted"]:
            out_path = run / (key.replace("/", "__") + ".md")
            if not out_path.exists():
                continue
            output = out_path.read_text()
            stem = Path(key).stem
            # a text-layer cut holds whole pages, so its section is found inside it as in a whole book
            if key.startswith("sections/") and golds.get(stem, {}).get("gold"):
                cut, note = _section_of(
                    output, sections[stem]["title"], (FILES / golds[stem]["gold"]).read_text(), declared["score"]
                )
                targets = [(stem, cut, "text_pdf", note)]
            elif stem in sections:
                targets = [(stem, output, key.split("/")[0] if key.startswith("html") else key.split("/")[1], None)]
            else:
                targets = []
                for sid, s in sections.items():
                    gold = golds.get(sid, {}).get("gold")
                    if s.get("pdf") != key or not gold:
                        continue
                    cut, note = _section_of(output, s["title"], (FILES / gold).read_text(), declared["score"])
                    targets.append((sid, cut, "text_pdf", note))
            for sid, out_md, point, note in targets:
                gold = golds.get(sid, {}).get("gold")
                if not gold:
                    continue
                s = sections[sid]
                row = {
                    "id": sid,
                    "doc": s["doc"],
                    "lang": s["lang"],
                    "kind": s["kind"],
                    "point": point,
                    "identification_note": note,
                }
                if out_md is None:
                    rows.append({**row, "found": False})
                    continue
                rows.append({**row, "found": True, **_score((FILES / gold).read_text(), out_md)})
        # a cut far longer than its gold lost its closing heading: listed as an overrun, kept out of the means
        for r in rows:
            r["overrun"] = r["found"] and r["length_ratio"] > declared["score"]["overrun_ratio"]
        # a section two cuts claim alike is not read, only listed
        found = [r for r in rows if r["found"] and not r["identification_note"] and not r["overrun"]]
        t = {
            k: sum(r[k] for r in found)
            for k in (
                "heading_exact",
                "headings",
                "heading_level_wrong",
                "heading_lost",
                "phantoms",
                "fence_unbalanced",
                "code_exact",
                "code_blocks",
                "code_whitespace",
                "code_contained",
                "table_cells_found",
                "table_cells",
                "mixed_script_words",
                "gold_mixed_script_words",
                "words",
            )
        }
        # a mean arrives with its n, so a quoted number carries its denominator
        means = {
            k: {"mean": round(sum(r[k] for r in found) / max(1, len(found)), 4), "n": len(found)}
            for k in ("body_f1", "cer", "similarity")
        }
        # the headline: of the sections with a gold, the share that came through at the declared similarity
        recovered = {
            "recovered": sum(1 for r in found if r["recovered"]),
            "of": len(rows),
            "similarity": RECOVERED_SIMILARITY,
            "at": {cut: sum(1 for r in found if r["similarity"] >= cut) for cut in _SCORE["recovered_cuts"]},
        }
        doubtful = [r["id"] for r in rows if r["found"] and r["identification_note"]]
        overrun = [r["id"] for r in rows if r["overrun"] and not r["identification_note"]]
        (run / "score.json").write_text(
            json.dumps(
                {
                    "heading_reader": HEADING_READER,
                    "recovered": recovered,
                    "totals": t,
                    "means": means,
                    "sections_found": len(found),
                    "doubtful": doubtful,
                    "overrun": overrun,
                    "sections": len(rows),
                    "rows": rows,
                },
                indent=1,
                ensure_ascii=False,
            )
        )
        print(
            run.name,
            f"recovered of {recovered['of']}:",
            recovered["at"],
            ";",
            f"{len(found)}/{len(rows)} sections read, {len(doubtful)} doubtful, {len(overrun)} overrun;",
            f"headings exact {t['heading_exact']}/{t['headings']}, level wrong {t['heading_level_wrong']},",
            f"lost {t['heading_lost']}, fences unbalanced {t['fence_unbalanced']},",
            f"phantoms {t['phantoms']}; code exact {t['code_exact']}/{t['code_blocks']},",
            f"whitespace {t['code_whitespace']}, contained {t['code_contained']};",
            f"cells {t['table_cells_found']}/{t['table_cells']};",
            f"body f1 {sum(r['body_f1'] for r in found) / max(1, len(found)):.3f},",
            f"cer {sum(r['cer'] for r in found) / max(1, len(found)):.3f},",
            f"similarity {sum(r['similarity'] for r in found) / max(1, len(found)):.1f},",
            f"mixed-script words {t['mixed_script_words']}/{t['words']} (gold {t['gold_mixed_script_words']})",
        )


# whether the stand can cut each output: the corpus variant's own cut, its metrics and gates, no database
def gates(only):
    import config
    from use_cases.raw_quality import chunker_gates

    variant = config.settings.corpus.variant
    for run in sorted((FILES / "runs").iterdir()):
        if only and run.name not in only or not (run / "record.json").exists():
            continue
        outputs = {
            path.name: chunker_gates(path.read_text(encoding="utf-8", errors="ignore"), path.name)
            for path in sorted(run.glob("*.md"))
        }
        # the second headline: the share of outputs the stand can cut, a broken verdict being the one that cannot
        cuttable = sum(1 for o in outputs.values() if o["verdict"] in ("ok", "dirty"))
        (run / "gates.json").write_text(
            json.dumps(
                {"variant": variant, "cuttable": cuttable, "of": len(outputs), "outputs": outputs},
                indent=1,
                ensure_ascii=False,
            )
        )
        print(
            run.name,
            f"cuttable {cuttable}/{len(outputs)};",
            {
                name: (o["verdict"], o["hard"] + o["soft"], o["metrics"]["section_coverage"])
                for name, o in outputs.items()
            },
        )


# what a good text layer looks like page by page, the reading the intake's route thresholds are tuned on
def layer_band(only):
    from use_cases.route import page_signals

    band = {}
    for doc_id, lang, entry in _documents():
        if only and doc_id not in only or "text_pdf" not in entry:
            continue
        pdf = next((FILES / doc_id / lang / "pdf").rglob("*.pdf"), None)
        if pdf is None:
            continue
        signals = page_signals(pdf)
        chars = sorted(s["layer_chars"] for s in signals)
        band[f"{doc_id}/{lang}"] = {
            "pages": len(signals),
            "under_50_chars": sum(c < 50 for c in chars),
            "chars_p5": chars[len(chars) // 20],
            "mixed_script_max": max(
                (s["mixed_script"] for s in signals if s["mixed_script"] is not None), default=None
            ),
            "once_in_file_max": max(
                (s["once_in_file"] for s in signals if s["once_in_file"] is not None), default=None
            ),
        }
        print(doc_id, lang, band[f"{doc_id}/{lang}"])
    (GOLD / "layer_band.json").write_text(json.dumps(band, indent=2, ensure_ascii=False))


# how far a good conversion sits from the file's own text layer, the band the intake's quality gates are tuned on
def raw_band(only):
    from use_cases.raw_quality import conversion_signals
    from use_cases.route import layer_texts

    band = {}
    for run in ("docling_no_code_enrichment_en_cuts", "docling_no_code_enrichment_ru_cuts"):
        rows = []
        for md in sorted((FILES / "runs" / run).glob("sections__*.md")):
            pdf = FILES / "sections" / md.name.removeprefix("sections__").removesuffix(".md")
            rows.append({"cut": pdf.name, **conversion_signals(md.read_text(), "\n".join(layer_texts(pdf)))})
        band[run] = {
            key: sorted(r[key] for r in rows if r[key] is not None)
            for key in ("layer_f1", "output_share", "mixed_script")
        }
        print(run, {k: (v[0], v[len(v) // 2], v[-1]) for k, v in band[run].items()})
    (GOLD / "raw_band.json").write_text(json.dumps(band, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "fetch",
            "pairs",
            "smoke",
            "draw",
            "cut",
            "raster",
            "gold",
            "html",
            "site",
            "prepare",
            "layer_band",
            "raw_band",
            "score",
            "gates",
        ],
    )
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    commands = {
        "fetch": fetch,
        "pairs": pairs,
        "smoke": smoke,
        "draw": draw,
        "cut": cut,
        "raster": raster,
        "gold": gold,
        "html": html,
        "site": site,
        "prepare": prepare,
        "layer_band": layer_band,
        "raw_band": raw_band,
        "score": score,
        "gates": gates,
    }
    commands[args.command](set(args.only))


if __name__ == "__main__":
    main()
