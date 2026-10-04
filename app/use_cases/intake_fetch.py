import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import corpus_keys
import logging_setup
import requests
from errors import Final
from models.corpus import DataSource
from sources.declaration import DEFAULT_INCLUDE, GitFamily, site_of, skipped_path
from use_cases import fetch, route, site_page

log = logging_setup.get_logger(__name__)

# a sitemap that lists a page the site removed: that page is left out, the source is not
GONE = (404, 410)

# what a file name keeps of a url's last step
_SLUG = re.compile(r"[^\w.-]+")


# a url to a file under a folder of its own hash, so two index.html stay apart and keep their suffix for the route
def _download(url: str, folder: Path) -> tuple[Path, bool]:
    target = folder / corpus_keys.short_hash(url) / (_SLUG.sub("_", url.rstrip("/").rsplit("/", 1)[-1]) or "index.html")
    return target, fetch.download(url, target)


def _clone(git: dict, folder: Path) -> tuple[Path, dict]:
    if fetch.inside(folder, git.get("path")) is None:
        raise Final(f"{git.get('path')}: not a folder of the repository")
    # a repeated intake reads the upstream as it is now; the gold keeps its clone as it came, for a stable arm
    state = fetch.clone(git["repo"], folder, git.get("ref"), git.get("path"), update=True)
    return fetch.inside(folder, git.get("path")), state


# a folder origin read at the door as its job would read it, so a typo fails before the job's turn in the queue
def stand_folder(folder: str, stand: Path) -> tuple[Path, list[Path]]:
    root = (stand / folder).resolve()
    if stand.resolve() not in root.parents or not root.is_dir():
        raise Final(f"{folder}: not a folder of the stand")
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    if not files:
        raise Final(f"{folder}: a folder with no files")
    return root, files


# what an intake fetched: the root the route reads, its files, the fetch state, and the release a site's page said
class Gathered(NamedTuple):
    root: Path
    files: list[Path]
    fetched: dict
    release: str | None = None
    # pages the sitemap lists and the site no longer serves, by url, with the answer it gave
    gone: dict = {}


# the files of a declared source as the route will read them, fetched into its inbox when they come from outside
def gather(source: DataSource, inbox: Path, stand: Path) -> Gathered:
    origin = source.declaration or {}
    inbox.mkdir(parents=True, exist_ok=True)
    if "git_family" in origin:
        family = GitFamily.model_validate(origin["git_family"])
        origin = {**origin, "git": {"repo": family.repo_of(source.name), "include": family.include}}
    if "folder" in origin:
        return Gathered(*stand_folder(origin["folder"], stand), {})
    if "urls" in origin:
        got = [_download(url, inbox) for url in origin["urls"]]
        # an archive is the files inside it, as the gold's fetch reads it
        files = [p for f, _ in got for p in (fetch.unzip(f) if f.suffix.lower() == ".zip" else [f])]
        return Gathered(inbox, files, _fetched(any(new for _, new in got)))
    if "git" in origin:
        root, state = _clone(origin["git"], inbox / "repo")
        # a repeated intake reads the upstream anew, so a clone is always fetched now
        state = {**state, **_fetched(True)}
        found = {p.resolve() for pattern in origin["git"].get("include", DEFAULT_INCLUDE) for p in root.glob(pattern)}
        return Gathered(root, sorted(p for p in found if p.is_file() and root in p.parents), state)
    site = site_of(origin)
    release = site.release if site else None
    read = None
    if site and site.release_page:
        live = _live_release(site, inbox)
        if live != release:
            _refuse_unlisted_release(origin, live)
            release = read = live
    _drop_pages_of_another_release(inbox, release)
    files, fresh, gone = [], False, {}
    for url in origin.get("pages") or _sitemap_pages(site, inbox):
        try:
            page, new = _download(url, inbox / "pages")
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code not in GONE:
                raise
            gone[url] = f"the site answers {e.response.status_code} for a page its sitemap lists"
            continue
        fresh |= new
        main = site_page.prepared(page.read_text(errors="ignore"), site.main, site.drop) if site else None
        if main is None:
            files.append(page)
            continue
        target = inbox / "main" / page.parent.name / page.with_suffix(".html").name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(main)
        files.append(target)
    if release:
        (inbox / "release").write_text(release)
    if gone:
        log.warning("intake.pages_gone", source=source.name, n=len(gone))
    return Gathered(inbox, files, _fetched(fresh), read, gone)


# the release the site documents now, read from its own page every intake
def _live_release(site, inbox: Path) -> str:
    target = inbox / "release_page.html"
    target.unlink(missing_ok=True)
    fetch.download(site.release_page, target)
    found = re.search(site.release_pattern, target.read_text(errors="ignore"))
    if found is None:
        raise Final(f"{site.release_page}: no release matches {site.release_pattern}")
    return found.group(1)


# a newer release is a version of the source's category, so the category must list it before its pages are fetched
def _refuse_unlisted_release(origin: dict, live: str) -> None:
    import config

    categories = origin.get("categories") or []
    listed = config.settings.categories[categories[0]].versions if len(categories) == 1 else []
    if live not in listed:
        raise Final(f"{origin.get('name')}: the site documents {live} now; list it for {categories} and intake again")


# the site's pages as its sitemap lists them now, filtered by the declaration; the sitemap is read anew every intake
def _sitemap_pages(site, inbox: Path) -> list[str]:
    target = inbox / "sitemap.xml"
    target.unlink(missing_ok=True)
    fetch.download(site.sitemap, target)
    found = site_page.sitemap_urls(target.read_text(errors="ignore"), site.include, site.exclude)
    if not found:
        raise Final(f"{site.sitemap}: the sitemap lists no page the declaration keeps")
    return found


# pages kept from another release, or from before one was declared, would take its name unread: they are fetched anew
def _drop_pages_of_another_release(inbox: Path, release: str | None) -> None:
    stamp = inbox / "release"
    if not release or (stamp.is_file() and stamp.read_text() == release):
        return
    for folder in ("pages", "main", "flat"):
        shutil.rmtree(inbox / folder, ignore_errors=True)


# an HTML file with its highlighting flat, beside the fetched one, for any HTML and not only a site's own element
def _flat(file: Path, inbox: Path, rel: str) -> Path:
    if file.suffix.lower() not in route.HTML:
        return file
    text = file.read_text(errors="ignore")
    flat = site_page.flat_pre(text)
    if flat == text:
        return file
    target = inbox / "flat" / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(flat)
    return target


# every file under its report name, and what was left out; an EPUB stands for its chapters when the source reads them
def named_files(
    root: Path, files: list[Path], inbox: Path, skip=frozenset(), epub: bool = False, generated: list[str] = (),
    skip_paths: list[str] = (),
) -> tuple[dict[Path, str], dict[str, str]]:
    names, skipped = {}, {}
    # a docs repository's screenshots are its pages' figures: OCR of nine thousand of them read nothing worth a search
    beside_documents = any(f.suffix.lower() not in route.IMAGE and f.suffix.lower() in route.READABLE for f in files)
    for file in files:
        rel = str(file.relative_to(root))
        if beside_documents and file.suffix.lower() in route.IMAGE:
            skipped[rel] = "a picture beside documents: a page's figure, not a page"
            continue
        if pattern := skipped_path(rel, skip_paths):
            skipped[rel] = f"the declaration skips {pattern}"
            continue
        page = generated and file.suffix.lower() in route.HTML
        if page and (built := site_page.generated_by(file.read_text(errors="ignore"), generated)):
            skipped[rel] = f"the site builds this page: {built}"
            continue
        if file.suffix.lower() != ".epub":
            names[_flat(file, inbox, rel)] = rel
            continue
        if not epub:
            skipped[rel] = "EPUB reading is off until a second EPUB is measured (knob epub_chapters)"
            continue
        chapters, left_out = fetch.epub_chapters(file, inbox / "epub" / rel, skip)
        names |= {_flat(chapter, inbox, f"{rel}/{chapter.name}"): f"{rel}/{chapter.name}" for chapter in chapters}
        skipped |= {f"{rel}/{name}": why for name, why in left_out.items()}
    return names, skipped


# a fetch is stamped when something was fetched; a resume that found every file on disk keeps the earlier stamp
def _fetched(fresh: bool) -> dict:
    return {"fetched_at": datetime.now(UTC).isoformat(timespec="seconds")} if fresh else {}


# the source's language before any piece is read, since a scan's OCR is told it: declared, else from its own text
def source_language(source: DataSource, files: list[Path], layers: dict) -> str:
    # the row's column holds what an earlier run found; only the declaration says what was declared
    if declared := (source.declaration or {}).get("language"):
        return declared
    text = []
    for file in files:
        if file.suffix.lower() == ".pdf":
            # a file the reader cannot open is marked in the report, not a reason to stop before it
            try:
                layers.setdefault(file, route.layer_texts(file))
            except route.Unreadable:
                continue
            text += layers[file]
        elif file.suffix.lower() in route.MARKDOWN | route.HTML:
            text.append(file.read_text(errors="ignore"))
    joined = "".join(text)
    if not any(c.isalpha() for c in joined):
        raise Final(f"{source.name}: no text layer to read its language from; declare its language")
    return corpus_keys.language_by_alphabet(joined)
