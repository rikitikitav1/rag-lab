import hashlib
import json
import os

import config
import yaml
from sources.declaration import KINDS, Declaration, SourceFile

SOURCES_DIR = "sources"


# every worked-out source by its name, read once per call; no folder, or an empty one, refuses: nothing to cut
def source_files() -> dict[str, SourceFile]:
    paths = config.yaml_files(config.beside_config(SOURCES_DIR))
    if not paths:
        raise FileNotFoundError(f"no source files under {config.beside_config(SOURCES_DIR)}: the corpus would be empty")
    found, rows = {}, {}
    for path in paths:
        with open(path) as f:
            source = SourceFile(**(yaml.safe_load(f) or {}))
        if source.name != os.path.basename(path).removesuffix(".yaml"):
            raise ValueError(f"{path}: holds the source {source.name}, the file must be named after it")
        for row in rows_of(source):
            if row in rows:
                raise ValueError(f"{path}: the row {row} is already declared by {rows[row]}")
            rows[row] = source.name
        found[source.name] = source
    _refuse_unmapped(found)
    return found


# a category no row of the map names would become a value of the field nobody can ask for
def _refuse_unmapped(found: dict[str, SourceFile]) -> None:
    for source in found.values():
        if refusal := index_refusal(source):
            raise ValueError(refusal)


# what the index could not read, for the seed and both doors alike: an unknown reader, an unmapped category, a version
def index_refusal(source: Declaration) -> str | None:
    from sources import factory

    if refusal := factory.reader_refusal(source):
        return refusal
    named = {*source.categories, *source.category_by_path.values()}
    if unknown := sorted(named - set(config.settings.categories)):
        return f"{source.name}: {unknown} are not rows of config/categories.yaml"
    if len(source.categories) > 1 and not source.category_by_path:
        return f"{source.name}: several categories need a category_by_path to say which files are which"
    # a release read from the site's page is listed under its one category: refused here, not after a fetch in the queue
    if source.site and source.site.release_page and (len(source.categories) != 1 or source.category_by_path):
        return f"{source.name}: a site that reads its release from its page names exactly one category"
    if source.site and source.site.release:
        return _unlisted(source, [source.site.release], "release")
    return _unlisted(source, list(source.versions), "versions") if source.versions else None


# versions or a release are one category's, newest first as the map lists them, the newest held: search reads it alone
def _unlisted(source: Declaration, named: list[str], what: str) -> str | None:
    if len(source.categories) != 1 or source.category_by_path:
        return f"{source.name}: a source with {what} names exactly one category"
    category = source.categories[0]
    listed = config.settings.categories[category].versions
    if unknown := [v for v in named if v not in listed]:
        return f"{source.name}: {what} {' '.join(unknown)} is not listed for {category}"
    if named != [v for v in listed if v in named]:
        return f"{source.name}: {what} must go newest first as the map lists them, {listed}"
    if listed[0] not in named:
        return f"{source.name}: {what} lack {category}'s newest {listed[0]}; a book for any version has none"
    return None


# the map's rows no source covers yet: the coverage map's empty cells
def empty_rows(found: dict | None = None) -> list[str]:
    covered = {c for s in (found or source_files()).values() for c in (*s.categories, *s.category_by_path.values())}
    return sorted(set(config.settings.categories) - covered)


# the file a code source's reader parses; two files naming one reader would make the class ambiguous
def of_reader(reader: str, found: dict | None = None) -> SourceFile:
    named = [s for s in (found or source_files()).values() if s.reader == reader]
    if len(named) != 1:
        raise LookupError(f"reader {reader}: {len(named)} source files name it, one must")
    return named[0]


# the veto build's families over every file; a family the quotas do not count, or the reverse, refuses
def veto_families(found: dict | None = None) -> dict[str, str]:
    named = {f.name: f.prefix for s in (found or source_files()).values() for f in s.veto_families}
    counted = set(config.settings.evals.veto.quotas)
    if set(named) != counted:
        raise ValueError(f"veto families {sorted(named)} and quotas {sorted(counted)} must name the same set")
    return named


# the fields that decide the cut; the licence, the veto families, the drift flag and a skip's reason do not
CUT_RULES = (
    "name",
    "language",
    "folder",
    "git",
    "git_family",
    "reader",
    "categories",
    "category_by_path",
    "versions",
    "skip",
    "skip_paths",
    "drop_docs_containing",
)


# over the cut's rules in the loaded file, not its bytes: a comment or metadata beside the rules moves no row
def digest(source: SourceFile) -> str:
    rules = source.model_dump(mode="json", include=set(CUT_RULES))
    rules["skip_when_hygienic"] = sorted(source.skip_when_hygienic)
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:12]


def rows_of(source: SourceFile) -> list[str]:
    return list(source.git_family.repos) if source.git_family is not None else [source.name]


# a language or licence a declaration leaves out keeps what onboarding found or the door wrote
UNSAID_KEPT = ("language", "licence")


# a database row whole as its declaration says it; the doors, the seed and the index build it here alike
def row_of(source: Declaration, name: str) -> dict:
    kind = KINDS[source.origin]
    common = {"name": name, "language": source.language, "licence": source.licence, "kind": kind}
    if source.folder is not None:
        return {**common, "git_url": None, "path": source.folder}
    if source.urls or source.pages or source.site:
        return {**common, "git_url": None, "path": None}
    url = source.git.repo if source.git is not None else source.git_family.repo_of(name)
    return {**common, "git_url": url, "path": None}


# what a variant's digest reads once the run it was cut from is replaced: no declaration's digest equals it
RUN_REPLACED = "run replaced"


# the one upsert rule of the seed and the index: the declaration wins, save what it leaves unsaid
def upserted(insert, table, columns) -> dict:
    from sqlalchemy import func

    kept = {k: func.coalesce(insert.excluded[k], table.c[k]) for k in columns if k in UNSAID_KEPT}
    return {k: kept.get(k, insert.excluded[k]) for k in columns if k != "name"}


# a row's declaration moved since a variant was cut by it; a row indexed with no declaration left says so
def drift(row_name: str, indexed_with: dict, declaration: dict | None) -> dict | None:
    if not declaration:
        return {"source": None, "moved": "declaration gone", "indexed_with": indexed_with} if indexed_with else None
    source = Declaration.model_validate(declaration)
    now = digest(source)
    moved = sorted(v for v, was in (indexed_with or {}).items() if was != now)
    return {"source": source.name, "digest": now, "indexed_with": indexed_with or {}, "moved": moved}


# the preflight's reading over every row: moved declarations with their variants, orphans, rows cut unrecorded
def drift_report(rows: list[tuple[str, dict, dict | None]]) -> dict:
    moved: dict[str, set[str]] = {}
    orphaned, unrecorded = [], 0
    for name, indexed_with, declaration in rows:
        said = drift(name, indexed_with, declaration)
        if said is None:
            continue
        if said["source"] is None:
            orphaned.append(name)
        elif not said["indexed_with"]:
            unrecorded += 1
        elif said["moved"]:
            moved.setdefault(said["source"], set()).update(said["moved"])
    return {
        "moved": {f: sorted(v) for f, v in sorted(moved.items())},
        "orphaned": sorted(orphaned),
        "unrecorded": unrecorded,
    }

