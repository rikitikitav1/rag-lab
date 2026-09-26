import hashlib
import json
import os

import config
import yaml
from sources.declaration import SourceFile

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


# a technology no row of the map names would become a value of the field nobody can ask for
def _refuse_unmapped(found: dict[str, SourceFile]) -> None:
    mapped = set(config.settings.technologies)
    for source in found.values():
        named = {*source.technologies, *source.technology_by_path.values()}
        if unknown := sorted(named - mapped):
            raise ValueError(f"{source.name}: {unknown} are not rows of config/technologies.yaml")


# the map's rows no source covers yet: the coverage map's empty cells
def empty_rows(found: dict | None = None) -> list[str]:
    covered = {t for s in (found or source_files()).values() for t in (*s.technologies, *s.technology_by_path.values())}
    return sorted(set(config.settings.technologies) - covered)


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
CUT_RULES = ("name", "language", "folder", "git", "git_family", "reader", "categories", "skip", "drop_docs_containing")


# over the cut's rules in the loaded file, not its bytes: a comment or metadata beside the rules moves no row
def digest(source: SourceFile) -> str:
    rules = source.model_dump(mode="json", include=set(CUT_RULES))
    rules["skip_when_hygienic"] = sorted(source.skip_when_hygienic)
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:12]


def rows_of(source: SourceFile) -> list[str]:
    return list(source.git_family.repos) if source.git_family is not None else [source.name]


# a database row whole as its file declares it; the seed and the index write it alike, and the file wins
def row_of(source: SourceFile, name: str) -> dict:
    common = {"name": name, "language": source.language, "licence": source.licence}
    if source.folder is not None:
        return {**common, "kind": "local", "git_url": None, "path": source.folder}
    url = source.git.repo if source.git is not None else f"{source.git_family.base_url}/{name}"
    return {**common, "kind": "git", "git_url": url, "path": None}


def file_of_row(row_name: str, found: dict | None = None) -> SourceFile | None:
    return next((s for s in (found or source_files()).values() if row_name in rows_of(s)), None)


# the variants cut by another version of the row's file; a row whose file is gone says so, a declared one has none
def drift(row_name: str, indexed_with: dict, found: dict | None = None) -> dict | None:
    source = file_of_row(row_name, found)
    if source is None:
        return {"file": None, "moved": "file gone", "indexed_with": indexed_with} if indexed_with else None
    now = digest(source)
    moved = sorted(v for v, was in (indexed_with or {}).items() if was != now)
    return {
        "file": f"{SOURCES_DIR}/{source.name}.yaml",
        "digest": now,
        "indexed_with": indexed_with or {},
        "moved": moved,
    }


# the preflight's reading over every row, files read once: moved files with their variants, orphans, rows cut unrecorded
def drift_report(rows: list[tuple[str, dict]]) -> dict:
    found = source_files()
    moved: dict[str, set[str]] = {}
    orphaned, unrecorded = [], 0
    for name, indexed_with in rows:
        said = drift(name, indexed_with, found)
        if said is None:
            continue
        if said["file"] is None:
            orphaned.append(name)
        elif not said["indexed_with"]:
            unrecorded += 1
        elif said["moved"]:
            moved.setdefault(said["file"], set()).update(said["moved"])
    return {
        "moved": {f: sorted(v) for f, v in sorted(moved.items())},
        "orphaned": sorted(orphaned),
        "unrecorded": unrecorded,
    }


# the rows whose file says they drift on their own: a family's rows by their repository names
def drifting_rows() -> list[str]:
    return sorted(row for s in source_files().values() if s.drifts for row in rows_of(s))
