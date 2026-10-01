import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
import logging_setup
from sources import (  # noqa: F401
    arangodb_docs,
    cheatsheets,
    converted,
    files,
    interview,
    redis_docs,
    system_design_primer,
)
from sources.base import Base
from sources.declaration import Declaration

log = logging_setup.get_logger(__name__)


# every accepted repository row, one organisation's family unrolled, one checkout per released version
def _git_specs(found, rows) -> list[tuple[str, str, object, str | None]]:
    specs = []
    for source in found:
        if source.folder is not None or not (source.git or source.git_family):
            continue
        if source.versions:
            url = files.row_of(source, source.name)["git_url"]
            specs += [(f"{source.name}@{v}", url, source, origin.ref) for v, origin in source.versions.items()]
        else:
            named = [n for n in files.rows_of(source) if n in rows]
            specs += [(name, files.row_of(source, name)["git_url"], source, None) for name in named]
    return specs


# a reader named in a declaration and known to no class would parse nothing its declaration meant
def reader_refusal(source) -> str | None:
    if source.reader is not None and source.reader not in Base._registry:
        return f"{source.name}: reader {source.reader} is no class; known: {sorted(Base._registry)}"
    return None


def _reader(source) -> type[Base]:
    if refusal := reader_refusal(source):
        raise LookupError(refusal)
    return Base._registry.get(source.reader, Base)


# where onboarding said a row is read: its own tree, or the raw folder a converter wrote; an older row names its folder
def _onboarded(rows: dict) -> dict[str, tuple[Path, bool, str | None]]:
    out = {}
    for name, info in rows.items():
        raw = info["raw"]
        # the release the accepted run fetched is the one its pages document; the declaration only asks for one
        release = (raw.get("version") or {}).get("release")
        if raw.get("root"):
            out[name] = (Path(raw["root"]), raw.get("root_kind") == "converted", release)
        elif raw.get("folder"):
            out[name] = (Path(raw["folder"]), True, release)
    return out


# the accepted rows by name, a page of them or the ones named, each under the declaration it carries
def _accepted(names=None, limit=None, offset=0) -> tuple[dict, dict]:
    from models.corpus import DataSource, Stage
    from orm.sync_db import Session
    from sqlalchemy import select

    stmt = select(DataSource).where(DataSource.stage == Stage.accepted).order_by(DataSource.name).offset(offset)
    if names is not None:
        stmt = stmt.where(DataSource.name.in_(names))
    if limit is not None:
        stmt = stmt.limit(limit)
    found, rows = {}, {}
    with Session() as session:
        for row in session.scalars(stmt):
            # a row the seed has not declared yet reads as nothing, said, rather than as an empty source
            if not row.declaration:
                log.warning("sources.undeclared", row=row.name)
                continue
            declaration = found.setdefault(row.declaration["name"], Declaration.model_validate(row.declaration))
            rows[row.name] = {"source": declaration.name, "raw": row.raw or {}}
    return found, rows


# the index's sources from the accepted rows, all, a page or the ones named: converted, local folders, repositories
def sources(names=None, limit=None, offset=0):
    from sources.versioned import Versioned

    declared, rows = _accepted(names, limit, offset)
    # a versioned source is read a version a folder, which one onboarded root cannot give; it is cloned as before
    raw = {n: v for n, v in _onboarded(rows).items() if not declared[rows[n]["source"]].versions}
    # an onboarded row is read where onboarding said; the rest of a family's rows are cloned as before
    rest = {name: info for name, info in rows.items() if name not in raw}
    found = [s for s in declared.values() if any(info["source"] == s.name for info in rest.values())]
    for source in found:
        if not (source.folder or source.git or source.git_family):
            log.warning("sources.unreadable", source=source.name, why="no folder or repository and no converted folder")
    found = [s for s in found if s.folder or s.git or s.git_family]
    local = [s for s in found if s.folder is not None]
    specs = _git_specs(found, rest)
    readers = {s.name: _reader(s) for s in found}
    log.info("sources.gather", onboarded=len(raw), local=len(local), git=len(specs))
    roots = provision([(name, url, ref) for name, url, _, ref in specs])
    built = 0
    for name, (root, is_converted, release) in raw.items():
        declaration = declared[rows[name]["source"]]
        reader = Base._registry["converted"] if is_converted else _reader(declaration)
        built += 1
        # a run from before the release was stamped on it read the declared one
        release = release or (declaration.site.release if declaration.site else None)
        yield reader(root, declaration, name=name, onboarded=True, version=release)
    for source in local:
        built += 1
        if source.versions:
            yield Versioned(
                [readers[source.name](Path(o.folder), source, version=v) for v, o in source.versions.items()]
            )
        else:
            yield readers[source.name](Path(source.folder), source)
    versioned = {}
    for name, _, source, _ in sorted(specs, key=lambda spec: spec[2].git_family is not None):
        if not roots[name] and source.versions:
            # a declared version that did not arrive would read later as the asker's «no such version»
            raise LookupError(f"{source.name}: version {name.rsplit('@', 1)[1]} did not clone; see clone.failed")
        if not roots[name]:
            continue
        if source.versions:
            version = name.rsplit("@", 1)[1]
            versioned.setdefault(source.name, []).append(readers[source.name](roots[name], source, version=version))
            continue
        built += 1
        yield readers[source.name](roots[name], source, name=name)
    for group in versioned.values():
        order = list(group[0].settings.versions)
        built += 1
        yield Versioned(sorted(group, key=lambda r: order.index(r.version)))
    log.info("sources.built", total=built)


def clone_repo(name, url, ref=None) -> Path | None:
    dest = Path(config.settings.repos_dir) / name
    if dest.exists():
        log.info("clone.skip", repo=name)
        return dest
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", *(["--branch", ref] if ref else []), url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("clone.done", repo=name)
        return dest
    except subprocess.CalledProcessError as e:
        log.error("clone.failed", repo=name, stderr=e.stderr.strip())
        return None


def provision(specs, workers=8) -> dict:
    log.info("provision.start", repos=len(specs), workers=workers)
    roots = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(clone_repo, *spec): spec[0] for spec in specs}
        for fut in as_completed(futures):
            roots[futures[fut]] = fut.result()  # Path or None
    ok = sum(v is not None for v in roots.values())
    log.info("provision.done", ok=ok, failed=len(roots) - ok)
    return roots


def one(name):
    for source in sources([name]):
        return source
    raise LookupError(f"no accepted source named {name}")
