import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
import logging_setup
from sources import (  # noqa: F401
    cheatsheets,
    converted,
    files,
    interview,
    notes,
    redis_docs,
    system_design_primer,
)
from sources.base import Base

log = logging_setup.get_logger(__name__)


# every repository a source file names, one organisation's family unrolled, one checkout per released version
def _git_specs(found) -> list[tuple[str, str, object, str | None]]:
    specs = []
    for source in found:
        if source.folder is not None:
            continue
        if source.versions:
            url = files.row_of(source, source.name)["git_url"]
            specs += [(f"{source.name}@{v}", url, source, origin.ref) for v, origin in source.versions.items()]
        else:
            specs += [(name, files.row_of(source, name)["git_url"], source, None) for name in files.rows_of(source)]
    return specs


# a reader named in a file and known to no class would parse nothing its file meant
def _reader(source) -> type[Base]:
    if source.reader is not None and source.reader not in Base._registry:
        raise LookupError(f"{source.name}: reader {source.reader} is no class; known: {sorted(Base._registry)}")
    return Base._registry.get(source.reader, Base)


# the sources the index reads, from their files: local folders first, then one repository each, then families
def all_sources():
    from sources.versioned import Versioned

    found = list(files.source_files().values())
    local = [s for s in found if s.folder is not None]
    specs = _git_specs(found)
    readers = {s.name: _reader(s) for s in found}
    log.info("sources.gather", local=len(local), git=len(specs))
    roots = provision([(name, url, ref) for name, url, _, ref in specs])
    built = 0
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
    # provision skips a checkout that already exists, so filtering the full build is cheap
    for source in all_sources():
        if source.name == name:
            return source
    raise LookupError(f"no such source: {name}")
