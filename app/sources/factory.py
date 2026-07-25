import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
import logging_setup
from sources import (  # noqa: F401
    cheatsheets,
    developer_roadmap,
    interview,
    notes,
    redis_docs,
    system_design_primer,
)
from sources.base import Base
from sources.interview import InterviewSource

log = logging_setup.get_logger(__name__)


def all_sources():
    git_classes = [c for c in Base._registry.values() if getattr(c, "url", None)]
    local_classes = [
        c for c in Base._registry.values() if getattr(c, "url", None) is None
    ]
    interview_config = config.settings.sources.interview
    specs = [(c.name, c.url) for c in git_classes]
    specs += [
        (repo, f"{interview_config.base_url}/{repo}") for repo in interview_config.repos
    ]
    log.info(
        "sources.gather",
        local=len(local_classes),
        git=len(git_classes),
        interview=len(interview_config.repos),
    )
    roots = provision(specs)

    return build_classes(local_classes, git_classes, roots)


def build_classes(local_classes, git_classes, roots):
    interview_config = config.settings.sources.interview
    interview_repos = interview_config.repos
    built = 0
    for c in local_classes:
        built += 1
        yield c(Path(c.path))
    for c in git_classes:
        if roots[c.name]:
            built += 1
            yield c(roots[c.name])
    for repo in interview_repos:
        if roots[repo]:
            built += 1
            yield InterviewSource(
                roots[repo],
                name=repo,
                url=f"{interview_config.base_url}/{repo}",
                language=interview_config.language,
            )
    log.info("sources.built", total=built)


def clone_repo(name, url) -> Path | None:
    dest = Path(config.settings.repos_dir) / name
    if dest.exists():
        log.info("clone.skip", repo=name)
        return dest
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
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
        futures = {pool.submit(clone_repo, n, u): n for n, u in specs}
        for fut in as_completed(futures):
            roots[futures[fut]] = fut.result()  # Path или None
    ok = sum(v is not None for v in roots.values())
    log.info("provision.done", ok=ok, failed=len(roots) - ok)
    return roots
