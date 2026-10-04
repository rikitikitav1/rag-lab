import re

from errors import Refusal
from models.corpus import DataSource
from orm.sync_db import Session
from paths import RAW, ROOT
from sqlalchemy import select
from use_cases.source_intake import run_under_review

# a chapter or a page is read whole by an agent; a whole book in one answer would not fit its window
MAX_CHARS = 20000
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _files_dir(name: str):
    with Session() as session:
        found = session.scalar(select(DataSource).where(DataSource.name == name))
    if found is None:
        raise Refusal("missing", f"no source named {name}")
    folder = run_under_review(found).get("folder")
    if not folder:
        raise Refusal("invalid", f"{name} has no converted run yet; onboard it first")
    files = (ROOT / folder / "files").resolve()
    if not files.is_relative_to(RAW.resolve()) or not files.is_dir():
        raise Refusal("invalid", f"{name}'s run folder {folder} holds no converted files")
    return folder, files


# the run's whole markdown per file, as converted and before the cut: what a dirty verdict was read on
def read(name: str, file: str | None = None, heading: str | None = None, offset: int = 0,
         limit: int = MAX_CHARS) -> dict:
    folder, files = _files_dir(name)
    if file is None:
        return {"run": folder, "files": [{"file": p.name, "chars": p.stat().st_size}
                                         for p in sorted(files.glob("*.md"))]}
    path = (files / file).resolve()
    if not path.is_relative_to(files) or not path.is_file():
        raise Refusal("missing", f"{file} is not a file of {name}'s run; list the files first")
    text = path.read_text(encoding="utf-8", errors="replace")
    if heading is not None:
        text = _section(text, heading)
        if text is None:
            raise Refusal("missing", f"no heading containing {heading!r} in {file}")
    limit = max(1, min(limit, MAX_CHARS))
    return {"run": folder, "file": file, "chars": len(text), "offset": offset,
            "text": text[offset:offset + limit], "more": offset + limit < len(text)}


# from the first heading holding the words to the next heading of its level or above
def _section(text: str, heading: str) -> str | None:
    lines = text.splitlines(keepends=True)
    wanted = heading.lower()
    for i, line in enumerate(lines):
        m = _HEADING.match(line)
        if m and wanted in m.group(2).lower():
            level = len(m.group(1))
            end = next((j for j in range(i + 1, len(lines))
                        if (n := _HEADING.match(lines[j])) and len(n.group(1)) <= level), len(lines))
            return "".join(lines[i:end])
    return None
