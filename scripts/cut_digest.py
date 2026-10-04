"""Does a variant still cut into the rows it holds?"""

import difflib
import hashlib
import json
import sys

import config
import sources.factory
from orm.sync_db import Session
from sqlalchemy import text

import db


def digest(value: str) -> str:
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()


# keyed by file and position; the comparison reads the order of a file's texts, not the numbers
def stored(variant: str) -> dict[tuple[str, str, int], str]:
    with Session() as session:
        rows = session.execute(
            text(
                "SELECT ds.name, dc.source, dc.chunk_index, dc.content FROM data_chunks dc "
                f"JOIN data_sources ds ON ds.id = dc.source_id WHERE {db.live_rows('dc')}"
            ),
            {"variant": variant},
        )
        return {(name, src, idx): digest(content) for name, src, idx, content in rows}


def freshly_cut(variant: str) -> dict[tuple[str, str, int], str]:
    policy = config.settings.corpus.policy(variant)
    out = {}
    for source in sources.factory.sources():
        # the same method the indexer walks, or the variant reads as changed for its whole life
        for doc in source.documents(policy):
            out[(source.name, doc.source, doc.chunk_index)] = digest(doc.content)
    return out


# a file's chunks in their order: a renumbering with the same texts in the same order is the same cut
def _by_file(chunks: dict[tuple[str, str, int], str]) -> dict[tuple[str, str], list[str]]:
    files: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for (name, src, idx), value in chunks.items():
        files.setdefault((name, src), []).append((idx, value))
    return {key: [value for _, value in sorted(rows)] for key, rows in files.items()}


def compare(variant: str) -> dict:
    was, now = _by_file(stored(variant)), _by_file(freshly_cut(variant))
    counts = {"gone": 0, "new": 0, "changed": 0}
    moved = set()
    for key in was.keys() | now.keys():
        a, b = was.get(key, []), now.get(key, [])
        if a == b:
            continue
        moved.add(key)
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op == "delete":
                counts["gone"] += i2 - i1
            elif op == "insert":
                counts["new"] += j2 - j1
            elif op == "replace":
                counts["changed"] += max(i2 - i1, j2 - j1)
    return {
        "variant": variant,
        "sources": len({k[0] for k in was}),
        "sources_differing": len({k[0] for k in moved}),
        "files_differing": len({k[1] for k in moved}),
        "chunks_gone": counts["gone"],
        "chunks_new": counts["new"],
        "chunks_changed": counts["changed"],
        "differing": sorted({k[0] for k in moved})[:20],
    }


if __name__ == "__main__":
    variants = sys.argv[1:]
    if not variants:
        with Session() as s:
            variants = [
                r[0]
                for r in s.execute(text("SELECT DISTINCT variant FROM data_chunks ORDER BY 1"))
            ]
    print(json.dumps([compare(v) for v in variants]))
