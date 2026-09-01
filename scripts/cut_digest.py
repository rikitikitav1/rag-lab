"""Does a variant still cut into the rows it holds?

Row counts hide it: fourteen of the sixteen sources that changed under the new parser
kept the same count. This compares the text itself.
"""

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


# keyed by file and position: the same texts in another order is still a changed cut
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
    for source in sources.factory.all_sources():
        # the same method the indexer walks, or the variant reads as changed for its whole life
        for doc in source.documents(policy):
            out[(source.name, doc.source, doc.chunk_index)] = digest(doc.content)
    return out


def compare(variant: str) -> dict:
    was, now = stored(variant), freshly_cut(variant)
    moved = {k for k in was.keys() | now.keys() if was.get(k) != now.get(k)}
    return {
        "variant": variant,
        "sources": len({k[0] for k in was}),
        "sources_differing": len({k[0] for k in moved}),
        "files_differing": len({k[1] for k in moved}),
        "chunks_gone": sum(1 for k in moved if k not in now),
        "chunks_new": sum(1 for k in moved if k not in was),
        "chunks_changed": sum(1 for k in moved if k in was and k in now),
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
