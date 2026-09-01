"""How deep the hnsw walk goes, asked of the planner rather than remembered.

Where the planner abandons the index is a property of the whole table: the alternative
to the walk is a sequential read of data_chunks filtered by variant, so rows of every
other variant pay into its cost. Indexing one more variant moved the crossover from
about 197 to about 265, which is why a number written down on Tuesday describes
Tuesday. There is no rule of thumb either: the crossover is nonlinear in rows and moves
with random_page_cost. The plan is the only oracle, and it costs a millisecond to ask.
"""

import contextlib

import config
import logging_setup
from sqlalchemy import text

import db

log = logging_setup.get_logger(__name__)

AUTO = "auto"
INDEX_SCAN = "Index Scan"

# pages, not rows: dropping a variant of 17 498 rows moved neither pages nor the crossover
_ESTIMATE = ("SELECT relpages, reltuples::bigint FROM pg_class"
             " WHERE relname = 'data_chunks'")
# the shape `hybrid_search` gives the planner: same table, rows, operator and limit
def _probe(limit_vector: int | None = None) -> str:
    # asked per call, because `limit_vector` is a sweepable axis and an arm may move it
    limit = limit_vector or config.settings.retrieval.limit_vector
    return f"""
SELECT id FROM data_chunks
WHERE {db.live_rows()} AND embedding IS NOT NULL
ORDER BY embedding <=> (SELECT embedding FROM data_chunks
                        WHERE {db.live_rows()} AND embedding IS NOT NULL LIMIT 1)
LIMIT {int(limit)}
"""

_resolved: dict[tuple[str, tuple[int, int]], int] = {}


def ladder() -> list[int]:
    return sorted(config.settings.retrieval.ef_ladder)


def _shape(conn) -> tuple[int, int]:
    row = conn.execute(text(_ESTIMATE)).first()
    return (int(row[0]), int(row[1])) if row else (0, 0)


def uses_index(conn, variant: str, ef: int, limit_vector: int | None = None) -> bool:
    # a caller measuring exact search turns index scans off on this connection, so ask with them on
    was = conn.execute(text("SHOW enable_indexscan")).scalar()
    conn.execute(text("SET LOCAL enable_indexscan = on"))
    conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef)}"))
    try:
        plan = (
            conn.execute(
                text("EXPLAIN (COSTS OFF) " + _probe(limit_vector)), {"variant": variant}
            )
            .scalars().all()
        )
    finally:
        conn.execute(text(f"SET LOCAL enable_indexscan = {'on' if was == 'on' else 'off'}"))
    return any(INDEX_SCAN in line for line in plan)


def deepest_indexed(conn, variant: str) -> int | None:
    found = None
    for ef in ladder():
        if uses_index(conn, variant, ef):
            found = ef
    return found


# per variant and per row estimate: the key is the statistic the planner itself reads
def resolve(variant: str | None = None, override: int | None = None, conn=None) -> int:
    if override is not None:
        return override
    declared = config.settings.retrieval.ef_search
    if declared != AUTO:
        return int(declared)
    variant = variant or config.settings.corpus.variant
    with contextlib.ExitStack() as stack:
        # a caller inside a search already holds one; nothing else should open a second
        conn = conn or stack.enter_context(db.engine.connect())
        key = (variant, _shape(conn))
        if key in _resolved:
            return _resolved[key]
        depth = deepest_indexed(conn, variant)
    rungs = ladder()
    if depth is None:
        # never cached: a poisoned answer that sticks is worse than a slow one asked again
        log.warning("depth.no_rung_uses_the_index", variant=variant, ladder=rungs)
        return rungs[0]
    _resolved[key] = depth
    log.info("depth.resolved", variant=variant, ef_search=depth, ladder=rungs)
    return depth


def forget() -> None:
    _resolved.clear()


# one reading for the preflight, for the end of a build, and for a person asking
def audit(variants: list[str] | None = None) -> list[dict]:
    names = variants if variants is not None else [v["variant"] for v in db.corpus_variants()]
    declared = config.settings.retrieval.ef_search
    out = []
    with db.engine.connect() as conn:
        pages, rows = _shape(conn)
        for variant in names:
            depth = deepest_indexed(conn, variant)
            serving = depth or ladder()[0] if declared == AUTO else int(declared)
            out.append({
                "variant": variant,
                "rows_estimate": rows,
                "pages": pages,
                "ladder": ladder(),
                "deepest_indexed": depth,
                "declared": declared,
                "serving": serving,
                "serving_uses_index": uses_index(conn, variant, serving),
            })
    return out
