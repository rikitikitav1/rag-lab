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

# the planner's own inputs, so the cache is invalidated by exactly what moves the answer.
# Pages, not rows: the alternative to the walk is a sequential read, priced per page, and
# a DELETE leaves the pages where they were until something rewrites the table. Dropping
# a variant of 17 498 rows moved neither the page count nor the crossover
_ESTIMATE = ("SELECT relpages, reltuples::bigint FROM pg_class"
             " WHERE relname = 'data_chunks'")
# this has to keep the shape `db.hybrid_search` gives the planner, or the gate guards a
# query nobody runs: same table, same variant filter, same operator, same LIMIT. Changing
# the vector leg without changing this leaves the depth audit green on the wrong plan
_PROBE = """
SELECT id FROM data_chunks
WHERE variant = :variant AND embedding IS NOT NULL
ORDER BY embedding <=> (SELECT embedding FROM data_chunks
                        WHERE variant = :variant AND embedding IS NOT NULL LIMIT 1)
LIMIT 20
"""

_resolved: dict[tuple[str, tuple[int, int]], int] = {}


def ladder() -> list[int]:
    return sorted(config.settings.retrieval.ef_ladder)


def _shape(conn) -> tuple[int, int]:
    row = conn.execute(text(_ESTIMATE)).first()
    return (int(row[0]), int(row[1])) if row else (0, 0)


def uses_index(conn, variant: str, ef: int) -> bool:
    # the measuring path disables index scans on every pooled connection to force exact
    # search, and this asks the same pool: without turning it back on the probe reads
    # "the planner wants a sort" for every rung and answers a question nobody asked.
    # Put back afterwards: the connection belongs to the caller, and a caller that was
    # measuring exactly would go on measuring something else for the rest of its
    # transaction
    was = conn.execute(text("SHOW enable_indexscan")).scalar()
    conn.execute(text("SET LOCAL enable_indexscan = on"))
    conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef)}"))
    try:
        plan = (
            conn.execute(text("EXPLAIN (COSTS OFF) " + _PROBE), {"variant": variant})
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


# resolved per variant and per estimated row count: a variant indexed beside this one
# changes the answer for this one, and the estimate is what says so. The cache is per
# process and nobody clears the API's: it does not need clearing, because the key is the
# statistic the planner itself reads, so the two change their minds together
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
        # never cached: a table whose index the planner refuses at every rung is either
        # news or a poisoned session, and a poisoned answer that sticks is worse than a
        # slow one asked again
        log.warning("depth.no_rung_uses_the_index", variant=variant, ladder=rungs)
        return rungs[0]
    _resolved[key] = depth
    log.info("depth.resolved", variant=variant, ef_search=depth, ladder=rungs)
    return depth


def forget() -> None:
    _resolved.clear()


# what every indexed variant would run at, and whether the plan agrees: one reading for
# the preflight, for the end of an index build, and for a person asking
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
