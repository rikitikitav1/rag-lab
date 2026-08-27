"""Ranks under several settings at once, compared pairwise on the same questions.

The generation experiment spends the card and a judge; this one reads where the right
chunk landed, which costs minutes and neither. The measuring lives here rather than in
the script so the job and the script cannot grow two different notions of a rank.
"""

import contextlib
import itertools
import random
import re
from dataclasses import dataclass, field

import config
import job_queue
import logging_setup

import db

CANDIDATES = 100
DEPTH = 20
CUTOFFS = (1, 3, 5, 10)
BOOTSTRAP = 2000
NO_THRESHOLD = 2.0
EF_SEARCH = config.settings.retrieval.ef_search

log = logging_setup.get_logger(__name__)


@dataclass
class ComparisonPlan:
    axes: dict
    param: str | None = None
    dataset: str = ""
    sample_size: int | None = None
    question_ids: list[int] | None = field(default=None)
    job_id: int | None = None


# one reading of the arm: derived twice, the label and the depth drift apart
def depth_of(arm: dict) -> tuple[bool, int]:
    ef = arm.get("ef_search")
    return ef is None, ef or EF_SEARCH


def _clean(text):
    return re.sub(r"[_*`]", "", text or "").strip().lower()


def _heading_text(section):
    """section is a heading path ("h1 > 12. Question?"); only the leaf identifies the section."""
    leaf = (section or "").split(" > ")[-1]
    return _clean(re.sub(r"^\d+\.\s*", "", leaf))


# ids win over the set when both are given: an experiment fixes its questions before it
# runs, and a comparison that quietly measured a different list would answer nothing
def questions(conn, set_name, limit, ids=None):
    from sqlalchemy import text as sql

    where = "q.set_name = :set_name" if not ids else "q.id = ANY(:ids)"
    rows = conn.execute(
        sql(f"""
            SELECT q.id, q.original_text, q.marked_sources, q.embedding::text AS emb,
                   COALESCE(o.original_text, q.original_text) AS gold_heading
            FROM questions q
            LEFT JOIN questions o ON o.id = q.source_question_id
            WHERE {where}
              AND q.embedding IS NOT NULL
              AND array_length(q.marked_sources, 1) > 0
            ORDER BY q.id
            LIMIT :limit
        """),
        {"set_name": set_name, "ids": ids, "limit": limit},
    ).mappings().all()
    return [dict(r) for r in rows]


def ranked_lists(db, question, variant, depth=DEPTH, limit_keyword=CANDIDATES,
                 limit_vector=CANDIDATES, distance_threshold=NO_THRESHOLD, rerank_top=0):
    rows = db.hybrid_search(
        question["original_text"],
        question["emb"],
        None,
        limit_vector=limit_vector,
        limit_keyword=limit_keyword,
        limit=CANDIDATES,
        variant=variant,
        distance_threshold=distance_threshold,
    )
    if rerank_top:
        rows = _reranked(question["original_text"], rows, rerank_top)
    files, sections = [], []
    for row in rows:
        source, section = row[1], row[8]
        if source not in files:
            files.append(source)
        key = (source, _heading_text(section))
        if key not in sections:
            sections.append(key)
    return files[:depth], sections[:depth], rows


# the cross-encoder reorders the head of the fused list and the tail keeps its order,
# which is what the runtime does: retrieve wide, rerank, narrow
def _reranked(question: str, rows, top: int):
    import rerank

    head, tail = list(rows[:top]), list(rows[top:])
    scores = rerank.score_pairs([(question, row[0]) for row in head])
    order = sorted(range(len(head)), key=lambda i: -scores[i])
    return [head[i] for i in order] + tail


def rank_of_file(files, marked):
    for i, source in enumerate(files, 1):
        if any(m in source for m in marked):
            return i
    return None


_APPLIED = None


def prepare(exact: bool, ef: int = EF_SEARCH) -> None:
    """Exact search measures the corpus; hnsw measures the index, and it is not reproducible.

    On checkout, not on connect: hybrid_search opens its own connection, and a pooled one created
    before this ran would come back without the setting and quietly search through the index.
    """
    global _APPLIED
    from orm.sync_db import engine
    from sqlalchemy import event

    if _APPLIED is not None:
        event.remove(engine, "checkout", _APPLIED)
        _APPLIED = None

    setting = "SET enable_indexscan = off" if exact else f"SET hnsw.ef_search = {int(ef)}"

    def apply(dbapi_conn, _record, _proxy):
        cursor = dbapi_conn.cursor()
        cursor.execute("RESET enable_indexscan")
        cursor.execute(setting)
        cursor.close()

    event.listen(engine, "checkout", apply)
    _APPLIED = apply


# the listener is on the process-wide engine, so a job that leaves it installed hands its
# search mode to every later job in that worker while their records keep reporting the ef
# from the config. Removing it is not enough on its own: a pooled connection carries the
# guc it was checked out with, so the pool goes too
def release() -> None:
    global _APPLIED
    from orm.sync_db import engine
    from sqlalchemy import event

    if _APPLIED is not None:
        event.remove(engine, "checkout", _APPLIED)
        _APPLIED = None
    engine.dispose()


# every arm sets the mode on the shared engine, so the grid holds one guard rather than
# one per arm: what must not survive the job is the last arm's setting
@contextlib.contextmanager
def search_mode_restored():
    try:
        yield
    finally:
        release()


@contextlib.contextmanager
def search_mode(exact: bool, ef: int = EF_SEARCH):
    with search_mode_restored():
        prepare(exact, ef)
        yield


# a pool smaller than asked is a fact about one run, so `run()` empties it before measuring
# rather than letting it grow for the life of a worker
POOL_SHORTFALL: list = []


def assert_pool(rows, question_id, asked) -> None:
    """A pool smaller than asked means something capped the search, and the label is then a lie."""
    if len(rows) < asked:
        POOL_SHORTFALL.append((question_id, len(rows)))


# `position()` on source cannot use an index, so this was a full pass over the variant
# once per question per arm. It is constant per (variant, marked), so it is asked once
_SECTIONS_UNDER: dict = {}


def _sections_under(conn, variant, marked: tuple[str, ...]) -> set[str]:
    from sqlalchemy import text as sql

    key = (variant, marked)
    if key in _SECTIONS_UNDER:
        return _SECTIONS_UNDER[key]
    rows = conn.execute(
        sql("""
            SELECT DISTINCT section FROM data_chunks
            WHERE variant = :variant AND section IS NOT NULL
              AND source_id IN (SELECT id FROM data_sources WHERE active)
              AND (""" + " OR ".join(f"position(:m{i} in source) > 0" for i in range(len(marked))) + """)
        """),
        {"variant": variant, **{f"m{i}": m for i, m in enumerate(marked)}},
    ).scalars().all()
    found = {_heading_text(r) for r in rows}
    _SECTIONS_UNDER[key] = found
    return found


def section_exists(conn, variant, marked, gold_heading) -> bool:
    gold = _clean(gold_heading)
    if not gold:
        return False
    return gold in _sections_under(conn, variant, tuple(marked))


def rank_of_section(sections, marked, gold_heading):
    gold = _clean(gold_heading)
    for i, (source, heading) in enumerate(sections, 1):
        if any(m in source for m in marked) and heading and heading == gold:
            return i
    return None


def measure(db, conn, set_name, variant, limit, exact, ef=EF_SEARCH, limit_keyword=CANDIDATES,
            limit_vector=CANDIDATES, distance_threshold=NO_THRESHOLD, rerank_top=0,
            question_ids=None, source=None):
    qs = questions(conn, set_name, limit, ids=question_ids)
    if source:
        qs = [q for q in qs if any(m.startswith(source) for m in q["marked_sources"])]
    prepare(exact, ef)
    out = []
    for q in qs:
        files, sections, rows = ranked_lists(
            db, q, variant, limit_keyword=limit_keyword, limit_vector=limit_vector,
            distance_threshold=distance_threshold, rerank_top=rerank_top,
        )
        assert_pool(rows, q["id"], min(limit_vector, CANDIDATES))
        scorable = section_exists(conn, variant, q["marked_sources"], q["gold_heading"])
        out.append({
            "id": q["id"],
            # the repository a question belongs to. The halves are not drawn inside it
            # (that rule moved when the set grew), so each comparison prints how many
            # repositories its half actually covered instead of assuming both cover all
            "repo": q["marked_sources"][0].split("/")[0] if q["marked_sources"] else None,
            "file_rank": rank_of_file(files, q["marked_sources"]),
            "section_scorable": scorable,
            "section_rank": (
                rank_of_section(sections, q["marked_sources"], q["gold_heading"])
                if scorable else None
            ),
            # the keyword leg reached the gold file AND that row survived the fusion cap:
            # `rows` is the fused list, so this is a floor on "the leg reached it"
            "gold_by_keyword_in_pool": any(
                row[5] is not None
                and any(m in row[1] for m in q["marked_sources"])
                for row in rows
            ),
            "files": files,
            "sections": [list(s) for s in sections],
        })
    return out

def _rr(rank) -> float:
    return 1.0 / rank if rank else 0.0


def bootstrap_ci(deltas, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP)]


def paired_delta(before: list[dict], after: list[dict], level: str) -> dict:
    key = f"{level}_rank"
    if level == "section":
        before = [r for r in before if r.get("section_scorable")]
        after = [r for r in after if r.get("section_scorable")]
    was = {r["id"]: r for r in before}
    paired = [(was[r["id"]], r) for r in after if r["id"] in was]
    if not paired:
        return {"error": "no shared questions"}
    deltas = [_rr(now[key]) - _rr(then[key]) for then, now in paired]
    better = sum(1 for d in deltas if d > 0)
    worse = sum(1 for d in deltas if d < 0)
    low, high = bootstrap_ci(deltas)
    return {
        "level": level,
        "questions": len(paired),
        "delta_MRR": round(sum(deltas) / len(deltas), 4),
        "ci95": [round(low, 4), round(high, 4)],
        "better": better,
        "worse": worse,
        "unchanged": len(deltas) - better - worse,
    }


def summarise(rows: list[dict], level: str) -> dict:
    key = f"{level}_rank"
    if level == "section":
        rows = [r for r in rows if r.get("section_scorable")]
    if not rows:
        return {}
    stats = {
        f"hit@{c}": round(sum(1 for r in rows if r[key] and r[key] <= c) / len(rows), 4)
        for c in CUTOFFS
    }
    stats[f"MRR@{DEPTH}"] = round(sum(_rr(r[key]) for r in rows) / len(rows), 4)
    stats["n"] = len(rows)
    return stats


# the grid is the product of the axes and an arm is one point of it, named from its own
# values so the name says what it was rather than which position it held
# an arm is minutes and the product grows by multiplication: a grid nobody meant to ask
# for is hours of card time and a record of megabytes
GRID_CAP = 32


def arms(axes: dict) -> list[dict]:
    names = sorted(axes)
    size = 1
    for n in names:
        size *= len(axes[n])
    if size > GRID_CAP:
        raise ValueError(f"grid of {size} arms is over the cap of {GRID_CAP}: {axes}")
    return [
        dict(zip(names, values, strict=True)) for values in itertools.product(*(axes[n] for n in names))
    ]


def arm_name(arm: dict) -> str:
    return "_".join(f"{k}={_suffix(v)}" for k, v in sorted(arm.items()))


# a bool is an int in Python, and `rerank_top: [true]` would become rows[:1]
def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# a range, not a shape: a value refused after the arms are measured is refused too late
AXIS_RULES = {
    # 1..1000 is what hnsw.ef_search accepts; 0 used to be measured at the configured depth
    "ef_search": lambda v: _whole(v) and 1 <= v <= 1000,
    # a pool of nothing disarms assert_pool, the instrument that says a search was capped
    "limit_vector": lambda v: _whole(v) and v >= 1,
    "limit_keyword": lambda v: _whole(v) and v >= 1,
    "rerank_top": lambda v: _whole(v) and v >= 0,
    "distance_threshold": lambda v: isinstance(v, int | float)
    and not isinstance(v, bool)
    and 0 <= v <= 2,
    "source": lambda v: isinstance(v, str) and bool(v),
    "variant": lambda v: isinstance(v, str) and v in config.settings.corpus.variants,
}
AXIS_LIMITS = {
    "ef_search": "a whole number 1..1000 (what hnsw.ef_search accepts)",
    "limit_vector": "a whole number 1 or more",
    "limit_keyword": "a whole number 1 or more",
    "rerank_top": "a whole number 0 or more",
    "distance_threshold": "a number 0..2",
    "source": "a non-empty name",
    "variant": "a variant declared in config",
}


# derived, not listed: an axis cannot be admitted without a rule saying what it may hold
AXES = frozenset(AXIS_RULES)


# what a later reading needs and nothing else: the candidate lists are the bulk of a row
def _keep(row: dict) -> dict:
    return {
        "id": row["id"],
        "file_rank": row["file_rank"],
        "section_rank": row["section_rank"],
        "section_scorable": row["section_scorable"],
        "repo": row.get("repo"),
        "gold_by_keyword_in_pool": row.get("gold_by_keyword_in_pool"),
    }


# what two measurements must agree on to be compared. Beside the producer, so a field
# added to one is not invisible to the other
# the procedure of measuring, not the corpus measured: `variant`, `policy` and
# `fingerprint` differ between two cuts by construction, and comparing two cuts is the job
COMPARABLE = (
    "set", "search", "candidates", "limit_vector", "limit_keyword",
    "distance_threshold", "keyword", "questions_hash", "rerank_top",
)
# absent is not a difference: it is the value the run had before anyone wrote it down
ABSENT_MEANS = {"rerank_top": 0}


def comparable(before, after) -> list[tuple]:
    return [
        (field, was, now)
        for field in COMPARABLE
        for was in [before.get(field, ABSENT_MEANS.get(field))]
        for now in [after.get(field, ABSENT_MEANS.get(field))]
        if was != now
    ]


# the shape the script's report writes, so one instrument reads a record and a file
def arm_procedure(arm: dict, rows: list[dict], dataset: str) -> dict:
    exact, ef = depth_of(arm)
    variant = arm.get("variant") or config.settings.corpus.variant
    return {
        "variant": variant,
        "set": dataset,
        "search": "exact" if exact else f"hnsw ef_search={ef}",
        "rerank_top": arm.get("rerank_top", 0),
        "candidates": CANDIDATES,
        "limit_vector": arm.get("limit_vector", CANDIDATES),
        "limit_keyword": arm.get("limit_keyword", CANDIDATES),
        "distance_threshold": arm.get("distance_threshold", NO_THRESHOLD),
        "source": arm.get("source"),
        "keyword": {
            "query": config.settings.retrieval.keyword_query,
            "rank": config.settings.retrieval.keyword_rank,
            "norm": config.settings.retrieval.keyword_norm,
            "query_lang": config.settings.retrieval.query_lang,
        },
        "questions": len(rows),
        "questions_hash": _ids_hash(rows),
        # descriptive, never compared: what cut these rows is not recoverable from them,
        # which is what `scripts/cut_digest.py` re-cuts the corpus to answer
        "policy": config.settings.corpus.policy_or_none(variant),
        "fingerprint": db.fingerprint_or_none(variant=variant),
    }


def _ids_hash(rows) -> str:
    import hashlib

    joined = ",".join(str(r["id"]) for r in sorted(rows, key=lambda r: r["id"]))
    return hashlib.md5(joined.encode(), usedforsecurity=False).hexdigest()[:12]


# an arm is compared against the arm differing from it only in the axis of record: with
# one axis that is its first value, with two one reference per combination of the rest
def _reference_for(arm: dict, param: str | None, axes: dict) -> dict | None:
    if not param or param not in axes:
        return None
    if arm[param] == axes[param][0]:
        return None
    return {**arm, param: axes[param][0]}


def run(experiment) -> dict:
    from orm.sync_db import engine

    unknown = sorted(set(experiment.axes) - AXES)
    if unknown:
        raise ValueError(f"unknown axes: {unknown}")
    # product() of nothing yields one empty tuple, so an empty axes dict would run a
    # single nameless arm and compare it against itself
    if not experiment.axes:
        raise ValueError("no axes to compare")
    # the route validated these on the way in, and a retry re-reads them from the row long
    # after: the rules move, the stored axes do not
    for name, values in experiment.axes.items():
        bad = [v for v in values if not AXIS_RULES[name](v)]
        if bad:
            raise ValueError(f"{name} takes {AXIS_LIMITS[name]}, got: {bad}")
    grid = arms(experiment.axes)
    # the route is not the only door: a hand-written row or a replay reaches here too
    if not grid:
        raise ValueError(f"no arms to measure: {experiment.axes}")
    # two arms sharing a name overwrite each other in `measured`
    names = [arm_name(a) for a in grid]
    if len(set(names)) != len(names):
        raise ValueError(f"arms do not have distinct names: {sorted(names)}")
    param = getattr(experiment, "param", None)
    # arms along `source` share no questions, so every delta would be "no shared questions"
    if param == "source":
        raise ValueError("source stratifies a comparison, it cannot be the axis of record")

    # before the connection: a cancelled job must not open one to find out it is cancelled
    job_id = getattr(experiment, "job_id", None)
    if job_id is not None and job_queue.is_cancelled(job_id):
        raise RuntimeError("comparison cancelled before it measured anything")

    POOL_SHORTFALL.clear()
    _SECTIONS_UNDER.clear()
    measured, summary, kept = {}, {}, {}
    with contextlib.ExitStack() as stack:
        stack.enter_context(search_mode_restored())
        conn = stack.enter_context(engine.connect())
        for arm in grid:
            if job_id is not None and job_queue.is_cancelled(job_id):
                raise RuntimeError(f"comparison cancelled after {len(measured)} arms")
            name = arm_name(arm)
            exact, ef = depth_of(arm)
            rows = measure(
                db,
                conn,
                experiment.dataset,
                arm.get("variant") or config.settings.corpus.variant,
                experiment.sample_size or 10**6,
                exact=exact,
                ef=ef,
                limit_keyword=arm.get("limit_keyword", CANDIDATES),
                limit_vector=arm.get("limit_vector", CANDIDATES),
                distance_threshold=arm.get("distance_threshold", NO_THRESHOLD),
                rerank_top=arm.get("rerank_top", 0),
                question_ids=experiment.question_ids,
                source=arm.get("source"),
            )
            measured[name] = rows
            summary[name] = {level: summarise(rows, level) for level in ("file", "section")}
            kept[name] = [_keep(row) for row in rows]
            log.info("compare.arm_measured", arm=name, questions=len(rows))

    deltas = {}
    for arm in grid:
        against = _reference_for(arm, param, experiment.axes)
        if against is None:
            continue
        name, base = arm_name(arm), arm_name(against)
        deltas[name] = {
            "against": base,
            **{
                level: paired_delta(measured[base], measured[name], level)
                for level in ("file", "section")
            },
        }
    return {
        "reference_axis": param,
        # per question, per arm: without them a delta cannot be recomputed under another
        # reference, the questions that paid cannot be named, and a stratified reading
        # needs the whole grid measured again
        "rows": kept,
        "arms": summary,
        "deltas": deltas,
        # a pool smaller than asked means something capped the search, and a comparison
        # that hit it measured a different thing than it says
        "pool_shortfall": len(POOL_SHORTFALL),
        "procedure": {
            "schema": 2,
            "dataset": experiment.dataset,
            "axes": experiment.axes,
            "param": param,
            # per arm: arms differ in variant, depth and, along `source`, in the questions
            "arms": {
                arm_name(a): arm_procedure(a, measured[arm_name(a)], experiment.dataset)
                for a in grid
            },
        },
    }


def _suffix(value) -> str:
    return str(value).replace(".", "").replace("/", "_")
