"""Ranks under several settings at once, compared pairwise on the same questions."""

import contextlib
import itertools
from dataclasses import dataclass, field

import config
import corpus_search
import job_queue
import llm
import logging_setup
import query_aliases
from config import KEYWORD_QUERY_MODES
from corpus_keys import HAS_GOLD_SQL, READ_BY_RUNS_SQL, TERM_SHARE_FLOOR, Gold, source_name_of
from evals.stats import bootstrap_ci, deltas_over, tally
from gold_match import heading_text, rank_of_exact_section, rank_of_gold, rank_of_section
from search_scope import Scope, ScopeRefused

import db

# 1 before the halves and the per-arm procedure; 2 those; 3 p unrounded, the way Holm reads it
SCHEMA = 3

CANDIDATES = config.settings.evals.retrieval_compare.candidates
DEPTH = config.settings.evals.retrieval_compare.depth
CUTOFFS = tuple(config.settings.evals.retrieval_compare.cutoffs)
# cosine distance never exceeds 2, so this threshold lets every candidate through
NO_THRESHOLD = 2.0

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
    import search_depth

    ef = arm.get("ef_search")
    return ef is None, ef or search_depth.resolve(arm.get("variant"))


# ids win over the set: an experiment fixes its questions before it runs
def questions(conn, set_name, limit, ids=None):
    from sqlalchemy import text as sql

    where = "q.set_name = :set_name" if not ids else "q.id = ANY(:ids)"
    # the limit samples a set; trimming a fixed list would record the prefix as the plan
    cap = "" if ids else "LIMIT :limit"
    rows = (
        conn.execute(
            sql(f"""
            SELECT q.id, q.original_text, q.marked_sources, q.gold, q.pair_id, q.embedding::text AS emb,
                   q.embedded_by,
                   COALESCE(o.original_text, q.original_text) AS gold_heading
            FROM questions q
            LEFT JOIN questions o ON o.id = q.source_question_id
            WHERE {where}
              AND q.embedding IS NOT NULL
              AND {HAS_GOLD_SQL.format(q="q")} AND {READ_BY_RUNS_SQL.format(q="q")}
            ORDER BY q.id
            {cap}
        """),
            {"set_name": set_name, "ids": ids, "limit": limit},
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


def ranked_lists(
    searcher,
    question,
    variant,
    depth=DEPTH,
    limit_keyword=CANDIDATES,
    limit_vector=CANDIDATES,
    distance_threshold=NO_THRESHOLD,
    rerank_top=0,
    ef_search=None,
    exact=False,
    scope=None,
):
    # the depth travels with the call, or the record names one number and the run uses another
    rows = searcher.hybrid_search(
        question["original_text"],
        question["emb"],
        scope,
        limit_vector=limit_vector,
        limit_keyword=limit_keyword,
        limit=CANDIDATES,
        variant=variant,
        distance_threshold=distance_threshold,
        ef_search=ef_search,
        exact=exact,
        # embedded before the run, maybe by another embedder than the role serves today
        embedded_by=question.get("embedded_by") or llm.embedder_label(),
    )
    if rerank_top:
        rows = _reranked(question["original_text"], rows, rerank_top)
    files, sections = [], []
    for hit in rows:
        source, section = hit.source, hit.section
        if source not in files:
            files.append(source)
        key = (source, heading_text(section))
        if key not in sections:
            sections.append(key)
    return files[:depth], sections[:depth], rows


# the head is reordered and the tail keeps its order: retrieve wide, rerank, narrow
def _reranked(question: str, rows, top: int):
    import rerank

    head, tail = list(rows[:top]), list(rows[top:])
    scores = rerank.score_pairs([(question, hit.content) for hit in head])
    order = sorted(range(len(head)), key=lambda i: -scores[i])
    return [head[i] for i in order] + tail


# two variants can hold different files, so such a delta is part cut and part corpus
FILE_DRIFT_SQL = """
WITH files AS (
    SELECT DISTINCT variant, source FROM data_chunks WHERE variant IN (:before, :after)
), per_file AS (
    SELECT split_part(source, '/', 1) AS family, source,
           bool_or(variant = :before) AS in_before,
           bool_or(variant = :after) AS in_after
    FROM files GROUP BY 1, 2
)
SELECT family,
       count(*) FILTER (WHERE in_before AND NOT in_after) AS only_before,
       count(*) FILTER (WHERE in_after AND NOT in_before) AS only_after,
       count(*) FILTER (WHERE in_before AND in_after) AS shared
FROM per_file GROUP BY 1
HAVING count(*) FILTER (WHERE NOT (in_before AND in_after)) > 0
ORDER BY 1
"""


def file_drift(conn, before: str, after: str) -> list[dict]:
    from sqlalchemy import text as sql

    rows = conn.execute(sql(FILE_DRIFT_SQL), {"before": before, "after": after}).mappings()
    return [dict(r) for r in rows]


# a pool smaller than asked is a fact about one run, so `run()` empties it first
POOL_SHORTFALL: list = []


def assert_pool(rows, question_id, asked) -> None:
    """A pool smaller than asked means something capped the search, and the label is then a lie."""
    if len(rows) < asked:
        POOL_SHORTFALL.append((question_id, len(rows)))


# an exact gold ranks the fused rows by their whole section path; an older one ranks the heading pairs as before
def _section_rank(rows, sections, gold: Gold, gold_heading):
    if not gold.exact:
        return rank_of_section(sections, gold, gold_heading)
    rank = rank_of_exact_section(
        [{"source": h.source, "section": h.section, "versions": list(h.versions)} for h in rows], gold
    )
    # the same depth the heading pairs are cut at, counted in its own grain: distinct whole paths, not heading pairs
    return rank if rank and rank <= DEPTH else None


def measure(
    searcher,
    conn,
    set_name,
    variant,
    limit,
    exact,
    ef=None,
    limit_keyword=CANDIDATES,
    limit_vector=CANDIDATES,
    distance_threshold=NO_THRESHOLD,
    rerank_top=0,
    question_ids=None,
    source=None,
    clamped=False,
):
    qs = questions(conn, set_name, limit, ids=question_ids)
    for q in qs:
        q["gold"] = Gold.of(q["marked_sources"], q["gold"])
    if source:
        qs = [q for q in qs if any(m.startswith(source) for m in q["gold"].marks)]
    out = []
    for q in qs:
        gold = q["gold"]
        try:
            files, sections, rows = ranked_lists(
                searcher,
                q,
                variant,
                limit_keyword=limit_keyword,
                limit_vector=limit_vector,
                distance_threshold=distance_threshold,
                rerank_top=rerank_top,
                ef_search=None if exact else ef,
                exact=exact,
                # clamped to the gold's own sources: a miss here is the source's, not the corpus around it
                scope=Scope(sources=tuple(dict.fromkeys(source_name_of(m) for m in gold.marks))) if clamped else None,
            )
        except ScopeRefused:
            # the gold's source is out of search: the clamped question finds nothing rather than stopping the run
            if not clamped:
                raise
            files, sections, rows = [], [], []
        # a source holds fewer chunks than the pool asks, so a clamped pool is short by design
        if not clamped:
            assert_pool(rows, q["id"], min(limit_vector, CANDIDATES))
        scorable = db.section_exists(conn, variant, gold, q["gold_heading"])
        out.append(
            {
                "id": q["id"],
                "pair_id": q.get("pair_id"),
                # which corpus repository the gold sits in: halves are drawn across repos, not inside
                "repo": source_name_of(gold.marks[0]),
                "file_rank": rank_of_gold(files, gold),
                "section_scorable": scorable,
                "section_rank": (
                    _section_rank(rows, sections, gold, q["gold_heading"]) if scorable else None
                ),
                # `rows` is the fused list, so this is a floor on "the keyword leg reached the gold"
                "gold_by_keyword_in_pool": any(
                    hit.keyword_rank is not None and gold.holds_file(hit.source) for hit in rows
                ),
                "files": files,
                "sections": [list(s) for s in sections],
                # the same reword the search made, so a loser is pinned to the dictionary entry that fired
                "aliases_fired": query_aliases.reword(q["original_text"])[1] or None,
            }
        )
    return out


def rr(rank) -> float:
    return 1.0 / rank if rank else 0.0


# the half is a pure function of the id: A chooses the winner, B reports on it
SPLIT_SEED = "hygiene_v1"


# drawn by size: a split that moves when the set grows is no pre-registration; a pair's two languages share a half
def half_of(question_id, pair_id: str | None = None) -> str:
    import hashlib

    digest = hashlib.md5(f"{pair_id or question_id}:{SPLIT_SEED}".encode(), usedforsecurity=False).hexdigest()
    return "A" if int(digest, 16) % 2 == 0 else "B"


# the ids both arms carry that fall in one half; the hash names the questions the numbers came from
def half_ids(before: list[dict], after: list[dict], which: str) -> set:
    shared = {r["id"] for r in before} & {r["id"] for r in after}
    pairs = {r["id"]: r.get("pair_id") for r in after}
    return {qid for qid in shared if half_of(qid, pairs.get(qid)) == which}


def paired_delta_half(before: list[dict], after: list[dict], level: str, which: str) -> dict:
    kept = half_ids(before, after, which)
    out = paired_delta(
        [r for r in before if r["id"] in kept],
        [r for r in after if r["id"] in kept],
        level,
    )
    out["half"] = which
    out["repos"] = len({r.get("repo") for r in after if r["id"] in kept and r.get("repo")})
    out["ids_hash"] = ids_hash(kept)
    return out


def paired_delta(before: list[dict], after: list[dict], level: str) -> dict:
    key = f"{level}_rank"
    if level == "section":
        before = [r for r in before if r.get("section_scorable")]
        after = [r for r in after if r.get("section_scorable")]
    was = {r["id"]: r for r in before}
    paired = [(was[r["id"]], r) for r in after if r["id"] in was]
    if not paired:
        return {"error": "no shared questions"}
    ids = [r["id"] for r in after if r["id"] in was]
    deltas = deltas_over(
        {r["id"]: rr(r[key]) for r in before},
        {r["id"]: rr(r[key]) for r in after},
        ids,
    )
    low, high = bootstrap_ci(deltas)
    return {
        "level": level,
        "questions": len(paired),
        "delta_MRR": round(sum(deltas) / len(deltas), 4),
        "ci95": [round(low, 4), round(high, 4)],
        **tally(deltas),
    }


def summarise(rows: list[dict], level: str) -> dict:
    key = f"{level}_rank"
    if level == "section":
        rows = [r for r in rows if r.get("section_scorable")]
    if not rows:
        return {}
    stats = {f"hit@{c}": round(sum(1 for r in rows if r[key] and r[key] <= c) / len(rows), 4) for c in CUTOFFS}
    stats[f"MRR@{DEPTH}"] = round(sum(rr(r[key]) for r in rows) / len(rows), 4)
    stats["n"] = len(rows)
    return stats


# the product grows by multiplication, and the grids run so far hold two to six arms
GRID_CAP = 32


def arms(axes: dict) -> list[dict]:
    names = sorted(axes)
    size = 1
    for n in names:
        size *= len(axes[n])
    if size > GRID_CAP:
        raise ValueError(f"grid of {size} arms is over the cap of {GRID_CAP}: {axes}")
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*(axes[n] for n in names))]


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
    "distance_threshold": lambda v: isinstance(v, int | float) and not isinstance(v, bool) and 0 <= v <= 2,
    "source": lambda v: isinstance(v, str) and bool(v),
    "variant": lambda v: isinstance(v, str) and v in config.settings.corpus.variants,
    "keyword_query": lambda v: v in KEYWORD_QUERY_MODES,
    "keyword_translation": lambda v: v in TRANSLATION_MODES,
    "max_term_share": lambda v: (isinstance(v, int | float) and not isinstance(v, bool)
                                 and (v == 0 or TERM_SHARE_FLOOR <= v <= 1)),
    "keyword_aliases": lambda v: isinstance(v, bool),
}
AXIS_LIMITS = {
    "ef_search": "a whole number 1..1000 (what hnsw.ef_search accepts)",
    "limit_vector": "a whole number 1 or more",
    "limit_keyword": "a whole number 1 or more",
    "rerank_top": "a whole number 0 or more",
    "distance_threshold": "a number 0..2",
    "source": "a non-empty name",
    "variant": "a variant declared in config",
    "keyword_query": "and or or",
    "keyword_translation": "off, beside or replaces",
    "max_term_share": f"0 (off) or a share {TERM_SHARE_FLOOR}..1",
    "keyword_aliases": "true or false",
}


TRANSLATION_MODES = ("off", "beside", "replaces")


# an arm's keyword axis and the config field it sets, with how the axis value becomes the field's
KEYWORD_AXES = (
    ("keyword_query", ("query",), lambda v: v),
    ("keyword_translation", ("translation", "enabled"), lambda v: v != "off"),
    ("keyword_translation", ("translation", "replaces"), lambda v: v == "replaces"),
    ("max_term_share", ("max_term_share",), lambda v: v),
    ("keyword_aliases", ("aliases", "enabled"), lambda v: v),
)


def _holder(path: tuple) -> tuple:
    node = config.settings.retrieval.keyword
    for name in path[:-1]:
        node = getattr(node, name)
    return node, path[-1]


# the keyword switches an arm names, set for its measuring only: the search reads them from config
@contextlib.contextmanager
def keyword_settings(arm: dict):
    was = [(path, getattr(*_holder(path))) for _, path, _ in KEYWORD_AXES]
    try:
        for axis, path, value_of in KEYWORD_AXES:
            if axis in arm:
                # the settings are frozen for everyone else; an arm's value passed AXIS_RULES before it got here
                object.__setattr__(*_holder(path), value_of(arm[axis]))
        yield
    finally:
        for path, value in was:
            object.__setattr__(*_holder(path), value)


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
        "aliases_fired": row.get("aliases_fired"),
    }


# the procedure, not the corpus: two cuts differ in `variant` and `fingerprint` anyway
COMPARABLE = (
    "set",
    "search",
    "candidates",
    "limit_vector",
    "limit_keyword",
    "distance_threshold",
    "keyword",
    "questions_hash",
    "rerank_top",
)
# absent is not a difference: it is the value the run had before anyone wrote it down
ABSENT_MEANS = {"rerank_top": 0}
# two arms may differ in the axis of record and nothing else; `ef_search` names differ
AXIS_FIELD = {"ef_search": "search", "keyword_query": "keyword", "keyword_translation": "keyword",
              "max_term_share": "keyword", "keyword_aliases": "keyword"}


def comparable(before, after) -> list[tuple]:
    return [
        (field, was, now)
        for field in COMPARABLE
        for was in [before.get(field, ABSENT_MEANS.get(field))]
        for now in [after.get(field, ABSENT_MEANS.get(field))]
        if was != now
    ]


def _switches_of(arm: dict) -> dict:
    with keyword_settings(arm):
        return config.keyword_switches()


# what a rare cut read its shares over; a stale count refuses the arm rather than measuring another corpus
def _term_counts(arm: dict, variant: str) -> dict | None:
    if not _switches_of(arm)["max_term_share"]:
        return None
    import term_frequencies

    if why := term_frequencies.stale(variant):
        raise ValueError(f"{why}: queue count_terms for {variant} before this arm")
    return term_frequencies.counted(variant)


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
        "keyword": _switches_of(arm),
        "term_counts": _term_counts(arm, variant),
        "questions": len(rows),
        "questions_hash": ids_hash(r["id"] for r in rows),
        # descriptive, never compared: what cut these rows is not recoverable from them
        "policy": config.settings.corpus.policy_or_none(variant),
        "fingerprint": db.fingerprint_or_none(variant=variant),
    }


# the name a grown set is recognised by, and the report computed its own copy of it
def ids_hash(ids) -> str:
    import hashlib

    joined = ",".join(str(i) for i in sorted(ids))
    return hashlib.md5(joined.encode(), usedforsecurity=False).hexdigest()[:12]


# compared against the arm differing only in the axis of record, one per combination
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
    # product() of nothing yields one empty tuple: an empty axes dict ran a nameless arm
    if not experiment.axes:
        raise ValueError("no axes to compare")
    # the route validated these on the way in, and a retry re-reads them long after
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
    # without this the record lands with `deltas: {}` and no complaint
    if not param or param not in experiment.axes:
        raise ValueError(f"param must name one of the axes, got {param!r} against {sorted(experiment.axes)}")

    # before the connection: a cancelled job must not open one to find out it is cancelled
    job_id = getattr(experiment, "job_id", None)
    if job_id is not None and job_queue.is_cancelled(job_id):
        raise RuntimeError("comparison cancelled before it measured anything")

    POOL_SHORTFALL.clear()
    db.SECTIONS_UNDER.clear()
    measured, summary, kept = {}, {}, {}
    with contextlib.ExitStack() as stack:
        conn = stack.enter_context(engine.connect())
        for arm in grid:
            if job_id is not None and job_queue.is_cancelled(job_id):
                raise RuntimeError(f"comparison cancelled after {len(measured)} arms")
            name = arm_name(arm)
            exact, ef = depth_of(arm)
            with keyword_settings(arm):
                rows = measure(
                    corpus_search,
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
            # empty means the pair differs in the axis of record and in nothing else
            "not_comparable": [
                {"field": field, "was": was, "now": now}
                for field, was, now in comparable(
                    {**arm_procedure(against, measured[base], experiment.dataset), AXIS_FIELD.get(param, param): None},
                    {**arm_procedure(arm, measured[name], experiment.dataset), AXIS_FIELD.get(param, param): None},
                )
            ],
            **{level: paired_delta(measured[base], measured[name], level) for level in ("file", "section")},
            # the winner is chosen on A and reported on B, so the whole alone cannot answer
            "halves": {
                level: {which: paired_delta_half(measured[base], measured[name], level, which) for which in ("A", "B")}
                for level in ("file", "section")
            },
        }
    return {
        "reference_axis": param,
        # per question, per arm: a delta cannot be recomputed under another reference without
        "rows": kept,
        "arms": summary,
        "deltas": deltas,
        # a pool smaller than asked means something capped the search
        "pool_shortfall": len(POOL_SHORTFALL),
        "procedure": {
            "schema": SCHEMA,
            "dataset": experiment.dataset,
            "axes": experiment.axes,
            "param": param,
            # per arm: arms differ in variant, depth and, along `source`, in the questions
            "arms": {arm_name(a): arm_procedure(a, measured[arm_name(a)], experiment.dataset) for a in grid},
        },
    }


def _suffix(value) -> str:
    return str(value).replace(".", "").replace("/", "_")


def for_reading(results: dict) -> dict:
    return {
        "source_run": None,
        # each arm against its own reference, question by question
        "pairing": "by question",
        "multiplicity": None,
        "ranking": None,
        "arms": results.get("arms") or {},
        "deltas": results.get("deltas") or {},
        # the arms' code: every reader answers with the same keys, filled or empty
        "code": results.get("code") or {},
    }
