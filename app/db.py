from functools import lru_cache
from typing import NamedTuple

import config
import logging_setup
import search_scope
from corpus_keys import GOLD_SQL, HAS_GOLD_SQL, Gold, exact_gold_sql, language_by_alphabet, vector_index_name
from errors import Final, StandFault
from langdetect import DetectorFactory, LangDetectException, detect
from orm.sync_db import engine
from search_scope import (  # noqa: F401
    CATEGORY_RE,
    VERSION_RE,
    Scope,
    ScopeRefused,
    as_scope,
    categories_of,
    refuse_bad_category,
)
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

DetectorFactory.seed = 0
log = logging_setup.get_logger(__name__)

RANK_FUNCTIONS = {"ts_rank", "ts_rank_cd"}


# built by name, so a column added anywhere but the end cannot move a reader's `row[6]`
class Hit(NamedTuple):
    content: str
    source: str
    category: str | None
    chunk_index: int | None
    vector_rank: int | None
    keyword_rank: int | None
    distance: float | None
    score: float
    section: str | None
    # the releases the chunk stands for, so a gold's version is read on the row the search returned
    versions: tuple = ()


# every config knows its own function words, so ask them instead of guessing the language
FUNCTION_WORDS = """
SELECT cfg, coalesce(array_length(tsvector_to_array(to_tsvector(cfg::regconfig, :q)), 1), 0) AS kept
FROM unnest(CAST(:configs AS text[])) cfg
ORDER BY kept, cfg
"""


def _code_of(cfg: str, fts) -> str:
    return next((code for code, name in fts.languages.items() if name == cfg), "en")


def _by_alphabet(text_, fts) -> str:
    return language_by_alphabet(text_)


def _by_function_words(text_, fts) -> str | None:
    """The config that drops the most tokens recognised them as its own stopwords."""
    configs = sorted(set(fts.languages.values()) | {fts.fallback})
    if len(configs) < 2:
        return None
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(FUNCTION_WORDS), {"q": text_, "configs": configs}).all()
    except SQLAlchemyError as e:
        # picking a language must not need a database: the alphabet rule answers on its own
        log.warning("db.function_words_unavailable", error=str(e))
        return None
    # a tie means no function word of any candidate showed up: this rule has nothing to say
    return None if rows[0][1] == rows[1][1] else _code_of(rows[0][0], fts)


def detect_language(text_, mode=None) -> str:
    """One rule for the search config and for the language the answer comes back in."""
    return _detect(text_, mode or config.settings.retrieval.keyword.query_lang)


# a hop asks for the same question several times, and function_words costs a round trip
@lru_cache(maxsize=4096)
def _detect(text_: str, mode: str) -> str:
    fts = config.settings.fts
    if mode == "function_words":
        return _by_function_words(text_, fts) or _by_alphabet(text_, fts)
    if mode == "cyrillic_ratio":
        # langdetect misreads short mixed-script questions, and a wrong config kills the match
        return _by_alphabet(text_, fts)
    fallback = _code_of(fts.fallback, fts)
    try:
        # a language we cannot search is a language we should not answer in either
        code = detect(text_)
    except LangDetectException:
        return fallback
    return code if code in fts.languages else fallback


def _ts_config(text_, mode=None):
    fts = config.settings.fts
    return fts.languages.get(detect_language(text_, mode), fts.fallback)


def _keyword_query_sql(mode: str) -> str:
    if mode == "or":
        # cast, not to_tsquery: a second pass would stem and drop stopwords twice
        return """CAST(nullif(replace(
                    plainto_tsquery(CAST(:ts_config AS regconfig), :question)::text,
                    ' & ', ' | '), '') AS tsquery)"""
    return "plainto_tsquery(CAST(:ts_config AS regconfig), :question)"


# this variant, under a source still active: seven places wrote it and one had drifted
def live_rows(alias: str = "") -> str:
    col = f"{alias}." if alias else ""
    return f"{col}variant = :variant AND {col}source_id IN (SELECT id FROM data_sources WHERE active)"


def _drop_variant(conn, variant) -> int:
    dropped = conn.execute(text("DELETE FROM data_chunks WHERE variant = :variant"), {"variant": variant}).rowcount
    # an empty partial index left behind makes the next index of the name insert row by row
    conn.execute(text(f"DROP INDEX IF EXISTS {vector_index_name(variant)}"))
    # a row kept by another variant must not say this one was cut by some file
    conn.execute(text("UPDATE data_sources SET indexed_with = indexed_with - :variant"), {"variant": variant})
    return dropped


def cleanup(*, variant):
    with engine.begin() as conn:
        # a row is the source's declaration now, so emptying a variant never deletes one
        _drop_variant(conn, variant)


def remove_variant(variant) -> int:
    with engine.begin() as conn:
        return _drop_variant(conn, variant)


# the row and its chunks in every variant; the chunks go by the foreign key's cascade
def remove_source(source_id: int) -> int:
    try:
        with engine.begin() as conn:
            chunks = conn.execute(
                text("SELECT count(*) FROM data_chunks WHERE source_id = :id"), {"id": source_id}
            ).scalar()
            conn.execute(text("DELETE FROM data_sources WHERE id = :id"), {"id": source_id})
    except IntegrityError as e:
        raise Final(f"source {source_id} was taken by another row while it was being removed; nothing removed") from e
    return chunks


# a set's size, its questions that answer logs hold, and questions of other sets drawn from it
def question_set_holds(set_name: str) -> dict:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT count(*), "
                "(SELECT count(*) FROM question_logs l JOIN questions q ON q.id = l.question_id "
                "WHERE q.set_name = :s), "
                "(SELECT count(*) FROM questions c JOIN questions o ON o.id = c.source_question_id "
                "WHERE o.set_name = :s AND c.set_name <> :s) "
                "FROM questions WHERE set_name = :s"
            ),
            {"s": set_name},
        ).one()
    return {"questions": row[0], "answered": row[1], "drawn_from": row[2]}


# the door checks the holds first; a log or a set that took a question since is a refusal, not a server error
def remove_question_set(set_name: str) -> int:
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE questions SET source_question_id = NULL WHERE set_name = :s AND source_question_id IN "
                    "(SELECT id FROM questions WHERE set_name = :s)"
                ),
                {"s": set_name},
            )
            return conn.execute(text("DELETE FROM questions WHERE set_name = :s"), {"s": set_name}).rowcount
    except IntegrityError as e:
        raise Final(f"questions of {set_name} were taken by a log or another set while it was being removed") from e


# questions whose gold lies in the source, by the stand's own predicate: a mark is a file or a folder prefix
def questions_marking(source_id: int) -> int:
    with engine.connect() as conn:
        files = (
            conn.execute(text("SELECT DISTINCT source FROM data_chunks WHERE source_id = :id"), {"id": source_id})
            .scalars()
            .all()
        )
        if not files:
            return 0
        rows = conn.execute(
            text(f"SELECT marked_sources, gold FROM questions q WHERE {HAS_GOLD_SQL.format(q='q')}")
        ).all()
    return count_marking(files, [Gold.of(marks, gold) for marks, gold in rows])


# marks no searched chunk holds, by the stand's gold predicate (a file or a folder prefix), as the preflight reads them
def unreachable_marks(marks: list[str], *, variant: str) -> list[str]:
    if not marks:
        return []
    gold = GOLD_SQL.format(mark="m", source="dc.source")
    query = f"""SELECT m FROM unnest(CAST(:marks AS text[])) m
                WHERE NOT EXISTS (SELECT 1 FROM data_chunks dc WHERE {live_rows("dc")} AND {gold})
                ORDER BY m"""
    with engine.connect() as conn:
        return list(conn.execute(text(query), {"marks": sorted(set(marks)), "variant": variant}).scalars())


# the exact gold of a questions row `q` against a chunk `dc`, inside one query
_EXACT_GOLD_OF_Q = exact_gold_sql(
    "q.gold->>'file'", "q.gold->>'section'", "q.gold->>'version'", "dc.source", "dc.section", "dc.versions"
)


# questions per set whose gold no searched chunk holds, by the stand's one gold rule: a run on such a set misreads
def unreachable_by_set(*, variant: str) -> list[tuple[str | None, int]]:
    marked = GOLD_SQL.format(mark="m", source="dc.source")
    # two branches apart: one OR over both made the planner scan the chunks per question and hit the timeout
    query = f"""SELECT set_name, count(*) FROM (
                  SELECT q.set_name FROM questions q WHERE cardinality(q.marked_sources) > 0 AND NOT EXISTS (
                    SELECT 1 FROM data_chunks dc, unnest(q.marked_sources) m WHERE {live_rows("dc")} AND {marked})
                  UNION ALL
                  SELECT q.set_name FROM questions q WHERE q.gold IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM data_chunks dc WHERE {live_rows("dc")} AND {_EXACT_GOLD_OF_Q})
                ) unreachable GROUP BY set_name ORDER BY 2 DESC, 1"""
    with engine.connect() as conn:
        return [(name, n) for name, n in conn.execute(text(query), {"variant": variant}).all()]


def count_marking(files: list[str], golds) -> int:
    return sum(1 for gold in golds if any(Gold.coerce(gold).holds_file(f) for f in files))


def is_empty(*, variant):
    with engine.connect() as conn:
        return conn.execute(
            text(f"""SELECT NOT EXISTS (
                      SELECT 1 FROM data_chunks WHERE {live_rows()})"""),
            {"variant": variant},
        ).scalar()


# losing the bookkeeping must not cost the work: three call sites wrote this try/except
def fingerprint_or_none(*, variant) -> dict | None:
    try:
        return corpus_fingerprint(variant=variant)
    except Exception as e:
        log.error("corpus.fingerprint_failed", variant=variant, error=str(e))
        return None


# thresholds are calibrated against a corpus, so a run has to record which one it saw
def corpus_fingerprint(*, variant) -> dict:
    query = f"""
        SELECT count(*) AS chunks,
               count(DISTINCT source_id) AS sources,
               max(id) AS last_chunk_id
        FROM data_chunks WHERE {live_rows()}
    """
    with engine.connect() as conn:
        row = conn.execute(text(query), {"variant": variant}).mappings().one()
    return {"variant": variant, **dict(row)}


# `is_empty` asks whether any row exists, and a half-built variant has plenty
def sources_missing_from(*, variant) -> list[str]:
    with engine.connect() as conn:
        return [
            row[0]
            for row in conn.execute(
                text("""SELECT ds.name FROM data_sources ds
                        WHERE ds.active
                          AND NOT EXISTS (
                            SELECT 1 FROM data_chunks dc
                            WHERE dc.source_id = ds.id AND dc.variant = :variant)
                        ORDER BY ds.name"""),
                {"variant": variant},
            )
        ]


# a run on the wrong variant otherwise looks like an ordinary run with different numbers
def corpus_variants() -> list[dict]:
    query = """
        SELECT variant, count(*) AS chunks, count(DISTINCT source_id) AS sources
        FROM data_chunks GROUP BY variant ORDER BY variant
    """
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(query)).mappings()]


class ForeignVectors(StandFault):
    pass


# a vector meets only vectors of its own embedder: nothing else refused a search across two of them
def refuse_foreign_vectors(conn, variant: str, embedded_by: str) -> None:
    # an unmarked vector is refused; marks come off the index, a filter on `embedding` cost 26 ms
    seen = (
        conn.execute(
            text(
                "SELECT DISTINCT embedded_by FROM data_chunks"
                " WHERE variant = :variant AND embedded_by IS NOT NULL"
                " UNION ALL SELECT NULL WHERE EXISTS (SELECT 1 FROM data_chunks WHERE variant = :variant"
                " AND embedded_by IS NULL AND embedding IS NOT NULL)"
            ),
            {"variant": variant},
        )
        .scalars()
        .all()
    )
    foreign = sorted(label or "no recorded embedder" for label in set(seen) - {embedded_by})
    if foreign:
        raise ForeignVectors(
            f"variant {variant!r} holds vectors of {', '.join(foreign)} and this search embeds"
            f" with {embedded_by}: reindex the variant, or give the embedding role back"
        )


# the caller names the embedder: asked from here, it took a second pooled connection per search
def nearest_distance(embedding, *, variant, embedded_by: str) -> float | None:
    # same filters as hybrid_search: the topic axis must not see what retrieval cannot
    query = f"""
        SELECT embedding <=> CAST(:embedding AS vector) AS distance
        FROM data_chunks
        WHERE embedding IS NOT NULL AND {live_rows()}
        ORDER BY distance
        LIMIT 1
    """
    from use_cases import search_depth

    with engine.connect() as conn:
        refuse_foreign_vectors(conn, variant, embedded_by)
        # on the connection already held: `resolve` opens its own, and the pool is five plus five
        depth = search_depth.resolve(variant, conn=conn)
        conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(depth)}"))
        row = conn.execute(text(query), {"embedding": str(list(embedding)), "variant": variant}).scalar()
    return float(row) if row is not None else None


# the search's own step past the scope's pure rules: the sources it names must be in search now, its version held
def refuse_bad_scope(scope: Scope, variant: str | None = None) -> None:
    search_scope.refuse_bad_scope(scope)
    if scope.sources:
        refuse_sources_out_of_search(scope.sources)
    if scope.version:
        refuse_unheld_version(scope, variant or config.settings.corpus.variant)


# a version no searched source holds is refused, though a book for any version would answer: it is not that version's
def refuse_unheld_version(scope: Scope, variant: str) -> None:
    (category,) = search_scope.categories_of(scope.label)
    query = f"""SELECT EXISTS (SELECT 1 FROM data_chunks
                WHERE {live_rows()} AND category = :category AND :version = ANY(versions))"""
    with engine.connect() as conn:
        held = conn.execute(text(query), {"variant": variant, "category": category, "version": scope.version}).scalar()
    if not held:
        raise ScopeRefused(f"no source in search holds {category} {scope.version}")


# a chunk holds its category's newest version, the one clause the default scope and the older-version check read
_HOLDS_NEWEST = (
    "EXISTS (SELECT 1 FROM unnest(CAST(:newest_categories AS text[]), CAST(:newest_versions AS text[])) AS n(c, v)"
    " WHERE n.c = category AND n.v = ANY(versions))"
)


# the newest clause removes rows only once an older version is in search: then a scan cut at ef_search runs short
def older_versions_held(variant: str, conn=None) -> bool:
    cats, latest = newest()
    # the partial index on versioned rows answers this on every search, so nothing is cached between searches
    query = f"""SELECT EXISTS (SELECT 1 FROM data_chunks WHERE {live_rows()} AND cardinality(versions) > 0
                AND NOT {_HOLDS_NEWEST})"""
    params = {"variant": variant, "newest_categories": cats, "newest_versions": latest}
    if conn is not None:
        return bool(conn.execute(text(query), params).scalar())
    with engine.connect() as own:
        return bool(own.execute(text(query), params).scalar())


def newest() -> tuple[list[str], list[str]]:
    rows = [(key, row.versions[0]) for key, row in config.settings.categories.items() if row.versions]
    return [k for k, _ in rows], [v for _, v in rows]


# the versions each category holds where a search reads, so the preflight's newest check reads no inactive source
def versions_held(variant: str) -> dict[str, list[str]]:
    query = f"""SELECT category, array_agg(DISTINCT v) FROM data_chunks, unnest(versions) v
                WHERE {live_rows()} GROUP BY category"""
    with engine.connect() as conn:
        return {c: sorted(vs) for c, vs in conn.execute(text(query), {"variant": variant}).all()}


# without a version a versioned category answers from its newest released one; a rolling source has no versions
def _scope_filter(scope: Scope) -> tuple[str, dict]:
    cats, latest = newest()
    params = {"newest_categories": cats, "newest_versions": latest}
    sql = ""
    if scope.label:
        sql += " AND (category = ANY(:categories) OR :label = ANY(tags))"
        params |= {"categories": categories_of(scope.label), "label": scope.label}
    if scope.sources:
        sql += " AND source_id IN (SELECT id FROM data_sources WHERE name = ANY(:scope_sources))"
        params["scope_sources"] = list(scope.sources)
    if scope.version:
        # a book or a sheet of the category holds for every version, so it stays beside the version's own docs
        sql += " AND (:scope_version = ANY(versions) OR cardinality(versions) = 0)"
        params["scope_version"] = scope.version
    else:
        sql += f" AND (cardinality(versions) = 0 OR {_HOLDS_NEWEST})"
    return sql, params


# rows of (label, group, chunks); a chunk of no category counts under "none", so the unmapped mass stays in view
def list_categories(category=None, only_top=None, *, variant):
    params = {"variant": variant}
    cat_filter = ""
    if category:
        cat_filter = "AND category = ANY(:categories)"
        params["categories"] = categories_of(category)
    query = f"""SELECT category, COUNT(*) FROM data_chunks
                WHERE {live_rows()} {cat_filter}
                GROUP BY category"""
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
    groups = {key: row.group for key, row in config.settings.categories.items()}
    named = [(c or "none", groups.get(c, "none"), n) for c, n in rows]
    if only_top:
        totals: dict[str, int] = {}
        for _, group, n in named:
            totals[group] = totals.get(group, 0) + n
        named = [(group, group, n) for group, n in totals.items()]
    return sorted(named)


# the tags the filter also takes, most used first, so a client can discover them as it does the categories
def list_tags(limit: int, *, variant):
    query = f"""SELECT tag, COUNT(*) AS n FROM data_chunks, unnest(tags) AS tag
                WHERE {live_rows()}
                GROUP BY tag ORDER BY n DESC, tag LIMIT :limit"""
    with engine.connect() as conn:
        return conn.execute(text(query), {"variant": variant, "limit": limit}).fetchall()


# a name the base lacks or the search cannot read (declared, raw, inactive) refuses before the embed is paid
def refuse_sources_out_of_search(names) -> None:
    with engine.connect() as conn:
        rows = dict(
            conn.execute(
                text("SELECT name, active AND stage = 'accepted' FROM data_sources WHERE name = ANY(:n)"),
                {"n": list(names)},
            ).all()
        )
    if missing := sorted(set(names) - set(rows)):
        raise ScopeRefused(f"no source named {missing}")
    if outside := sorted(n for n, searched in rows.items() if not searched):
        raise ScopeRefused(f"{outside} are not in search: not accepted or not active")


# a scan cut at ef_search keeps few rows of a narrow scope, and the fusion quietly turns into keyword search alone
def filtered_scan(scope: Scope, configured: str, older_held: bool = False) -> str:
    if configured == "off" and (scope.sources or scope.version or older_held):
        # strict: the vector leg numbers its rows in the order the index hands them
        return "strict_order"
    return configured


def hybrid_search(
    question,
    embedding,
    scope=None,
    limit_vector=config.settings.retrieval.limit_vector,
    limit_keyword=config.settings.retrieval.limit_keywords,
    limit=None,
    *,
    variant,
    distance_threshold=None,
    ef_search=None,
    exact=False,
    embedded_by: str,
):
    retrieval = config.settings.retrieval
    limit = limit or retrieval.results_limit
    if distance_threshold is None:
        distance_threshold = retrieval.distance_threshold
    rank_fn = retrieval.keyword.rank
    if rank_fn not in RANK_FUNCTIONS:
        raise ValueError(f"keyword_rank must be one of {sorted(RANK_FUNCTIONS)}")
    keyword_query = _keyword_query_sql(retrieval.keyword.query)
    scope = as_scope(scope)
    refuse_bad_scope(scope, variant)
    cat_filter, cat_params = _scope_filter(scope)
    src_filter = f"AND {live_rows()}"
    query = f"""WITH vector_search AS (
                    SELECT id,
                           embedding <=> CAST(:embedding AS vector) AS distance,
                           ROW_NUMBER() OVER (
                               ORDER BY embedding <=> CAST(:embedding AS vector) ASC, id
                           ) AS rank
                    FROM data_chunks
                    WHERE embedding <=> CAST(:embedding AS vector) <= :distance_threshold
                      {cat_filter} {src_filter}
                    ORDER BY distance, id
                    LIMIT :limit_vector
                ),
                keyword_search AS (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               ORDER BY {rank_fn}(content_tsv, q, :keyword_norm) DESC, id
                           ) AS rank
                    FROM data_chunks, {keyword_query} q
                    WHERE content_tsv @@ q {cat_filter} {src_filter}
                    ORDER BY rank
                    LIMIT :limit_keyword
                )
                SELECT d.content, d.source, d.category, d.chunk_index,
                       v.rank AS vector_rank, k.rank AS keyword_rank, v.distance AS distance,
                    COALESCE(1.0/(:rrf_k + v.rank), 0) + COALESCE(1.0/(:rrf_k + k.rank), 0) AS score,
                       d.section, d.versions
                FROM data_chunks d
                LEFT JOIN vector_search v ON d.id = v.id
                LEFT JOIN keyword_search k ON d.id = k.id
                WHERE v.id IS NOT NULL OR k.id IS NOT NULL
                -- RRF ties are structural (best by vector and best by keyword both score
                -- 1/(k+1)); without a second key the physical row order decides, and any
                -- migration or vacuum silently reshuffles the answer
                ORDER BY score DESC, d.id
                LIMIT :limit
                """
    params = {
        "embedding": embedding,
        "question": question,
        "limit_vector": limit_vector,
        "limit_keyword": limit_keyword,
        "limit": limit,
        "distance_threshold": distance_threshold,
        "variant": variant,
        "rrf_k": config.settings.retrieval.rrf_k,
        "ts_config": _ts_config(question),
        "keyword_norm": retrieval.keyword.norm,
    }
    params |= cat_params
    # on this connection: the depth is per request and a pooled connection outlives it
    from use_cases import search_depth

    # an argument that names a switch and does not throw it labelled a run exact at 40
    depth = None if exact else search_depth.resolve(variant, ef_search)
    with engine.connect() as conn:
        refuse_foreign_vectors(conn, variant, embedded_by)
        if exact:
            conn.execute(text("SET LOCAL enable_indexscan = off"))
        else:
            conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(depth)}"))
            # the default version clause filters every search once a versioned source is in, asked or not
            older = not scope.version and older_versions_held(variant, conn)
            if (scan := filtered_scan(scope, retrieval.filtered_scan, older)) != "off":
                conn.execute(text(f"SET LOCAL hnsw.iterative_scan = {scan}"))
        rows = conn.execute(text(query), params).mappings().all()
    return [Hit(**{**row, "versions": tuple(row["versions"] or ())}) for row in rows]


# which language an answer belongs in: asked for, else the one the question is written in
def resolve_language(question: str, language: str | None) -> str:
    return language or detect_language(question)
