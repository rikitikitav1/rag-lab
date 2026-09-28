import re
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import config
import logging_setup
from errors import StandFault
from langdetect import DetectorFactory, LangDetectException, detect
from orm.sync_db import engine
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

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


# every config knows its own function words, so ask them instead of guessing the language
FUNCTION_WORDS = """
SELECT cfg, coalesce(array_length(tsvector_to_array(to_tsvector(cfg::regconfig, :q)), 1), 0) AS kept
FROM unnest(CAST(:configs AS text[])) cfg
ORDER BY kept, cfg
"""


def _code_of(cfg: str, fts) -> str:
    return next((code for code, name in fts.languages.items() if name == cfg), "en")


def _by_alphabet(text_, fts) -> str:
    letters = [c for c in text_ if c.isalpha()]
    cyrillic = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    return "ru" if letters and cyrillic / len(letters) >= 0.3 else "en"


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
    from use_cases.index import vector_index_name

    dropped = conn.execute(text("DELETE FROM data_chunks WHERE variant = :variant"), {"variant": variant}).rowcount
    # an empty partial index left behind makes the next index of the name insert row by row
    conn.execute(text(f"DROP INDEX IF EXISTS {vector_index_name(variant)}"))
    # a row kept by another variant must not say this one was cut by some file
    conn.execute(text("UPDATE data_sources SET indexed_with = indexed_with - :variant"), {"variant": variant})
    return dropped


def cleanup(*, variant):
    with engine.begin() as conn:
        _drop_variant(conn, variant)
        # only a row the code's source files made and no chunk holds goes; an added source keeps its origin and report
        conn.execute(
            text("""
                DELETE FROM data_sources ds
                WHERE ds.origin IS NULL
                  AND NOT EXISTS (SELECT 1 FROM data_chunks dc WHERE dc.source_id = ds.id)
            """)
        )


def remove_variant(variant) -> int:
    with engine.begin() as conn:
        return _drop_variant(conn, variant)


# the row and its chunks in every variant; the chunks go by the foreign key's cascade
def remove_source(source_id: int) -> int:
    with engine.begin() as conn:
        chunks = conn.execute(
            text("SELECT count(*) FROM data_chunks WHERE source_id = :id"), {"id": source_id}
        ).scalar()
        conn.execute(text("DELETE FROM data_sources WHERE id = :id"), {"id": source_id})
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


def remove_question_set(set_name: str) -> int:
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE questions SET source_question_id = NULL WHERE set_name = :s AND source_question_id IN "
                "(SELECT id FROM questions WHERE set_name = :s)"
            ),
            {"s": set_name},
        )
        return conn.execute(text("DELETE FROM questions WHERE set_name = :s"), {"s": set_name}).rowcount


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
        marks = conn.execute(
            text("SELECT marked_sources FROM questions WHERE cardinality(marked_sources) > 0")
        ).scalars()
        return count_marking(files, marks)


# marks no searched chunk holds, by the stand's gold predicate (a file or a folder prefix), as the preflight reads them
def unreachable_marks(marks: list[str], *, variant: str) -> list[str]:
    if not marks:
        return []
    query = f"""SELECT m FROM unnest(CAST(:marks AS text[])) m
                WHERE NOT EXISTS (SELECT 1 FROM data_chunks dc WHERE {live_rows("dc")} AND position(m in dc.source) > 0)
                ORDER BY m"""
    with engine.connect() as conn:
        return list(conn.execute(text(query), {"marks": sorted(set(marks)), "variant": variant}).scalars())


def count_marking(files: list[str], marks) -> int:
    from evals.retrieval_metrics import is_gold

    return sum(1 for marked in marks if any(is_gold(f, marked) for f in files))


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


# one rule for both doors: a literal label; the old dotted path refuses rather than finds nothing
CATEGORY_RE = re.compile(r"^[\w-]+$")


def refuse_bad_category(category: str | None) -> None:
    if category is not None and not CATEGORY_RE.fullmatch(category):
        raise ValueError(f"invalid category filter: {category[:60]!r}")


# a label is a category of the map, a group of them, or a tag; the filter takes any of the three
def categories_of(label: str) -> list[str]:
    cats = config.settings.categories
    return [key for key, row in cats.items() if key == label or row.group == label]


class ScopeRefused(ValueError):
    pass


# what a search may read: a label (a category, a group or a tag), sources by name, one version of one category
@dataclass(frozen=True)
class Scope:
    label: str | None = None
    sources: tuple[str, ...] = ()
    version: str | None = None

    # tags are stored lowercased, so a label asked as «Redis» reads what the index wrote as «redis»
    def __post_init__(self):
        if self.label:
            object.__setattr__(self, "label", self.label.lower())

    @property
    def narrowed(self) -> bool:
        return bool(self.label or self.sources or self.version)


def as_scope(scope) -> Scope:
    if scope is None:
        return Scope()
    return scope if isinstance(scope, Scope) else Scope(label=scope)


# a version is a category's, so it needs one; a category without versions or a version it lacks is said, not emptied
def refuse_bad_scope(scope: Scope) -> None:
    refuse_bad_category(scope.label)
    if scope.sources:
        _refuse_sources_out_of_search(scope.sources)
    if scope.version is None:
        return
    if not scope.label:
        raise ScopeRefused(f"version {scope.version} names no category; add the category it belongs to")
    cats = categories_of(scope.label)
    if len(cats) != 1:
        raise ScopeRefused(f"a version belongs to one category, and {scope.label} names {len(cats)}")
    listed = config.settings.categories[cats[0]].versions
    if not listed:
        raise ScopeRefused(f"{cats[0]} has no versions declared")
    if scope.version not in listed:
        raise ScopeRefused(f"{cats[0]} has no version {scope.version}; listed: {listed}")


def _newest() -> tuple[list[str], list[str]]:
    rows = [(key, row.versions[0]) for key, row in config.settings.categories.items() if row.versions]
    return [k for k, _ in rows], [v for _, v in rows]


# without a version a versioned category answers from its newest released one; a rolling source has no versions
def _scope_filter(scope: Scope) -> tuple[str, dict]:
    cats, newest = _newest()
    params = {"newest_categories": cats, "newest_versions": newest}
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
        sql += (
            " AND (cardinality(versions) = 0 OR EXISTS (SELECT 1 FROM unnest("
            "CAST(:newest_categories AS text[]), CAST(:newest_versions AS text[])) AS n(c, v)"
            " WHERE n.c = category AND n.v = ANY(versions)))"
        )
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
def _refuse_sources_out_of_search(names) -> None:
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
    refuse_bad_scope(scope)
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
                       d.section
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
            if retrieval.filtered_scan != "off":
                conn.execute(text(f"SET LOCAL hnsw.iterative_scan = {retrieval.filtered_scan}"))
        rows = conn.execute(text(query), params).mappings().all()
    return [Hit(**row) for row in rows]


# which language an answer belongs in: asked for, else the one the question is written in
def resolve_language(question: str, language: str | None) -> str:
    return language or detect_language(question)
