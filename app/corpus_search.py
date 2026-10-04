from typing import NamedTuple

import config
import logging_setup
import query_aliases
import query_translation
import search_depth
import search_scope
import text_language
from corpus_keys import SHARED_BODY_SQL
from errors import StandFault
from orm.sync_db import engine
from search_scope import Scope, ScopeRefused, as_scope, categories_of
from sqlalchemy import text

from db import live_rows

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
    # the rank by the question's English words, when the translation is on and the question is Russian
    translated_rank: int | None = None


def _keyword_query_sql(mode: str, text_param: str = "question", config_param: str = "ts_config") -> str:
    if mode == "or":
        # cast, not to_tsquery: a second pass would stem and drop stopwords twice
        return f"""CAST(nullif(replace(
                    plainto_tsquery(CAST(:{config_param} AS regconfig), :{text_param})::text,
                    ' & ', ' | '), '') AS tsquery)"""
    return f"plainto_tsquery(CAST(:{config_param} AS regconfig), :{text_param})"


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
    with engine.connect() as conn:
        refuse_foreign_vectors(conn, variant, embedded_by)
        # on the connection already held: `resolve` opens its own, and the pool is five plus five
        depth = search_depth.resolve(variant, conn=conn)
        conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(depth)}"))
        row = conn.execute(text(query), {"embedding": str(list(embedding)), "variant": variant}).scalar()
    return float(row) if row is not None else None


# the search's own step past the scope's pure rules: the sources it names must be in search now, its version held
def refuse_bad_scope(scope: Scope, variant: str | None = None) -> None:
    search_scope.refuse_malformed_scope(scope)
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


# candidates hold a rare word of the query and the full query ranks them: ranking every common-word match took a search
def _rare_cut(text_param: str, config_param: str) -> str:
    return f"""AND content_tsv @@ coalesce((
                        SELECT to_tsquery('simple', string_agg(quote_literal(w), ' | '))
                        FROM unnest(tsvector_to_array(to_tsvector(CAST(:{config_param} AS regconfig), :{text_param}))) w
                        LEFT JOIN term_frequencies tf ON tf.variant = :variant AND tf.lexeme = w
                        WHERE coalesce(tf.share, 0) <= :max_term_share
                    ), q)"""


# the third ranking RRF reads: the keyword search over the question's English words, the same shape as the first
def _translated_branch(mode: str, rank_fn: str, cat_filter: str, src_filter: str, rare: bool) -> str:
    query = _keyword_query_sql(mode, "translated", "translated_config")
    cut = _rare_cut("translated", "translated_config") if rare else ""
    return f""",
                keyword_translated AS (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               ORDER BY {rank_fn}(content_tsv, q, :keyword_norm) DESC, id
                           ) AS rank
                    FROM data_chunks, {query} q
                    WHERE content_tsv @@ q {cut} {cat_filter} {src_filter}
                    ORDER BY rank
                    LIMIT :limit_keyword
                )"""


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
    reworded, fired = query_aliases.reword(question)
    if fired:
        log.info("keyword.aliases_fired", fired=fired)
    translated = query_translation.keyword_translation(reworded)
    # the translation replaces the question's words or ranks beside them; a reworded English question ranks there too
    keyword_text = translated if translated and retrieval.keyword.translation.replaces else question
    third = translated if translated and keyword_text == question else (reworded if fired else None)
    rare = retrieval.keyword.max_term_share > 0
    cut = _rare_cut("question", "ts_config") if rare else ""
    scope = as_scope(scope)
    refuse_bad_scope(scope, variant)
    cat_filter, cat_params = _scope_filter(scope)
    src_filter = f"AND {live_rows()}"
    # one copy of a text a source holds under several files: the man pages' aliases took three places of five
    copies = "WHERE copy_rank = 1" if retrieval.collapse_copies_in_source else ""
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
                    WHERE content_tsv @@ q {cut} {cat_filter} {src_filter}
                    ORDER BY rank
                    LIMIT :limit_keyword
                ){_translated_branch(retrieval.keyword.query, rank_fn, cat_filter, src_filter, rare) if third else ""}
                -- the candidates first: joining the branches onto the whole table read every chunk on every search
                , candidates AS (
                    SELECT id FROM vector_search UNION SELECT id FROM keyword_search
                    {"UNION SELECT id FROM keyword_translated" if third else ""}
                ), fused AS (
                    SELECT d.id, d.source_id, d.content, d.source, d.category, d.chunk_index,
                           CASE WHEN {SHARED_BODY_SQL.replace("c.", "d.")} THEN d.content_hash END AS content_hash,
                           v.rank AS vector_rank, k.rank AS keyword_rank, v.distance AS distance,
                        COALESCE(1.0/(:rrf_k + v.rank), 0) + COALESCE(1.0/(:rrf_k + k.rank), 0)
                        {"+ COALESCE(1.0/(:rrf_k + t.rank), 0)" if third else ""} AS score,
                           d.section, d.versions, {"t.rank" if third else "NULL::int"} AS translated_rank
                    FROM candidates c
                    JOIN data_chunks d ON d.id = c.id
                    LEFT JOIN vector_search v ON d.id = v.id
                    LEFT JOIN keyword_search k ON d.id = k.id
                    {"LEFT JOIN keyword_translated t ON d.id = t.id" if third else ""}
                ), ranked AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY source_id, coalesce(content_hash, id::text) ORDER BY score DESC, id
                    ) AS copy_rank
                    FROM fused
                )
                SELECT content, source, category, chunk_index, vector_rank, keyword_rank, distance, score,
                       section, versions, translated_rank
                FROM ranked
                {copies}
                -- RRF ties are structural (best by vector and best by keyword both score
                -- 1/(k+1)); without a second key the physical row order decides, and any
                -- migration or vacuum silently reshuffles the answer
                ORDER BY score DESC, id
                LIMIT :limit
                """
    params = {
        "embedding": embedding,
        "question": keyword_text,
        "limit_vector": limit_vector,
        "limit_keyword": limit_keyword,
        "limit": limit,
        "distance_threshold": distance_threshold,
        "variant": variant,
        "rrf_k": config.settings.retrieval.rrf_k,
        "ts_config": text_language.ts_config(keyword_text),
        "keyword_norm": retrieval.keyword.norm,
    }
    if third:
        params |= {"translated": third, "translated_config": text_language.ts_config(third)}
    if rare:
        params["max_term_share"] = retrieval.keyword.max_term_share
    params |= cat_params
    # an argument that names a switch and does not throw it labelled a run exact at 40
    depth = None if exact else search_depth.resolve(variant, ef_search)
    # on this connection: the depth is per request and a pooled connection outlives it
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
