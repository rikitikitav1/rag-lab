from functools import lru_cache

import config
import logging_setup
from langdetect import DetectorFactory, LangDetectException, detect
from orm.sync_db import engine
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

DetectorFactory.seed = 0
log = logging_setup.get_logger(__name__)

RANK_FUNCTIONS = {"ts_rank", "ts_rank_cd"}


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
    return _detect(text_, mode or config.settings.retrieval.query_lang)


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


def cleanup(*, variant):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM data_chunks WHERE variant = :variant"), {"variant": variant})
        conn.execute(
            text("""
                DELETE FROM data_sources ds
                WHERE NOT EXISTS (SELECT 1 FROM data_chunks dc WHERE dc.source_id = ds.id)
            """)
        )


def is_empty(*, variant):
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT NOT EXISTS (SELECT 1 FROM data_chunks WHERE variant = :variant)"),
            {"variant": variant},
        ).scalar()


# thresholds are calibrated against a corpus, so a run has to record which one it saw
def corpus_fingerprint(*, variant) -> dict:
    query = """
        SELECT count(*) AS chunks,
               count(DISTINCT source_id) AS sources,
               max(id) AS last_chunk_id
        FROM data_chunks
        WHERE variant = :variant
          AND source_id IN (SELECT id FROM data_sources WHERE active)
    """
    with engine.connect() as conn:
        row = conn.execute(text(query), {"variant": variant}).mappings().one()
    return {"variant": variant, **dict(row)}


# a run on the wrong variant otherwise looks like an ordinary run with different numbers
def corpus_variants() -> list[dict]:
    query = """
        SELECT variant, count(*) AS chunks, count(DISTINCT source_id) AS sources
        FROM data_chunks GROUP BY variant ORDER BY variant
    """
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(query)).mappings()]


def nearest_distance(embedding, *, variant) -> float | None:
    # same filters as hybrid_search: the topic axis must not see what retrieval cannot
    query = """
        SELECT embedding <=> CAST(:embedding AS vector) AS distance
        FROM data_chunks
        WHERE embedding IS NOT NULL
          AND variant = :variant
          AND source_id IN (SELECT id FROM data_sources WHERE active)
        ORDER BY distance
        LIMIT 1
    """
    with engine.connect() as conn:
        row = conn.execute(
            text(query), {"embedding": str(list(embedding)), "variant": variant}
        ).scalar()
    return float(row) if row is not None else None


def list_categories(category=None, only_top=None, *, variant):
    cat_filter = "AND category ~ (:category)::lquery" if category else ""
    cat_select = "subpath(category, 0, 1)::text" if only_top else "category"
    params = {"variant": variant}
    if category:
        params["category"] = f"*.{category}.*"
    query = f"""SELECT {cat_select} AS cat, COUNT(*) FROM data_chunks
                WHERE variant = :variant {cat_filter}
                GROUP BY cat ORDER BY {cat_select}"""

    with engine.connect() as conn:
        return conn.execute(text(query), params).fetchall()


def hybrid_search(
    question,
    embedding,
    category=None,
    limit_vector=config.settings.retrieval.limit_vector,
    limit_keyword=config.settings.retrieval.limit_keywords,
    limit=None,
    *,
    variant,
    distance_threshold=None,
):
    retrieval = config.settings.retrieval
    limit = limit or retrieval.results_limit
    if distance_threshold is None:
        distance_threshold = retrieval.distance_threshold
    rank_fn = retrieval.keyword_rank
    if rank_fn not in RANK_FUNCTIONS:
        raise ValueError(f"keyword_rank must be one of {sorted(RANK_FUNCTIONS)}")
    keyword_query = _keyword_query_sql(retrieval.keyword_query)
    cat_filter = "AND category ~ (:category)::lquery" if category else ""
    src_filter = """AND variant = :variant
                    AND source_id IN (SELECT id FROM data_sources WHERE active)"""
    query = f"""WITH vector_search AS (
                    SELECT id,
                           embedding <=> CAST(:embedding AS vector) AS distance,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:embedding AS vector) ASC) AS rank
                    FROM data_chunks WHERE embedding <=> CAST(:embedding AS vector) <= :distance_threshold {cat_filter} {src_filter}
                    ORDER BY distance
                    LIMIT :limit_vector
                ),
                keyword_search AS (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               ORDER BY {rank_fn}(content_tsv, q, :keyword_norm) DESC
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
                ORDER BY score DESC
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
        "keyword_norm": retrieval.keyword_norm,
    }
    if category:
        params["category"] = f"*.{category}.*"
    with engine.connect() as conn:
        return conn.execute(text(query), params).fetchall()
