import config
from langdetect import DetectorFactory, LangDetectException, detect
from orm.sync_db import engine
from sqlalchemy import text

DetectorFactory.seed = 0

TS_CONFIG = {"en": "english", "ru": "russian"}


def _ts_config(text_):
    try:
        return TS_CONFIG.get(detect(text_), "english")
    except LangDetectException:
        return "english"


def cleanup():
    with engine.begin() as conn:
        conn.execute(
            text("TRUNCATE data_chunks, data_sources RESTART IDENTITY CASCADE")
        )


def is_empty():
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT NOT EXISTS (SELECT 1 FROM data_chunks)")
        ).scalar()


def nearest_distance(embedding) -> float | None:
    query = """
        SELECT embedding <=> CAST(:embedding AS vector) AS distance
        FROM data_chunks
        WHERE embedding IS NOT NULL
        ORDER BY distance
        LIMIT 1
    """
    with engine.connect() as conn:
        row = conn.execute(text(query), {"embedding": str(list(embedding))}).scalar()
    return float(row) if row is not None else None


def list_categories(category=None, only_top=None):
    cat_filter = "WHERE category ~ (:category)::lquery" if category else ""
    cat_select = "subpath(category, 0, 1)::text" if only_top else "category"
    params = {"category": f"*.{category}.*"} if category else {}
    query = f"""SELECT {cat_select} AS cat, COUNT(*) FROM data_chunks {cat_filter} GROUP BY cat ORDER BY {cat_select}"""

    with engine.connect() as conn:
        return conn.execute(text(query), params).fetchall()


def hybrid_search(
    question,
    embedding,
    category=None,
    limit_vector=config.settings.retrieval.limit_vector,
    limit_keyword=config.settings.retrieval.limit_keywords,
    limit=None,
):
    limit = limit or config.settings.retrieval.results_limit
    cat_filter = "AND category ~ (:category)::lquery" if category else ""
    src_filter = "AND source_id IN (SELECT id FROM data_sources WHERE active)"
    query = f"""WITH vector_search AS (
                    SELECT id,
                           embedding <=> CAST(:embedding AS vector) AS distance,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:embedding AS vector) ASC) AS rank
                    FROM data_chunks WHERE embedding <=> CAST(:embedding AS vector) <= :distance_threshold {cat_filter} {src_filter}
                    ORDER BY distance
                    LIMIT :limit_vector
                ),
                keyword_search AS (
                    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(content_tsv, q) DESC) AS rank
                    FROM data_chunks, plainto_tsquery((:ts_config)::regconfig, :question) q
                    WHERE content_tsv @@ q {cat_filter} {src_filter}
                    ORDER BY rank
                    LIMIT :limit_keyword
                )
                SELECT d.content, d.source, d.category, d.chunk_index,
                       v.rank AS vector_rank, k.rank AS keyword_rank, v.distance AS distance,
                    COALESCE(1.0/(:rrf_k + v.rank), 0) + COALESCE(1.0/(:rrf_k + k.rank), 0) AS score
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
        "distance_threshold": config.settings.retrieval.distance_threshold,
        "rrf_k": config.settings.retrieval.rrf_k,
        "ts_config": _ts_config(question),
    }
    if category:
        params["category"] = f"*.{category}.*"
    with engine.connect() as conn:
        return conn.execute(text(query), params).fetchall()
