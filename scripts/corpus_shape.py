import config
from orm.sync_db import engine
from sqlalchemy import text
from use_cases.index import check_variant

import db

VARIANT = check_variant(config.settings.corpus.variant)
ACTIVE = db.live_rows("dc")

QUERIES = {
    "totals": f"""
        SELECT count(*) AS chunks, count(DISTINCT dc.source) AS files,
               count(DISTINCT dc.source_id) AS sources
        FROM data_chunks dc WHERE {ACTIVE}
    """,
    "length distribution": f"""
        SELECT min(length(content)) AS min,
               percentile_disc(0.25) WITHIN GROUP (ORDER BY length(content)) AS p25,
               percentile_disc(0.50) WITHIN GROUP (ORDER BY length(content)) AS median,
               percentile_disc(0.75) WITHIN GROUP (ORDER BY length(content)) AS p75,
               percentile_disc(0.95) WITHIN GROUP (ORDER BY length(content)) AS p95,
               max(length(content)) AS max, round(avg(length(content))) AS avg
        FROM data_chunks dc WHERE {ACTIVE}
    """,
    "short chunks": f"""
        SELECT count(*) FILTER (WHERE length(content) < 100) AS under_100,
               count(*) FILTER (WHERE length(content) < 200) AS under_200,
               count(*) FILTER (WHERE trim(content) = '') AS empty
        FROM data_chunks dc WHERE {ACTIVE}
    """,
    "exact duplicates by content": f"""
        SELECT count(*) AS duplicate_groups, sum(n) AS chunks_in_groups, sum(n - 1) AS redundant
        FROM (
            SELECT count(*) AS n FROM data_chunks dc WHERE {ACTIVE}
            GROUP BY md5(content) HAVING count(*) > 1
        ) g
    """,
    "duplicates spanning several sources": f"""
        SELECT count(*) AS groups, sum(srcs) AS source_slots
        FROM (
            SELECT count(DISTINCT dc.source) AS srcs FROM data_chunks dc WHERE {ACTIVE}
            GROUP BY md5(content) HAVING count(DISTINCT dc.source) > 1
        ) g
    """,
    "orphans: chunk does not start with a heading": f"""
        SELECT count(*) FILTER (WHERE content !~ '^\\s*#') AS no_heading,
               count(*) FILTER (WHERE content ~ '^\\s*```') AS starts_with_code,
               count(*) AS total
        FROM data_chunks dc WHERE {ACTIVE}
    """,
    "code weight": f"""
        SELECT count(*) FILTER (WHERE content LIKE '%```%') AS has_fence,
               count(*) FILTER (WHERE content !~ '[a-zA-Zа-яА-Я]{{40,}}' ) AS no_long_prose
        FROM data_chunks dc WHERE {ACTIVE}
    """,
    "top files by chunk count": f"""
        SELECT dc.source, count(*) AS chunks, round(avg(length(content))) AS avg_len
        FROM data_chunks dc WHERE {ACTIVE}
        GROUP BY dc.source ORDER BY count(*) DESC LIMIT 10
    """,
    "chunks per registered source": f"""
        SELECT ds.name, count(*) AS chunks, count(DISTINCT dc.source) AS files,
               round(avg(length(dc.content))) AS avg_len
        FROM data_chunks dc JOIN data_sources ds ON ds.id = dc.source_id
        WHERE {ACTIVE} GROUP BY ds.name ORDER BY count(*) DESC
    """,
    "devinterview boilerplate (chunk 0 with the badge)": f"""
        SELECT count(*) AS chunks
        FROM data_chunks dc
        WHERE {ACTIVE} AND dc.chunk_index = 0 AND content LIKE '%You can also find all%'
    """,
}

with engine.connect() as conn:
    for title, sql in QUERIES.items():
        print(f"### {title}")
        rows = conn.execute(text(sql), {"variant": VARIANT}).mappings().all()
        for r in rows:
            print("   " + "  ".join(f"{k}={v}" for k, v in r.items()))
        print()
