import config
from orm.sync_db import engine
from sqlalchemy import text

import db

VARIANT = config.settings.corpus.variant
ACTIVE = db.live_rows("dc")

FAMILY = """
    CASE WHEN ds.name LIKE '%-interview-questions' THEN 'interview'
         ELSE ds.name END
"""

by_family = f"""
SELECT {FAMILY} AS family, count(*) AS chunks,
       count(*) FILTER (WHERE dc.content !~ '^\\s*#') AS orphans,
       count(*) FILTER (WHERE dc.content ~ '^\\s*```') AS starts_with_code,
       count(*) FILTER (WHERE length(dc.content) >= 1000) AS at_the_cap,
       round(avg(length(dc.content))) AS avg_len
FROM data_chunks dc JOIN data_sources ds ON ds.id = dc.source_id
WHERE {ACTIVE} GROUP BY 1 ORDER BY 2 DESC
"""

# how a single question is spread across chunks, interview repos only
spread = f"""
SELECT n_chunks, count(*) AS questions FROM (
    SELECT source, question_no, count(*) AS n_chunks FROM (
        SELECT dc.source, dc.chunk_index,
               count(*) FILTER (WHERE dc.content ~ '^# .*\\n## ')
                   OVER (PARTITION BY dc.source ORDER BY dc.chunk_index) AS question_no
        FROM data_chunks dc JOIN data_sources ds ON ds.id = dc.source_id
        WHERE {ACTIVE} AND ds.name LIKE '%-interview-questions'
    ) marked WHERE question_no > 0
    GROUP BY source, question_no
) g GROUP BY 1 ORDER BY 1
"""

headings = f"""
SELECT count(*) FILTER (WHERE content ~ '^# .*\\n## ') AS question_heads,
       count(*) AS chunks
FROM data_chunks dc JOIN data_sources ds ON ds.id = dc.source_id
WHERE {ACTIVE} AND ds.name LIKE '%-interview-questions'
"""

with engine.connect() as conn:
    print("### by source family")
    for r in conn.execute(text(by_family), {"variant": VARIANT}).mappings():
        print("   " + "  ".join(f"{k}={v}" for k, v in r.items()))
    print()
    print("### interview repos: chunks that carry the question heading")
    for r in conn.execute(text(headings), {"variant": VARIANT}).mappings():
        print("   " + "  ".join(f"{k}={v}" for k, v in r.items()))
    print()
    print("### interview repos: how many chunks one question is spread over")
    for r in conn.execute(text(spread), {"variant": VARIANT}).mappings():
        print("   " + "  ".join(f"{k}={v}" for k, v in r.items()))
