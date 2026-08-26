import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import config  # noqa: E402
from orm.sync_db import engine  # noqa: E402
from sqlalchemy import text  # noqa: E402

VARIANT = config.settings.corpus.variant

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
WHERE ds.active AND dc.variant = :variant GROUP BY 1 ORDER BY 2 DESC
"""

# how a single question is spread across chunks, interview repos only
spread = """
SELECT n_chunks, count(*) AS questions FROM (
    SELECT source, question_no, count(*) AS n_chunks FROM (
        SELECT dc.source, dc.chunk_index,
               count(*) FILTER (WHERE dc.content ~ '^# .*\\n## ')
                   OVER (PARTITION BY dc.source ORDER BY dc.chunk_index) AS question_no
        FROM data_chunks dc JOIN data_sources ds ON ds.id = dc.source_id
        WHERE ds.active AND dc.variant = :variant AND ds.name LIKE '%-interview-questions'
    ) marked WHERE question_no > 0
    GROUP BY source, question_no
) g GROUP BY 1 ORDER BY 1
"""

headings = """
SELECT count(*) FILTER (WHERE content ~ '^# .*\\n## ') AS question_heads,
       count(*) AS chunks
FROM data_chunks dc JOIN data_sources ds ON ds.id = dc.source_id
WHERE ds.active AND dc.variant = :variant AND ds.name LIKE '%-interview-questions'
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
