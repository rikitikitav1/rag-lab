-- migrate:up
UPDATE data_chunks SET content_hash = md5(content) WHERE variant = 'baseline';

-- the heading lives only in the first chunk of a question; carry it to its siblings
WITH marked AS (
  SELECT id, source, chunk_index,
         count(*) FILTER (WHERE content ~ '^# .*\n## ')
           OVER (PARTITION BY source ORDER BY chunk_index) AS question_no
  FROM data_chunks WHERE variant = 'baseline'
),
heads AS (
  SELECT m.source, m.question_no, split_part(dc.content, E'\n', 2) AS heading
  FROM marked m
  JOIN data_chunks dc ON dc.id = m.id
  WHERE dc.content ~ '^# .*\n## '
)
UPDATE data_chunks d
SET section = h.heading
FROM marked m
JOIN heads h ON h.source = m.source AND h.question_no = m.question_no
WHERE d.id = m.id AND m.question_no > 0;

-- migrate:down
UPDATE data_chunks SET section = NULL, content_hash = NULL WHERE variant = 'baseline';
