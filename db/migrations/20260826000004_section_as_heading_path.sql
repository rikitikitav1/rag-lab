-- migrate:up
-- one format for every variant: heading path joined by " > ", no markers, numbers kept
WITH marked AS (
  SELECT id, source, chunk_index,
         count(*) FILTER (WHERE content ~ '^# .*\n## ')
           OVER (PARTITION BY source ORDER BY chunk_index) AS question_no
  FROM data_chunks WHERE variant = 'baseline'
),
heads AS (
  SELECT m.source, m.question_no,
         regexp_replace(split_part(dc.content, E'\n', 1), '^#+\s*', '') || ' > ' ||
         regexp_replace(split_part(dc.content, E'\n', 2), '^#+\s*', '') AS path
  FROM marked m
  JOIN data_chunks dc ON dc.id = m.id
  WHERE dc.content ~ '^# .*\n## '
)
UPDATE data_chunks d
SET section = h.path
FROM marked m
JOIN heads h ON h.source = m.source AND h.question_no = m.question_no
WHERE d.id = m.id AND m.question_no > 0;

-- migrate:down
UPDATE data_chunks SET section = split_part(section, ' > ', 2) WHERE variant = 'baseline';
