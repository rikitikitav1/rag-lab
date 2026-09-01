-- migrate:up
-- naming the copied H1 as a prefix is what lets the body metrics run on baseline at all
UPDATE data_chunks
SET prefix_len = position(E'\n' in content)
WHERE variant = 'baseline'
  AND prefix_len IS NULL
  AND section IS NOT NULL
  AND content LIKE '# %'
  AND position(E'\n' in content) > 3
  AND split_part(section, ' > ', 1) =
      substring(content from 3 for position(E'\n' in content) - 3);

-- zero is a measurement, not a default: a NULL would keep the row out of the body metrics
UPDATE data_chunks
SET prefix_len = 0
WHERE variant = 'baseline' AND prefix_len IS NULL;

-- migrate:down
UPDATE data_chunks SET prefix_len = NULL WHERE variant = 'baseline';
