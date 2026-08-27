-- migrate:up
-- the legacy cut copies the file's H1 onto every section chunk. Naming that copy as a
-- prefix is what lets the body metrics run on baseline at all: without it the variant
-- abstains on every one of them and can never be reported dirty, while a hygienic
-- variant of the same source is scored over the full weight set. Content is not touched.
UPDATE data_chunks
SET prefix_len = position(E'\n' in content)
WHERE variant = 'baseline'
  AND prefix_len IS NULL
  AND section IS NOT NULL
  AND content LIKE '# %'
  AND position(E'\n' in content) > 3
  AND split_part(section, ' > ', 1) =
      substring(content from 3 for position(E'\n' in content) - 3);

-- every other legacy chunk carries no copied heading, so its body is the whole content.
-- Zero here is a measurement, not a default: a NULL would keep the row out of the
-- metrics and leave them speaking for the first piece of each section only.
UPDATE data_chunks
SET prefix_len = 0
WHERE variant = 'baseline' AND prefix_len IS NULL;

-- migrate:down
UPDATE data_chunks SET prefix_len = NULL WHERE variant = 'baseline';
