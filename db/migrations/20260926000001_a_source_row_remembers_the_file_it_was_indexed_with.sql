-- migrate:up
-- the file wins over the row: the row keeps, per variant, the digest of the source file its chunks were cut by
ALTER TABLE data_sources ADD COLUMN indexed_with jsonb NOT NULL DEFAULT '{}'::jsonb;

-- migrate:down
ALTER TABLE data_sources DROP COLUMN indexed_with;
