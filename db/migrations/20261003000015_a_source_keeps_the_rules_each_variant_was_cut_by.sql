-- migrate:up
-- {variant: the cut's rules}: beside the digest in indexed_with, so a moved digest names the fields that moved
ALTER TABLE data_sources ADD COLUMN indexed_rules jsonb NOT NULL DEFAULT '{}'::jsonb;

-- migrate:down
ALTER TABLE data_sources DROP COLUMN indexed_rules;
