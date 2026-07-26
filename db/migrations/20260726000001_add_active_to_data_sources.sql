-- migrate:up
ALTER TABLE data_sources ADD COLUMN active boolean NOT NULL DEFAULT true;

-- migrate:down
ALTER TABLE data_sources DROP COLUMN active;
