-- migrate:up
-- the released versions a chunk's text stands for; empty is a source of one rolling version
ALTER TABLE data_chunks ADD COLUMN versions text[] NOT NULL DEFAULT '{}';
CREATE INDEX data_chunks_versions_idx ON data_chunks USING gin (versions);

-- migrate:down
DROP INDEX data_chunks_versions_idx;
ALTER TABLE data_chunks DROP COLUMN versions;
