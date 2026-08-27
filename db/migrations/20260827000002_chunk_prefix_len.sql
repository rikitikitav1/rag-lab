-- migrate:up
-- the heading prefix is repeated on every chunk and its length differs per chunk;
-- baseline stays NULL because its prefix cannot be recovered without recutting
ALTER TABLE data_chunks ADD COLUMN prefix_len integer;

-- migrate:down
ALTER TABLE data_chunks DROP COLUMN prefix_len;
