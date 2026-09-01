-- migrate:up
-- nullable on purpose: baseline's prefix cannot be recovered without recutting
ALTER TABLE data_chunks ADD COLUMN prefix_len integer;

-- migrate:down
ALTER TABLE data_chunks DROP COLUMN prefix_len;
