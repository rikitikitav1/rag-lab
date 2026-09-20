-- migrate:up
-- "did all the arms run on one code" was a human comparing docker inspect with file mtimes
ALTER TABLE jobs ADD COLUMN code jsonb;

-- migrate:down
ALTER TABLE jobs DROP COLUMN code;
