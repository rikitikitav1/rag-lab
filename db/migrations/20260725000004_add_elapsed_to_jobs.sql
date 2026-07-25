-- migrate:up
ALTER TABLE jobs ADD COLUMN elapsed double precision;

-- migrate:down
ALTER TABLE jobs DROP COLUMN elapsed;
