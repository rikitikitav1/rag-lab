-- migrate:up
-- what a finished job has to say beyond done, as a full reindex's refused sources
ALTER TABLE jobs ADD COLUMN result jsonb;

-- migrate:down
ALTER TABLE jobs DROP COLUMN result;
