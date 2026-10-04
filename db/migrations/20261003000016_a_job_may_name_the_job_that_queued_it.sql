-- migrate:up
-- the job whose handler queued this one, when one did; null for a job queued from outside
ALTER TABLE jobs ADD COLUMN parent_id integer REFERENCES jobs(id) ON DELETE SET NULL;
CREATE INDEX jobs_parent_id_idx ON jobs (parent_id) WHERE parent_id IS NOT NULL;

-- migrate:down
DROP INDEX jobs_parent_id_idx;
ALTER TABLE jobs DROP COLUMN parent_id;
