-- migrate:up
ALTER TABLE jobs ADD COLUMN queue text NOT NULL DEFAULT 'default';
DROP INDEX idx_jobs_apply_since_status;
CREATE INDEX idx_jobs_queue_status_apply_since ON jobs (queue, status, apply_since);

-- migrate:down
DROP INDEX idx_jobs_queue_status_apply_since;
CREATE INDEX idx_jobs_apply_since_status ON jobs (apply_since, status);
ALTER TABLE jobs DROP COLUMN queue;
