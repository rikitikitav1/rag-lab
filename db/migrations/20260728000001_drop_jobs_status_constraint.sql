-- migrate:up
-- status is validated in the model (Enum(JobStatus, native_enum=False)); drop the DB check so new statuses need no migration
ALTER TABLE jobs DROP CONSTRAINT jobs_status_check;

-- migrate:down
UPDATE jobs SET status = 'error' WHERE status NOT IN ('new', 'running', 'done', 'error', 'paused');
ALTER TABLE jobs ADD CONSTRAINT jobs_status_check CHECK (status = ANY (ARRAY['new', 'running', 'done', 'error', 'paused']));
