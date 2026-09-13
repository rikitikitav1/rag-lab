-- migrate:up
-- a quota is spent by calls, and every call that spends it in the worker belongs to a job
ALTER TABLE jobs ADD COLUMN tokens jsonb;

-- migrate:down
ALTER TABLE jobs DROP COLUMN tokens;
