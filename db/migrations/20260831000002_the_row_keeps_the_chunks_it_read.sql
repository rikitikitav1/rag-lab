-- migrate:up
-- the join into `context` cannot be undone: a separator may itself occur inside a chunk
ALTER TABLE question_logs ADD COLUMN contexts jsonb;

-- migrate:down
ALTER TABLE question_logs DROP COLUMN contexts;
