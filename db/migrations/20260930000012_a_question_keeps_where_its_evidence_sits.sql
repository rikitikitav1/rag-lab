-- migrate:up
ALTER TABLE questions ADD COLUMN evidence_at jsonb;

-- migrate:down
ALTER TABLE questions DROP COLUMN evidence_at;
