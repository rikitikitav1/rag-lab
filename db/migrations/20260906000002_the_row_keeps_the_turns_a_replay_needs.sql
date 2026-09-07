-- migrate:up
-- model turns and tool-call arguments per hop; the tool results live in `contexts` already
ALTER TABLE question_logs ADD COLUMN transcript jsonb;

-- migrate:down
ALTER TABLE question_logs DROP COLUMN transcript;
