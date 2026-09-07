-- migrate:up
-- `contexts` keeps the text of each chunk; this keeps its address, element for element
ALTER TABLE question_logs ADD COLUMN chunks jsonb;

-- migrate:down
ALTER TABLE question_logs DROP COLUMN chunks;
