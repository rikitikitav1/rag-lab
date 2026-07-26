-- migrate:up
ALTER TABLE question_logs ADD COLUMN pipeline text NOT NULL DEFAULT 'single_shot';
CREATE INDEX idx_question_logs_pipeline ON question_logs (pipeline);

-- migrate:down
DROP INDEX idx_question_logs_pipeline;
ALTER TABLE question_logs DROP COLUMN pipeline;
