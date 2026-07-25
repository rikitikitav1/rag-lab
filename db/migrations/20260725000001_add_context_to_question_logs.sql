-- migrate:up
ALTER TABLE question_logs ADD COLUMN context text;

-- migrate:down
ALTER TABLE question_logs DROP COLUMN context;
