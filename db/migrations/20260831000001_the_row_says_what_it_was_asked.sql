-- migrate:up
-- asked came by join, so a row read a year later got today's question text
ALTER TABLE question_logs ADD COLUMN question_text text;
ALTER TABLE question_logs ADD COLUMN reference_answer text;

-- the backfill is the join it replaces, taken once: earlier rows carry what questions says now
UPDATE question_logs ql
SET question_text = q.original_text,
    reference_answer = q.reference_answer
FROM questions q
WHERE q.id = ql.question_id AND ql.question_text IS NULL;

-- migrate:down
ALTER TABLE question_logs DROP COLUMN question_text;
ALTER TABLE question_logs DROP COLUMN reference_answer;
