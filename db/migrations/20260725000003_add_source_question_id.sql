-- migrate:up
ALTER TABLE questions ADD COLUMN source_question_id integer REFERENCES questions(id);

-- migrate:down
ALTER TABLE questions DROP COLUMN source_question_id;
