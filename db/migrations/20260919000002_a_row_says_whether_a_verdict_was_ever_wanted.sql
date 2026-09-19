-- migrate:up
-- the sweep judges every unjudged row there is, and a run may have meant to have no judge at all
ALTER TABLE question_logs ADD COLUMN judge_wanted boolean DEFAULT true NOT NULL;

-- migrate:down
ALTER TABLE question_logs DROP COLUMN judge_wanted;
