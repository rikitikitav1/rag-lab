-- migrate:up
-- a question's gold as a file, its section path and version; an older question keeps its marks and has none
ALTER TABLE questions ADD COLUMN gold jsonb;

-- migrate:down
ALTER TABLE questions DROP COLUMN gold;
