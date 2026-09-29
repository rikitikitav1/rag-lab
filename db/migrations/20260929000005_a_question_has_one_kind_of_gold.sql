-- migrate:up
-- a question's gold is an exact section or the older marks, never both: the reader would pick one silently
ALTER TABLE questions ADD CONSTRAINT questions_one_kind_of_gold CHECK (gold IS NULL OR cardinality(marked_sources) = 0);

-- migrate:down
ALTER TABLE questions DROP CONSTRAINT questions_one_kind_of_gold;
