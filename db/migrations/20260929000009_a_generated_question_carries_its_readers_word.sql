-- migrate:up
-- the reader's word on a generated question; a set made by hand was never asked
ALTER TABLE questions ADD COLUMN answerable_by_8b boolean;

-- migrate:down
ALTER TABLE questions DROP COLUMN answerable_by_8b;
