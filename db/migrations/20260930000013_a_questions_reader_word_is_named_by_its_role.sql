-- migrate:up
-- the reader is a role, and the model seated on it has changed; the column names the role, not one model
ALTER TABLE questions RENAME COLUMN answerable_by_8b TO answerable_by_reader;

-- migrate:down
ALTER TABLE questions RENAME COLUMN answerable_by_reader TO answerable_by_8b;
