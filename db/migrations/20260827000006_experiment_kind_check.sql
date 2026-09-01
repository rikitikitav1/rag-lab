-- migrate:up
-- the column was plain text while the model declares an enum, so a typo was valid in one
ALTER TABLE experiments
  ADD CONSTRAINT experiments_kind_check CHECK (kind IN ('generation', 'retrieval'));

-- migrate:down
ALTER TABLE experiments DROP CONSTRAINT experiments_kind_check;
