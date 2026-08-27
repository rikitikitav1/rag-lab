-- migrate:up
-- the model declares an enum and the column was plain text, so a typo would have been a
-- valid kind in the database and an invalid one in the code
ALTER TABLE experiments
  ADD CONSTRAINT experiments_kind_check CHECK (kind IN ('generation', 'retrieval'));

-- migrate:down
ALTER TABLE experiments DROP CONSTRAINT experiments_kind_check;
