-- migrate:up
-- one rule, the older one of 20260728000001: an enum is validated in the model, not here
ALTER TABLE experiments DROP CONSTRAINT IF EXISTS experiments_kind_check;

-- migrate:down
ALTER TABLE experiments
  ADD CONSTRAINT experiments_kind_check CHECK (kind IN ('generation', 'retrieval'));
