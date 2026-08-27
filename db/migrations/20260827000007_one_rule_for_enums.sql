-- migrate:up
-- 20260728000001 dropped the same kind of check off jobs.status and stated the rule:
-- an enum is validated in the model, so a new value needs no migration. 20260827000006
-- added one to experiments.kind two hours ago against that rule, and neither mentioned
-- the other. One rule, and it is the older one: experiments.status never had a check
-- either, and it is the column with a state machine reading it.
ALTER TABLE experiments DROP CONSTRAINT IF EXISTS experiments_kind_check;

-- migrate:down
ALTER TABLE experiments
  ADD CONSTRAINT experiments_kind_check CHECK (kind IN ('generation', 'retrieval'));
