-- migrate:up
-- a comparison is an experiment of another kind, not a second entity: same dataset, same
-- fixed question ids, same state machine. What differs is the unit of a run, so the kind
-- says which. Defaults rather than a weakened constraint: existing rows read as they were
-- and the migration needs no backfill.
ALTER TABLE experiments
  ADD COLUMN kind text NOT NULL DEFAULT 'generation',
  ADD COLUMN axes jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE experiments ALTER COLUMN param SET DEFAULT '';

-- migrate:down
ALTER TABLE experiments ALTER COLUMN param DROP DEFAULT;
ALTER TABLE experiments DROP COLUMN axes, DROP COLUMN kind;
