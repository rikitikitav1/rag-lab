-- migrate:up
-- defaults, not a weakened constraint: existing rows read as they were, with no backfill
ALTER TABLE experiments
  ADD COLUMN kind text NOT NULL DEFAULT 'generation',
  ADD COLUMN axes jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE experiments ALTER COLUMN param SET DEFAULT '';

-- migrate:down
ALTER TABLE experiments ALTER COLUMN param DROP DEFAULT;
ALTER TABLE experiments DROP COLUMN axes, DROP COLUMN kind;
