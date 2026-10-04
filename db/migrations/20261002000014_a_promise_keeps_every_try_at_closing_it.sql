-- migrate:up
-- [{at, runs, measurements, cleared, cleared_because} or {at, runs, refused}]: each close asked while the promise was open
ALTER TABLE preregistrations ADD COLUMN attempts jsonb NOT NULL DEFAULT '[]'::jsonb;

-- migrate:down
ALTER TABLE preregistrations DROP COLUMN attempts;
