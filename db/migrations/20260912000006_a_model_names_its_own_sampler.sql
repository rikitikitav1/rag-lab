-- migrate:up
-- a verbose or reasoning model asks a larger budget in any role; laid over the role's own options
ALTER TABLE models ADD COLUMN options jsonb DEFAULT '{}'::jsonb NOT NULL;

-- migrate:down
ALTER TABLE models DROP COLUMN options;
