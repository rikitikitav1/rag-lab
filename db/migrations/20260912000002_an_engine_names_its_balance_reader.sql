-- migrate:up
-- each broker shapes the route that tells what is left on the key its own way, so the row names a reader
ALTER TABLE engines ADD COLUMN balance_reader text NOT NULL DEFAULT 'none';

-- migrate:down
ALTER TABLE engines DROP COLUMN balance_reader;
