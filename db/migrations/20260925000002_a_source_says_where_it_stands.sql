-- migrate:up
-- a source added by hand is declared, converted to a raw folder, then accepted; the corpus already indexed is accepted
ALTER TABLE data_sources
    ADD COLUMN stage text NOT NULL DEFAULT 'accepted',
    ADD COLUMN language text,
    ADD COLUMN licence text,
    ADD COLUMN origin jsonb,
    ADD COLUMN raw jsonb NOT NULL DEFAULT '{}'::jsonb;

-- migrate:down
ALTER TABLE data_sources
    DROP COLUMN raw,
    DROP COLUMN origin,
    DROP COLUMN licence,
    DROP COLUMN language,
    DROP COLUMN stage;
