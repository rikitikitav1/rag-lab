-- migrate:up
-- the source as its door or the seed declared it, reader and rules included; the columns it repeats stay until the wipe
ALTER TABLE data_sources ADD COLUMN declaration jsonb;
-- a row the seed writes from a source file: the file speaks for it, so the knobs door refuses it
ALTER TABLE data_sources ADD COLUMN seeded boolean NOT NULL DEFAULT false;
ALTER TABLE data_sources ALTER COLUMN kind DROP NOT NULL;
-- every new row walks the line from declared; only the seed's old rows were born accepted
ALTER TABLE data_sources ALTER COLUMN stage SET DEFAULT 'declared';
UPDATE data_sources
SET declaration = origin || jsonb_strip_nulls(jsonb_build_object('name', name, 'language', language, 'licence', licence))
WHERE origin IS NOT NULL;

-- migrate:down
ALTER TABLE data_sources ALTER COLUMN stage SET DEFAULT 'accepted';
ALTER TABLE data_sources ALTER COLUMN kind SET NOT NULL;
ALTER TABLE data_sources DROP COLUMN seeded;
ALTER TABLE data_sources DROP COLUMN declaration;
