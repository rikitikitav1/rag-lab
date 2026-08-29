-- migrate:up
-- the verdict is about our cut of the source, never about the source itself:
-- the redis documentation is the cleanest material in the corpus and the worst cut
ALTER TABLE data_sources ADD COLUMN ingest_quality text;
ALTER TABLE data_sources ADD COLUMN ingest_variant text;
ALTER TABLE data_sources ADD COLUMN ingest_checked_at timestamptz;
ALTER TABLE data_sources ADD COLUMN ingest_reports jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE data_sources ADD CONSTRAINT data_sources_ingest_quality_check
  CHECK (ingest_quality IS NULL OR ingest_quality IN ('ok', 'dirty', 'broken'));

-- migrate:down
ALTER TABLE data_sources DROP CONSTRAINT data_sources_ingest_quality_check;
ALTER TABLE data_sources DROP COLUMN ingest_reports;
ALTER TABLE data_sources DROP COLUMN ingest_checked_at;
ALTER TABLE data_sources DROP COLUMN ingest_variant;
ALTER TABLE data_sources DROP COLUMN ingest_quality;
