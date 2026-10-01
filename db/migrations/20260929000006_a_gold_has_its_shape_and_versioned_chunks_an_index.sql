-- migrate:up
-- an exact gold names its file and section as text, or every reader of it would fail on one bad row
ALTER TABLE questions ADD CONSTRAINT questions_gold_shape CHECK (
    gold IS NULL OR (jsonb_typeof(gold->'file') = 'string' AND jsonb_typeof(gold->'section') = 'string')
);
-- the search asks on every call whether an older version is in search; the versioned rows alone answer it
CREATE INDEX data_chunks_versioned_idx ON data_chunks (variant, category) WHERE cardinality(versions) > 0;

-- migrate:down
DROP INDEX data_chunks_versioned_idx;
ALTER TABLE questions DROP CONSTRAINT questions_gold_shape;
