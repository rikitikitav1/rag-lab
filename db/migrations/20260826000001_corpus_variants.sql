-- migrate:up
ALTER TABLE data_chunks ADD COLUMN variant text NOT NULL DEFAULT 'baseline';
ALTER TABLE data_chunks ALTER COLUMN variant DROP DEFAULT;

ALTER TABLE data_chunks ADD COLUMN section text;
ALTER TABLE data_chunks ADD COLUMN content_hash text;

CREATE INDEX data_chunks_variant_source_idx ON data_chunks (variant, source_id);
CREATE INDEX data_chunks_content_hash_idx ON data_chunks (content_hash);

CREATE INDEX data_chunks_embedding_baseline_idx ON data_chunks
  USING hnsw (embedding vector_cosine_ops) WHERE variant = 'baseline';

-- migrate:down
DROP INDEX data_chunks_embedding_baseline_idx;
DROP INDEX data_chunks_content_hash_idx;
DROP INDEX data_chunks_variant_source_idx;
ALTER TABLE data_chunks DROP COLUMN content_hash;
ALTER TABLE data_chunks DROP COLUMN section;
ALTER TABLE data_chunks DROP COLUMN variant;
