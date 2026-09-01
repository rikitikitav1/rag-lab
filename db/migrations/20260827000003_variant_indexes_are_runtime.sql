-- migrate:up
-- a variant builds its index at runtime, and keeping this one made schema.sql local state
DROP INDEX IF EXISTS data_chunks_embedding_baseline_idx;

-- migrate:down
CREATE INDEX IF NOT EXISTS data_chunks_embedding_baseline_idx ON data_chunks
  USING hnsw (embedding vector_cosine_ops) WHERE variant = 'baseline';
