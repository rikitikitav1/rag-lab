-- migrate:up
-- a vector index belongs to a variant, and a variant is a line in the config, not a
-- migration. Keeping baseline's index in the schema made db/schema.sql a function of
-- what happened to be indexed locally, since every other variant builds its own at
-- runtime. bootstrap rebuilds this one on the next start, like all the others.
DROP INDEX IF EXISTS data_chunks_embedding_baseline_idx;

-- migrate:down
CREATE INDEX IF NOT EXISTS data_chunks_embedding_baseline_idx ON data_chunks
  USING hnsw (embedding vector_cosine_ops) WHERE variant = 'baseline';
