-- migrate:up
DROP INDEX IF EXISTS tmp_partial_probe;
-- a query that forgets the variant must be slow, not quietly wrong
DROP INDEX IF EXISTS data_chunks_embedding_idx;
REINDEX INDEX data_chunks_embedding_baseline_idx;

-- migrate:down
CREATE INDEX data_chunks_embedding_idx ON data_chunks USING hnsw (embedding vector_cosine_ops);
