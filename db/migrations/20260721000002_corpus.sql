-- migrate:up
CREATE TABLE data_sources (
  id serial PRIMARY KEY,
  name varchar(256) NOT NULL UNIQUE,
  kind varchar(32) NOT NULL,
  git_url text,
  path text
);

CREATE TABLE data_chunks (
  id serial PRIMARY KEY,
  source_id integer NOT NULL REFERENCES data_sources (id) ON DELETE CASCADE,
  source text NOT NULL,
  content text NOT NULL,
  embedding vector(1024),
  chunk_index integer NOT NULL,
  category ltree NOT NULL,
  language text NOT NULL,
  content_tsv tsvector
);
CREATE INDEX ON data_chunks (source_id);
CREATE INDEX ON data_chunks USING gin (content_tsv);
CREATE INDEX ON data_chunks USING gist (category);
CREATE INDEX ON data_chunks USING hnsw (embedding vector_cosine_ops);

CREATE FUNCTION data_chunks_content_tsv () RETURNS trigger AS $$
BEGIN
  NEW.content_tsv := to_tsvector(
    CASE NEW.language
      WHEN 'rus' THEN 'russian'
      WHEN 'eng' THEN 'english'
      ELSE 'simple'
    END::regconfig,
    NEW.content
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER data_chunks_content_tsv_trg
  BEFORE INSERT OR UPDATE ON data_chunks
  FOR EACH ROW EXECUTE FUNCTION data_chunks_content_tsv ();

-- migrate:down
DROP TABLE data_chunks;
DROP FUNCTION data_chunks_content_tsv;
DROP TABLE data_sources;
