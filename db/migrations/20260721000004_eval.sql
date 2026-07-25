-- migrate:up
CREATE TABLE questions (
  id serial PRIMARY KEY,
  text_hash varchar(64) NOT NULL UNIQUE,
  original_text text NOT NULL,
  normalized_text text,
  reference_answer text,
  marked_sources text[] NOT NULL DEFAULT '{}',
  set_name text,
  language text,
  kind text,
  status text,
  embedding vector(1024)
);

CREATE TABLE question_logs (
  id serial PRIMARY KEY,
  run_name text,
  question_id integer REFERENCES questions (id),
  answered boolean NOT NULL,
  answer text,
  sources jsonb,
  models jsonb NOT NULL DEFAULT '{}',
  prompts jsonb NOT NULL DEFAULT '{}',
  prompt_tokens integer,
  completion_tokens integer,
  elapsed double precision,
  faithfulness text,
  relevance text,
  metrics jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON question_logs (run_name);

-- migrate:down
DROP TABLE question_logs;
DROP TABLE questions;
