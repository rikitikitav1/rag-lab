-- migrate:up
CREATE TABLE jobs (
  id serial PRIMARY KEY,
  type text NOT NULL,
  status text NOT NULL DEFAULT 'new' CHECK (status IN ('new', 'running', 'done', 'error', 'paused')),
  options jsonb NOT NULL DEFAULT '{}',
  error jsonb,
  apply_since timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX idx_jobs_apply_since_status ON jobs (apply_since, status);

-- migrate:down
DROP TABLE jobs;
