-- migrate:up
CREATE TABLE experiments (
  id serial PRIMARY KEY,
  name text,
  status text NOT NULL DEFAULT 'draft',
  dataset text NOT NULL,
  sample_size integer,
  sample_seed integer,
  question_ids jsonb,
  data_prep jsonb NOT NULL DEFAULT '{}',
  procedure jsonb NOT NULL DEFAULT '{}',
  param text NOT NULL,
  param_values jsonb NOT NULL DEFAULT '[]',
  run_names jsonb NOT NULL DEFAULT '[]',
  results jsonb,
  conclusion text,
  started_at timestamptz,
  finished_at timestamptz,
  elapsed double precision,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- migrate:down
DROP TABLE experiments;
