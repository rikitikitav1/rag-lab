-- migrate:up
CREATE TABLE models (
  id serial PRIMARY KEY,
  name text NOT NULL UNIQUE,
  status text NOT NULL DEFAULT 'available' CHECK (status IN ('available', 'loading', 'ready'))
);

CREATE TABLE model_roles (
  role text PRIMARY KEY CHECK (role IN ('generation', 'embedding', 'judging')),
  model_id integer NOT NULL REFERENCES models (id) ON DELETE RESTRICT
);

CREATE TABLE prompts (
  id serial PRIMARY KEY,
  purpose text NOT NULL CHECK (purpose IN ('generate.answer', 'judge.faithfulness', 'judge.relevance')),
  version integer NOT NULL,
  template text NOT NULL,
  active boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (purpose, version)
);
CREATE UNIQUE INDEX one_active_prompt_per_purpose ON prompts (purpose) WHERE active;

-- migrate:down
DROP TABLE prompts;
DROP TABLE model_roles;
DROP TABLE models;
