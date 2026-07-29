-- migrate:up
CREATE TABLE mcp_integrations (
  id serial PRIMARY KEY,
  name varchar(64) NOT NULL UNIQUE,
  url text NOT NULL,
  status text NOT NULL DEFAULT 'disabled',
  allowed_tools jsonb NOT NULL DEFAULT '[]',
  tool_schemas jsonb NOT NULL DEFAULT '{}',
  auth jsonb,
  timeout_s integer NOT NULL DEFAULT 30,
  max_result_chars integer NOT NULL DEFAULT 4000,
  last_checked_at timestamptz,
  last_error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- migrate:down
DROP TABLE mcp_integrations;
