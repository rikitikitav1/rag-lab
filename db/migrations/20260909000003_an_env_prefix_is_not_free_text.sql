-- migrate:up
-- interpolated into an environment variable name, so a shape the door enforces the database keeps
ALTER TABLE engines ADD CONSTRAINT engines_env_prefix_shape CHECK (env_prefix ~ '^[A-Z][A-Z0-9_]{0,31}$');

-- migrate:down
ALTER TABLE engines DROP CONSTRAINT engines_env_prefix_shape;
