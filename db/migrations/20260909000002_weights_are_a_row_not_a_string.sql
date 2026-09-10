-- migrate:up
-- the closing measurement joins on this, and a join key spelled by hand drifts without saying so
CREATE TABLE weights (
    id serial PRIMARY KEY,
    name text NOT NULL UNIQUE,
    params text,
    created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE models ADD COLUMN weights_id integer REFERENCES weights (id) ON DELETE RESTRICT;
-- the artifact, not the weights: one base model is 4.36 GiB in Q4_K_M and 5.19 in AWQ
ALTER TABLE models ADD COLUMN size_bytes bigint;
ALTER TABLE models DROP COLUMN weights;

-- migrate:down
ALTER TABLE models ADD COLUMN weights text;
ALTER TABLE models DROP COLUMN size_bytes;
ALTER TABLE models DROP COLUMN weights_id;
DROP TABLE weights;
