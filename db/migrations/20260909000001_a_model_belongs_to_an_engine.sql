-- migrate:up
-- address and key stay in the environment: a row you can read a key out of leaks through any report
CREATE TABLE engines (
    id serial PRIMARY KEY,
    name text NOT NULL UNIQUE,
    kind text NOT NULL,
    env_prefix text NOT NULL,
    placement text NOT NULL,
    budget numeric(12, 6),
    spent numeric(12, 6) NOT NULL DEFAULT 0,
    reserved numeric(12, 6) NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- null is not "the same weights": a comparison must read it as unknown and refuse, never as a match
ALTER TABLE models ADD COLUMN engine_id integer REFERENCES engines (id) ON DELETE RESTRICT;
ALTER TABLE models ADD COLUMN weights text;
ALTER TABLE models ADD COLUMN quant text;

INSERT INTO engines (name, kind, env_prefix, placement)
VALUES ('ollama', 'ollama', 'OLLAMA', 'gpu');

UPDATE models SET engine_id = (SELECT id FROM engines WHERE name = 'ollama');
ALTER TABLE models ALTER COLUMN engine_id SET NOT NULL;

-- one name on two engines is two rows: `qwen2.5:7b` under ollama and the same weights under vllm
ALTER TABLE models DROP CONSTRAINT models_name_key;
ALTER TABLE models ADD CONSTRAINT models_engine_name_key UNIQUE (engine_id, name);

-- migrate:down
-- one-way in practice: `UNIQUE (name)` cannot come back once two engines hold one name
ALTER TABLE models DROP CONSTRAINT models_engine_name_key;
ALTER TABLE models ADD CONSTRAINT models_name_key UNIQUE (name);
ALTER TABLE models DROP COLUMN quant;
ALTER TABLE models DROP COLUMN weights;
ALTER TABLE models DROP COLUMN engine_id;
DROP TABLE engines;
