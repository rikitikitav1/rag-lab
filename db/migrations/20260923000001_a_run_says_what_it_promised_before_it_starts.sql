-- migrate:up
-- a preregistration written in prose has two readings; this one is a row the stand can refuse against
CREATE TABLE preregistrations (
    id          bigserial PRIMARY KEY,
    name        text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    -- question sets and ids the closing number lives on, declared before any row exists
    population  jsonb NOT NULL DEFAULT '{}'::jsonb,
    -- which arm is the control and which is the treatment, by orchestrator or run name
    arms        jsonb NOT NULL DEFAULT '{}'::jsonb,
    -- {columns: [...], direction, paired, draws, floor}: every name checked against the registry
    closing     jsonb NOT NULL DEFAULT '{}'::jsonb,
    -- [{column, margin, direction}], read as non-inferiority
    guards      jsonb NOT NULL DEFAULT '[]'::jsonb,
    -- floor, veto, stop rules, price and the unflattering expectation, kept verbatim
    declared    jsonb NOT NULL DEFAULT '{}'::jsonb,
    -- the run names that closed it, written when it closes and never before
    closed_with jsonb
);

-- a closing run names the promise it was made under, and the door refuses one that names none
ALTER TABLE jobs ADD COLUMN prereg text;

-- migrate:down
ALTER TABLE jobs DROP COLUMN prereg;
DROP TABLE preregistrations;
