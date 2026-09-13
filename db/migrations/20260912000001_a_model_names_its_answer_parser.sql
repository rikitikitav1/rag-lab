-- migrate:up
-- the same weights arrive clean through vLLM and with markup through a broker, so the cut lives on the row
ALTER TABLE models ADD COLUMN answer_parser text NOT NULL DEFAULT 'none';

-- migrate:down
ALTER TABLE models DROP COLUMN answer_parser;
