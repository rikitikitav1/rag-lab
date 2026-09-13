-- migrate:up
-- the stand counts tokens, not money, and nothing ever wrote these; a budget lives in a proxy like LiteLLM
ALTER TABLE engines DROP COLUMN budget, DROP COLUMN spent, DROP COLUMN reserved;

-- migrate:down
ALTER TABLE engines
    ADD COLUMN budget numeric(12,6),
    ADD COLUMN spent numeric(12,6) DEFAULT 0 NOT NULL,
    ADD COLUMN reserved numeric(12,6) DEFAULT 0 NOT NULL;
