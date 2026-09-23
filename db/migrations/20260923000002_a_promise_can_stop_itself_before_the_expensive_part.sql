-- migrate:up
-- [{column, above, margin, on}]: two columns inside one arm, read by point before the arms are spent
ALTER TABLE preregistrations ADD COLUMN vetoes jsonb NOT NULL DEFAULT '[]'::jsonb;

-- migrate:down
ALTER TABLE preregistrations DROP COLUMN vetoes;
