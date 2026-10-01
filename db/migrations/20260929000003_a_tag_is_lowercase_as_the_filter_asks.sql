-- migrate:up
-- tags carried over from the old category path kept its case; the index writes them lowercased and a label is asked so
UPDATE data_chunks SET tags = ARRAY(SELECT lower(t) FROM unnest(tags) AS t) WHERE tags::text <> lower(tags::text);

-- migrate:down
-- the case the old path had is not kept anywhere
SELECT 1;
