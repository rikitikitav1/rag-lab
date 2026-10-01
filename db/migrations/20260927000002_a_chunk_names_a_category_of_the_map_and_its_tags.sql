-- migrate:up
-- the labels the path carried become tags, so a filter by label keeps finding what it found
ALTER TABLE data_chunks ADD COLUMN tags text[] NOT NULL DEFAULT '{}';
UPDATE data_chunks SET tags = string_to_array(category::text, '.');

-- a category is a key of config/categories.yaml; the old corpus names one only where its source file does
ALTER TABLE data_chunks ADD COLUMN category_key text;
UPDATE data_chunks dc SET category_key = CASE ds.name WHEN 'redis-doc' THEN 'redis' WHEN 'system-design-primer' THEN 'system-design' END
FROM data_sources ds WHERE ds.id = dc.source_id;

DROP INDEX data_chunks_category_idx;
ALTER TABLE data_chunks DROP COLUMN category;
ALTER TABLE data_chunks RENAME COLUMN category_key TO category;
CREATE INDEX data_chunks_category_idx ON data_chunks USING btree (category);
CREATE INDEX data_chunks_tags_idx ON data_chunks USING gin (tags);
DROP EXTENSION ltree;

-- migrate:down
CREATE EXTENSION IF NOT EXISTS ltree;
DROP INDEX data_chunks_tags_idx;
DROP INDEX data_chunks_category_idx;
ALTER TABLE data_chunks RENAME COLUMN category TO category_key;
ALTER TABLE data_chunks ADD COLUMN category ltree;
UPDATE data_chunks SET category = text2ltree(CASE WHEN cardinality(tags) > 0 THEN array_to_string(tags, '.') ELSE 'misc' END);
ALTER TABLE data_chunks ALTER COLUMN category SET NOT NULL;
ALTER TABLE data_chunks DROP COLUMN category_key;
ALTER TABLE data_chunks DROP COLUMN tags;
CREATE INDEX data_chunks_category_idx ON data_chunks USING gist (category);
