-- migrate:up
-- two engines under one model name reordered the top-20 of 172 questions in 200, so a vector names both
ALTER TABLE data_chunks ADD COLUMN embedded_by text;
ALTER TABLE questions ADD COLUMN embedded_by text;
-- the rows before the column were written by the embedding role of that day, the only one recorded
UPDATE data_chunks SET embedded_by = (
    SELECT m.name || '@' || e.name FROM model_roles r
    JOIN models m ON m.id = r.model_id JOIN engines e ON e.id = m.engine_id
    WHERE r.role = 'embedding'
) WHERE embedding IS NOT NULL;
UPDATE questions SET embedded_by = (
    SELECT m.name || '@' || e.name FROM model_roles r
    JOIN models m ON m.id = r.model_id JOIN engines e ON e.id = m.engine_id
    WHERE r.role = 'embedding'
) WHERE embedding IS NOT NULL;
CREATE INDEX data_chunks_variant_embedded_by_idx ON data_chunks (variant, embedded_by);

-- migrate:down
DROP INDEX data_chunks_variant_embedded_by_idx;
ALTER TABLE questions DROP COLUMN embedded_by;
ALTER TABLE data_chunks DROP COLUMN embedded_by;
