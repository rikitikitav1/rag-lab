-- migrate:up
-- a stand migrated with no embedding role seated kept unmarked vectors, and a search now refuses them
UPDATE data_chunks SET embedded_by = (
    SELECT m.name || '@' || e.name FROM model_roles r
    JOIN models m ON m.id = r.model_id JOIN engines e ON e.id = m.engine_id
    WHERE r.role = 'embedding'
) WHERE embedding IS NOT NULL AND embedded_by IS NULL;
UPDATE questions SET embedded_by = (
    SELECT m.name || '@' || e.name FROM model_roles r
    JOIN models m ON m.id = r.model_id JOIN engines e ON e.id = m.engine_id
    WHERE r.role = 'embedding'
) WHERE embedding IS NOT NULL AND embedded_by IS NULL;

-- migrate:down
SELECT 1;
