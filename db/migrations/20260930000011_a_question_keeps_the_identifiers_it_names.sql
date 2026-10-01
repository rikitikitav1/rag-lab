-- migrate:up
ALTER TABLE questions ADD COLUMN anchors jsonb;

-- migrate:down
ALTER TABLE questions DROP COLUMN anchors;
