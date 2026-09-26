-- migrate:up
-- a scan goes to one tool and a text-layer PDF to another, so the one converter row becomes Docling's
UPDATE engines SET name = 'converter-docling', env_prefix = 'CONVERTER_DOCLING' WHERE name = 'converter' AND kind = 'converter';

-- migrate:down
UPDATE engines SET name = 'converter', env_prefix = 'CONVERTER' WHERE name = 'converter-docling' AND kind = 'converter';
