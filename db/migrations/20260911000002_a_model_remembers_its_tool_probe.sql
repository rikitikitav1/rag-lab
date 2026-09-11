-- migrate:up
-- the probe lived in the worker's memory and the door asked its own empty one, so it never refused
ALTER TABLE models ADD COLUMN tool_probe boolean;
-- a server's flags hold for one process start: an answer from an earlier start says nothing
ALTER TABLE models ADD COLUMN tool_probe_start text;

-- migrate:down
ALTER TABLE models DROP COLUMN tool_probe_start;
ALTER TABLE models DROP COLUMN tool_probe;
