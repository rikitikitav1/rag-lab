-- migrate:up
-- the two languages of one fact share a pair id, so a set is split by pair and never leaks the fact into both halves
ALTER TABLE questions ADD COLUMN pair_id text;
-- the section's own words the generated answer rests on, checked as a verbatim substring when the pair was written
ALTER TABLE questions ADD COLUMN evidence text;

-- migrate:down
ALTER TABLE questions DROP COLUMN evidence;
ALTER TABLE questions DROP COLUMN pair_id;
