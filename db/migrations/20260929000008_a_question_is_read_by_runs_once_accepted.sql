-- migrate:up
-- a run reads a question once accepted; the column was never written, so it is made again with every set accepted
ALTER TABLE questions DROP COLUMN status;
-- the values are the model's to know, as every enum of the stand
ALTER TABLE questions ADD COLUMN status text NOT NULL DEFAULT 'accepted';
-- why the acceptance refused it; a refused question stays, so the shares can be read again later
ALTER TABLE questions ADD COLUMN refused_why text;

-- migrate:down
ALTER TABLE questions DROP COLUMN refused_why;
ALTER TABLE questions DROP COLUMN status;
ALTER TABLE questions ADD COLUMN status text;
