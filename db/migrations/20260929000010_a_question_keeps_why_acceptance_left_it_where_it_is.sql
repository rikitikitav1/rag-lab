-- migrate:up
-- one reason for a refused pair and for one left to the judge; the status says which, and the judge's population is one query
ALTER TABLE questions RENAME COLUMN refused_why TO acceptance_why;

-- migrate:down
ALTER TABLE questions RENAME COLUMN acceptance_why TO refused_why;
