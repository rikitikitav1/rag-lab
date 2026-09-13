-- migrate:up
-- what each broker said was left on the key before and after the job's calls; the key's balance, not the job's alone
ALTER TABLE jobs ADD COLUMN balances jsonb;

-- migrate:down
ALTER TABLE jobs DROP COLUMN balances;
