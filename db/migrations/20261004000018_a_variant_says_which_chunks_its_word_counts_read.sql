-- migrate:up
-- one row a variant: how many live chunks the word counts were read over and the newest of them
CREATE TABLE term_frequency_counts (
    variant text PRIMARY KEY,
    chunks integer NOT NULL,
    newest_chunk integer NOT NULL,
    computed_at timestamp with time zone DEFAULT now() NOT NULL
);

-- migrate:down
DROP TABLE term_frequency_counts;
