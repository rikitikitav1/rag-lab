-- migrate:up
-- per variant, the words a keyword query may hold and the share of chunks holding each; the rare ones pick candidates
CREATE TABLE term_frequencies (
    variant text NOT NULL,
    lexeme text NOT NULL,
    chunks integer NOT NULL,
    share real NOT NULL,
    -- the variant's newest chunk when counted: a reindex writes new rows, so a newer one means the shares are stale
    newest_chunk integer NOT NULL,
    computed_at timestamp with time zone DEFAULT now() NOT NULL,
    PRIMARY KEY (variant, lexeme)
);

-- migrate:down
DROP TABLE term_frequencies;
