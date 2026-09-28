-- migrate:up
-- the trigger first: a row written as `en` before it changes gets a `simple` tsvector and loses stemming silently
CREATE OR REPLACE FUNCTION data_chunks_content_tsv() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  NEW.content_tsv := to_tsvector(
    CASE NEW.language
      WHEN 'ru' THEN 'russian'
      WHEN 'en' THEN 'english'
      ELSE 'simple'
    END::regconfig,
    NEW.content
  );
  RETURN NEW;
END;
$$;

UPDATE data_chunks SET language = CASE language WHEN 'eng' THEN 'en' WHEN 'rus' THEN 'ru' ELSE language END;

-- three live questions an older detector read as so, tl and et: «xxxx…», «ping», «Redis stream сравни с Kafka»
UPDATE questions SET language = 'en' WHERE id IN (32566, 49410) AND language IN ('so', 'tl');
UPDATE questions SET language = 'ru' WHERE id = 29663 AND language = 'et';
UPDATE questions SET language = CASE language WHEN 'eng' THEN 'en' WHEN 'rus' THEN 'ru' ELSE language END;

-- a language outside the two refuses the migration rather than passing as a stray; the model keeps it after
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM data_chunks WHERE language NOT IN ('en', 'ru'))
     OR EXISTS (SELECT 1 FROM questions WHERE language NOT IN ('en', 'ru')) THEN
    RAISE EXCEPTION 'a language other than en and ru is left';
  END IF;
END;
$$;

-- migrate:down
UPDATE questions SET language = CASE language WHEN 'en' THEN 'eng' WHEN 'ru' THEN 'rus' ELSE language END;
CREATE OR REPLACE FUNCTION data_chunks_content_tsv() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  NEW.content_tsv := to_tsvector(
    CASE NEW.language
      WHEN 'rus' THEN 'russian'
      WHEN 'eng' THEN 'english'
      ELSE 'simple'
    END::regconfig,
    NEW.content
  );
  RETURN NEW;
END;
$$;
UPDATE data_chunks SET language = CASE language WHEN 'en' THEN 'eng' WHEN 'ru' THEN 'rus' ELSE language END;
