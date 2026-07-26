-- migrate:up
ALTER TABLE prompts DROP CONSTRAINT prompts_purpose_check;
ALTER TABLE prompts ADD CONSTRAINT prompts_purpose_check CHECK (purpose = ANY (ARRAY['generate.answer', 'judge.faithfulness', 'judge.relevance', 'paraphrase.question', 'translate.question', 'agent.system']));

-- migrate:down
ALTER TABLE prompts DROP CONSTRAINT prompts_purpose_check;
ALTER TABLE prompts ADD CONSTRAINT prompts_purpose_check CHECK (purpose = ANY (ARRAY['generate.answer', 'judge.faithfulness', 'judge.relevance', 'paraphrase.question', 'translate.question']));
