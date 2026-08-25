-- migrate:up
ALTER TABLE prompts DROP CONSTRAINT prompts_purpose_check;
ALTER TABLE prompts ADD CONSTRAINT prompts_purpose_check CHECK (purpose = ANY (ARRAY['generate.answer', 'judge.faithfulness', 'judge.relevance', 'judge.completeness', 'paraphrase.question', 'translate.question', 'agent.system', 'agent.fallback']));

-- migrate:down
DELETE FROM prompts WHERE purpose = 'agent.fallback';
ALTER TABLE prompts DROP CONSTRAINT prompts_purpose_check;
ALTER TABLE prompts ADD CONSTRAINT prompts_purpose_check CHECK (purpose = ANY (ARRAY['generate.answer', 'judge.faithfulness', 'judge.relevance', 'judge.completeness', 'paraphrase.question', 'translate.question', 'agent.system']));
