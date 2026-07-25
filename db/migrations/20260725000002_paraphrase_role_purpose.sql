-- migrate:up
ALTER TABLE model_roles DROP CONSTRAINT model_roles_role_check;
ALTER TABLE model_roles ADD CONSTRAINT model_roles_role_check
  CHECK (role IN ('generation', 'embedding', 'judging', 'paraphrasing'));

ALTER TABLE prompts DROP CONSTRAINT prompts_purpose_check;
ALTER TABLE prompts ADD CONSTRAINT prompts_purpose_check
  CHECK (purpose IN ('generate.answer', 'judge.faithfulness', 'judge.relevance', 'paraphrase.question', 'translate.question'));

-- migrate:down
ALTER TABLE model_roles DROP CONSTRAINT model_roles_role_check;
ALTER TABLE model_roles ADD CONSTRAINT model_roles_role_check
  CHECK (role IN ('generation', 'embedding', 'judging'));

ALTER TABLE prompts DROP CONSTRAINT prompts_purpose_check;
ALTER TABLE prompts ADD CONSTRAINT prompts_purpose_check
  CHECK (purpose IN ('generate.answer', 'judge.faithfulness', 'judge.relevance'));
