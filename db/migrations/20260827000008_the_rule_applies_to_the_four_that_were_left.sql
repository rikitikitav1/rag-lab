-- migrate:up
-- the four checks 20260827000007 left: the model is the one place that decides a value
ALTER TABLE data_sources DROP CONSTRAINT IF EXISTS data_sources_ingest_quality_check;
ALTER TABLE model_roles DROP CONSTRAINT IF EXISTS model_roles_role_check;
ALTER TABLE models DROP CONSTRAINT IF EXISTS models_status_check;
ALTER TABLE prompts DROP CONSTRAINT IF EXISTS prompts_purpose_check;

-- migrate:down
ALTER TABLE data_sources
  ADD CONSTRAINT data_sources_ingest_quality_check
  CHECK (ingest_quality IS NULL OR ingest_quality IN ('ok', 'dirty', 'broken'));
ALTER TABLE model_roles
  ADD CONSTRAINT model_roles_role_check
  CHECK (role IN ('generation', 'embedding', 'judging', 'paraphrasing'));
ALTER TABLE models
  ADD CONSTRAINT models_status_check
  CHECK (status IN ('available', 'loading', 'ready'));
ALTER TABLE prompts
  ADD CONSTRAINT prompts_purpose_check
  CHECK (purpose IN ('generate.answer', 'judge.faithfulness', 'judge.relevance',
                     'judge.completeness', 'paraphrase.question', 'translate.question',
                     'agent.system', 'agent.fallback', 'agent.tool_match',
                     'agent.no_evidence'));
