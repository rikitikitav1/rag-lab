\restrict dbmate

-- Dumped from database version 17.10 (Debian 17.10-1.pgdg12+1)
-- Dumped by pg_dump version 18.4

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: ltree; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS ltree WITH SCHEMA public;


--
-- Name: EXTENSION ltree; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION ltree IS 'data type for hierarchical tree-like structures';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: data_chunks_content_tsv(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.data_chunks_content_tsv() RETURNS trigger
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


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: data_chunks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.data_chunks (
    id integer NOT NULL,
    source_id integer NOT NULL,
    source text NOT NULL,
    content text NOT NULL,
    embedding public.vector(1024),
    chunk_index integer NOT NULL,
    category public.ltree NOT NULL,
    language text NOT NULL,
    content_tsv tsvector
);


--
-- Name: data_chunks_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.data_chunks_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: data_chunks_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.data_chunks_id_seq OWNED BY public.data_chunks.id;


--
-- Name: data_sources; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.data_sources (
    id integer NOT NULL,
    name character varying(256) NOT NULL,
    kind character varying(32) NOT NULL,
    git_url text,
    path text,
    active boolean DEFAULT true NOT NULL
);


--
-- Name: data_sources_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.data_sources_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: data_sources_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.data_sources_id_seq OWNED BY public.data_sources.id;


--
-- Name: experiments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.experiments (
    id integer NOT NULL,
    name text,
    status text DEFAULT 'draft'::text NOT NULL,
    dataset text NOT NULL,
    sample_size integer,
    sample_seed integer,
    question_ids jsonb,
    data_prep jsonb DEFAULT '{}'::jsonb NOT NULL,
    procedure jsonb DEFAULT '{}'::jsonb NOT NULL,
    param text NOT NULL,
    param_values jsonb DEFAULT '[]'::jsonb NOT NULL,
    run_names jsonb DEFAULT '[]'::jsonb NOT NULL,
    results jsonb,
    conclusion text,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    elapsed double precision,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: experiments_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.experiments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: experiments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.experiments_id_seq OWNED BY public.experiments.id;


--
-- Name: jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.jobs (
    id integer NOT NULL,
    type text NOT NULL,
    status text DEFAULT 'new'::text NOT NULL,
    options jsonb DEFAULT '{}'::jsonb NOT NULL,
    error jsonb,
    apply_since timestamp with time zone DEFAULT now() NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    elapsed double precision,
    queue text DEFAULT 'default'::text NOT NULL
);


--
-- Name: jobs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.jobs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: jobs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.jobs_id_seq OWNED BY public.jobs.id;


--
-- Name: mcp_integrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mcp_integrations (
    id integer NOT NULL,
    name character varying(64) NOT NULL,
    url text NOT NULL,
    status text DEFAULT 'disabled'::text NOT NULL,
    allowed_tools jsonb DEFAULT '[]'::jsonb NOT NULL,
    tool_schemas jsonb DEFAULT '{}'::jsonb NOT NULL,
    auth jsonb,
    timeout_s integer DEFAULT 30 NOT NULL,
    max_result_chars integer DEFAULT 4000 NOT NULL,
    last_checked_at timestamp with time zone,
    last_error text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: mcp_integrations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.mcp_integrations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: mcp_integrations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.mcp_integrations_id_seq OWNED BY public.mcp_integrations.id;


--
-- Name: model_roles; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_roles (
    role text NOT NULL,
    model_id integer NOT NULL,
    CONSTRAINT model_roles_role_check CHECK ((role = ANY (ARRAY['generation'::text, 'embedding'::text, 'judging'::text, 'paraphrasing'::text])))
);


--
-- Name: models; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.models (
    id integer NOT NULL,
    name text NOT NULL,
    status text DEFAULT 'available'::text NOT NULL,
    CONSTRAINT models_status_check CHECK ((status = ANY (ARRAY['available'::text, 'loading'::text, 'ready'::text])))
);


--
-- Name: models_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.models_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: models_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.models_id_seq OWNED BY public.models.id;


--
-- Name: prompts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompts (
    id integer NOT NULL,
    purpose text NOT NULL,
    version integer NOT NULL,
    template text NOT NULL,
    active boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT prompts_purpose_check CHECK ((purpose = ANY (ARRAY['generate.answer'::text, 'judge.faithfulness'::text, 'judge.relevance'::text, 'judge.completeness'::text, 'paraphrase.question'::text, 'translate.question'::text, 'agent.system'::text, 'agent.fallback'::text, 'agent.tool_match'::text, 'agent.no_evidence'::text])))
);


--
-- Name: prompts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prompts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompts_id_seq OWNED BY public.prompts.id;


--
-- Name: question_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.question_logs (
    id integer NOT NULL,
    run_name text,
    question_id integer,
    answered boolean NOT NULL,
    answer text,
    sources jsonb,
    models jsonb DEFAULT '{}'::jsonb NOT NULL,
    prompts jsonb DEFAULT '{}'::jsonb NOT NULL,
    prompt_tokens integer,
    completion_tokens integer,
    elapsed double precision,
    faithfulness text,
    relevance text,
    metrics jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    context text,
    pipeline text DEFAULT 'single_shot'::text NOT NULL,
    completeness text
);


--
-- Name: question_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.question_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: question_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.question_logs_id_seq OWNED BY public.question_logs.id;


--
-- Name: questions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.questions (
    id integer NOT NULL,
    text_hash character varying(64) NOT NULL,
    original_text text NOT NULL,
    normalized_text text,
    reference_answer text,
    marked_sources text[] DEFAULT '{}'::text[] NOT NULL,
    set_name text,
    language text,
    kind text,
    status text,
    embedding public.vector(1024),
    source_question_id integer
);


--
-- Name: questions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.questions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: questions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.questions_id_seq OWNED BY public.questions.id;


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_migrations (
    version character varying NOT NULL
);


--
-- Name: data_chunks id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.data_chunks ALTER COLUMN id SET DEFAULT nextval('public.data_chunks_id_seq'::regclass);


--
-- Name: data_sources id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.data_sources ALTER COLUMN id SET DEFAULT nextval('public.data_sources_id_seq'::regclass);


--
-- Name: experiments id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiments ALTER COLUMN id SET DEFAULT nextval('public.experiments_id_seq'::regclass);


--
-- Name: jobs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs ALTER COLUMN id SET DEFAULT nextval('public.jobs_id_seq'::regclass);


--
-- Name: mcp_integrations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mcp_integrations ALTER COLUMN id SET DEFAULT nextval('public.mcp_integrations_id_seq'::regclass);


--
-- Name: models id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.models ALTER COLUMN id SET DEFAULT nextval('public.models_id_seq'::regclass);


--
-- Name: prompts id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompts ALTER COLUMN id SET DEFAULT nextval('public.prompts_id_seq'::regclass);


--
-- Name: question_logs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.question_logs ALTER COLUMN id SET DEFAULT nextval('public.question_logs_id_seq'::regclass);


--
-- Name: questions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.questions ALTER COLUMN id SET DEFAULT nextval('public.questions_id_seq'::regclass);


--
-- Name: data_chunks data_chunks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.data_chunks
    ADD CONSTRAINT data_chunks_pkey PRIMARY KEY (id);


--
-- Name: data_sources data_sources_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.data_sources
    ADD CONSTRAINT data_sources_name_key UNIQUE (name);


--
-- Name: data_sources data_sources_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.data_sources
    ADD CONSTRAINT data_sources_pkey PRIMARY KEY (id);


--
-- Name: experiments experiments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiments
    ADD CONSTRAINT experiments_pkey PRIMARY KEY (id);


--
-- Name: jobs jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs
    ADD CONSTRAINT jobs_pkey PRIMARY KEY (id);


--
-- Name: mcp_integrations mcp_integrations_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mcp_integrations
    ADD CONSTRAINT mcp_integrations_name_key UNIQUE (name);


--
-- Name: mcp_integrations mcp_integrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mcp_integrations
    ADD CONSTRAINT mcp_integrations_pkey PRIMARY KEY (id);


--
-- Name: model_roles model_roles_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_roles
    ADD CONSTRAINT model_roles_pkey PRIMARY KEY (role);


--
-- Name: models models_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.models
    ADD CONSTRAINT models_name_key UNIQUE (name);


--
-- Name: models models_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.models
    ADD CONSTRAINT models_pkey PRIMARY KEY (id);


--
-- Name: prompts prompts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompts
    ADD CONSTRAINT prompts_pkey PRIMARY KEY (id);


--
-- Name: prompts prompts_purpose_version_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompts
    ADD CONSTRAINT prompts_purpose_version_key UNIQUE (purpose, version);


--
-- Name: question_logs question_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.question_logs
    ADD CONSTRAINT question_logs_pkey PRIMARY KEY (id);


--
-- Name: questions questions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.questions
    ADD CONSTRAINT questions_pkey PRIMARY KEY (id);


--
-- Name: questions questions_text_hash_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.questions
    ADD CONSTRAINT questions_text_hash_key UNIQUE (text_hash);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: data_chunks_category_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX data_chunks_category_idx ON public.data_chunks USING gist (category);


--
-- Name: data_chunks_content_tsv_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX data_chunks_content_tsv_idx ON public.data_chunks USING gin (content_tsv);


--
-- Name: data_chunks_embedding_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX data_chunks_embedding_idx ON public.data_chunks USING hnsw (embedding public.vector_cosine_ops);


--
-- Name: data_chunks_source_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX data_chunks_source_id_idx ON public.data_chunks USING btree (source_id);


--
-- Name: idx_jobs_queue_status_apply_since; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_jobs_queue_status_apply_since ON public.jobs USING btree (queue, status, apply_since);


--
-- Name: idx_question_logs_pipeline; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_question_logs_pipeline ON public.question_logs USING btree (pipeline);


--
-- Name: one_active_prompt_per_purpose; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX one_active_prompt_per_purpose ON public.prompts USING btree (purpose) WHERE active;


--
-- Name: question_logs_run_name_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX question_logs_run_name_idx ON public.question_logs USING btree (run_name);


--
-- Name: data_chunks data_chunks_content_tsv_trg; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER data_chunks_content_tsv_trg BEFORE INSERT OR UPDATE ON public.data_chunks FOR EACH ROW EXECUTE FUNCTION public.data_chunks_content_tsv();


--
-- Name: data_chunks data_chunks_source_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.data_chunks
    ADD CONSTRAINT data_chunks_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.data_sources(id) ON DELETE CASCADE;


--
-- Name: model_roles model_roles_model_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_roles
    ADD CONSTRAINT model_roles_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.models(id) ON DELETE RESTRICT;


--
-- Name: question_logs question_logs_question_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.question_logs
    ADD CONSTRAINT question_logs_question_id_fkey FOREIGN KEY (question_id) REFERENCES public.questions(id);


--
-- Name: questions questions_source_question_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.questions
    ADD CONSTRAINT questions_source_question_id_fkey FOREIGN KEY (source_question_id) REFERENCES public.questions(id);


--
-- PostgreSQL database dump complete
--

\unrestrict dbmate


--
-- Dbmate schema migrations
--

INSERT INTO public.schema_migrations (version) VALUES
    ('20260721000001'),
    ('20260721000002'),
    ('20260721000003'),
    ('20260721000004'),
    ('20260722000001'),
    ('20260725000001'),
    ('20260725000002'),
    ('20260725000003'),
    ('20260725000004'),
    ('20260726000001'),
    ('20260726000002'),
    ('20260726000003'),
    ('20260726000004'),
    ('20260728000001'),
    ('20260728000002'),
    ('20260728000003'),
    ('20260729000001'),
    ('20260824000001'),
    ('20260824000002'),
    ('20260825000001');
