# rag-lab

RAG-система над личной базой знаний и внешними IT-репозиториями. Отвечает на технические вопросы, цитируя источники. Собрана **из примитивов** (без LangChain/LlamaIndex) на **локальном инференсе** (Ollama на своей GPU, через OpenAI-совместимый протокол).

Это **витрина-лаборатория**: стенд, чтобы руками отрабатывать инженерные подходы LLM/RAG (retrieval, eval на LLM-судье, lifecycle моделей, reranking, async-очереди) и показывать результат на цифрах. Не продакшн-сервис, а полигон подходов.

> English: [../README.md](../README.md) · Сценарии и полный справочник команд: [use_cases.md](use_cases.md)

## Что это

- **Гибридный retrieval**: векторный поиск (pgvector) + полнотекстовый (Postgres FTS с per-language стеммингом), слияние через **RRF** (Reciprocal Rank Fusion). Фильтр по иерархическим категориям (ltree), distance-порог с честным отказом на вопросах вне корпуса.
- **Локальные модели** через Ollama по ролям: генерация, эмбеддинги (`bge-m3`), LLM-судья. Роль отвязана от конкретной модели: какая модель обслуживает роль, хранится в БД и меняется на лету.
- **Мультиисточник**, единая таксономия: личные заметки (rus) + interview-репозитории Devinterview-io (eng, ~170 репо). Per-source стратегии ingestion.
- **Eval-стенд, 5 осей качества**: retrieval (hit@k / MRR), faithfulness (грунтован ли ответ в контексте), relevance (отвечает ли по существу), completeness (покрывает ли эталонный ответ), refusal-accuracy — через **LLM-as-judge** со structured output.
- **Асинхронный job-queue**: тяжёлые операции (pull/delete моделей, индексация корпуса, эмбеддинг банка вопросов) уходят в очередь на Postgres, их разбирает воркер. Сервис зависит только от Postgres: Ollama может быть недоступна на старте, джобы дефёрятся/ретраятся, приложение не падает.
- **Reranking (opt-in)**: cross-encoder `bge-reranker-v2-m3` поверх гибрида (retrieve-wide → rerank → narrow), включается флагом (per-request / per-run); A/B-протестирован на кросс-язычном наборе.
- **Агент (ReAct, из примитивов)**: рукописный tool-calling цикл, где модель сама решает когда искать по корпусу, может переформулировать запрос и сделать несколько хопов, затем отвечает. Выбирается как eval-пайплайн (`pipeline: agent`) и меряется в лоб против single-shot RAG (см. лог экспериментов).
- **Route-driven eval-платформа**: генерация не-циркулярных наборов (LLM-парафраз interview-вопросов + перевод на ru), импорт вопросов файлом, прогоны и оценка судьёй — всё булково через очередь; наблюдаемость через лог запросов (`question-log`) и джобы с `elapsed`.
- **Слоёвка**: транспортно-нейтральные `use_cases` → тонкие адаптеры (CLI / FastAPI REST).

## Стек

Python · PostgreSQL + pgvector · SQLAlchemy 2.0 (sync psycopg + async asyncpg) · Ollama (GPU, OpenAI-совместимый API) · FastAPI · dbmate (миграции) · uv/pyproject · Docker Compose.

## Архитектура моделей и промптов

- **Model / ModelRole**: `Model` (имя + статус: available/loading/ready), `ModelRole` (role как PK → одна модель на роль по построению, FK `ON DELETE RESTRICT` = БД не даст удалить назначенную модель). Резолвер `llm.resolve_name(role)` берёт имя из БД, параметры инференса — из `config.yaml`.
- **Prompt**: версионирование в БД (`purpose` + `version`, ровно один `active` на назначение). Промпты-исходники лежат файлами в `prompts/` (формат `<purpose>.v<N>.txt`), сид заливает их в БД, активной становится свежая версия.
- **Bootstrap на старте** (idempotent): завести Model-строки из конфига → засидить роли → сверить с Ollama (что не скачано → статус `loading` + джоба pull) → заэнкьюить индексацию, если корпус пуст → заэнкьюить эмбеддинг вопросов без вектора.

## Конфигурация

Всё тюнингуемое вынесено в **`config.yaml`** (монтируется в контейнер):

- `llm.roles` — модель + `options` на каждую роль (`generation` / `embedding` / `judging` / `paraphrasing`); `llm.candidates` — модели, которые тоже скачать, но не назначить.
- `service.retrieval` — `distance_threshold`, `results_limit`, `rrf_k`, лимиты кандидатов.
- `service.rerank` — `enabled`, `model`, `candidates`, `top`.
- `service.agent` — `max_hops` (кап хопов ReAct).
- `service.ingestion` — `chunk_max_size`, `batch_size`, `commit_size`.
- `service.sources` — источники (interview-репозитории и их base_url).
- `postgres` — подключение к БД.

В коде остаются source-специфичные деревья категорий.

## Быстрый старт

```bash
docker compose up -d
curl localhost:8000/readiness            # pg обязателен, ollama мягко
curl -X POST localhost:8000/v1/chat/question \
  -H 'Content-Type: application/json' -d '{"text": "What is a hash table?"}'
# Swagger: http://localhost:8000/docs
```

Первый `up` тянет ~16 ГБ моделей и строит индекс (~5-10 мин, следи за `docker compose logs -f worker`). Сервер стартует **не дожидаясь** моделей и индексации, поэтому первые запросы могут вернуть отказ, пока корпус наполняется.

Полные сценарии (мини-eval до цифр, reranking A/B, импорт своих вопросов, просмотр логов) и справочник команд: **[use_cases.md](use_cases.md)**.

## Сервисы compose

| Сервис | Роль |
|--------|------|
| `postgresql` | Postgres + pgvector, единственная жёсткая зависимость |
| `dbmate` | накатывает миграции, отрабатывает до старта остальных |
| `seed` | заливает промпты и банк вопросов, отрабатывает один раз после миграций |
| `rag-lab` | FastAPI-сервер (uvicorn) + bootstrap на старте |
| `worker` | разбирает job-queue (pull/delete/index/embed/paraphrase/eval/judge) |
| `ollama` | локальный инференс на GPU |

## REST API

Полный интерактивный справочник в Swagger: `/docs`.

Листинговые эндпоинты (`/v1/model`, `/v1/prompt`, `/v1/job`, `/v1/question-log`) используют общую пагинацию: `limit` (по умолчанию 100, максимум 1000), `offset`, `sort_by`, `sort_order` (`asc`/`desc`, по умолчанию `desc`).

Health:
- `GET /liveness`, `GET /readiness`

Чат и поиск:
- `POST /v1/chat/question` (полный RAG-ответ; опц. флаг `rerank`; опц. override языка `language` `ru`/`en`)
- `POST /v1/chat/fast_question` (только retrieval, без генерации)
- `POST /v1/agent/question` (ответ ReAct-агента; опц. `max_hops`, `language`, `debug` для полного трейса сообщений)
- `GET /v1/categories` (дерево категорий с числом чанков)

Lifecycle моделей:
- `GET /v1/model`, `GET /v1/model/{id}`, `POST /v1/model` (create ставит джобу pull), `DELETE /v1/model/{id}` (409 если назначена роли)
- `GET /v1/role`, `PUT /v1/role/{role}` (назначить модель на роль)

Промпты:
- `GET /v1/prompt`, `GET /v1/prompt/{id}`, `POST /v1/prompt`, `POST /v1/prompt/{id}/activate`, `DELETE /v1/prompt/{id}`

Eval-платформа:
- `POST /v1/eval/paraphrase` (сгенерить парафраз-набор), `POST /v1/eval/run` (прогнать набор → судья; `pipeline: single_shot|agent`, опц. `rerank`; 400 если `rerank` вместе с `pipeline: agent`)
- `GET /v1/eval/misses?run_name=X` (retrieval-промахи прогона: in-corpus вопросы, где ожидаемый источник не найден, expected vs retrieved)
- `POST /v1/questions/import` (залить файл вопросов, ≤5 МБ; опц. цепочка run)

Наблюдаемость:
- `GET /v1/question-log`, `GET /v1/question-log/{id}` (логи ответов; фильтры вкл. `pipeline`, `faithfulness`/`relevance`/`completeness`, `run_name`; детально с context)
- `GET /v1/job`, `GET /v1/job/{id}` (джобы + elapsed)

## Как это устроено

- `app/config.py` — loader `config.yaml`.
- `app/orm/` — SQLAlchemy: `base` (declarative), `sync_db` (psycopg), `async_db` (asyncpg).
- `app/models/` — ORM-модели: `registry` (Model/ModelRole/Prompt), `eval` (Question/QuestionLog), `jobs` (Job), `corpus` (DataSource/DataChunk).
- `app/llm.py` — клиент Ollama через OpenAI SDK (генерация / эмбеддинги / structured output) + резолвер роль→модель.
- `app/rerank.py` — cross-encoder реранкер (FlagEmbedding, CPU, ленивая загрузка).
- `app/job_queue.py`, `app/worker.py`, `app/job_handlers/` — очередь на Postgres (FOR UPDATE SKIP LOCKED) и воркер с ретраями/дефёром; хендлеры разнесены по тематике.
- `app/bootstrap.py` — idempotent-инициализация на старте.
- `app/sources/` — per-source ingestion (reader-паттерн: `Base` ABC + источники).
- `app/db.py` — гибридный поиск (сырой SQL: pgvector `<=>`, FTS, ltree, RRF).
- `app/use_cases/` — `chat` (retrieve/answer), `agent` (ReAct tool-calling цикл), `index` (сбор корпуса), `judge` (оценка ответа).
- `app/agent_tools.py` — реестр тулов + `dispatch` + тул `search_corpus` поверх гибридного поиска.
- `app/api/` — REST-адаптеры (health + v1: chat / agent / categories / model / role / source / prompt / eval / questions / question-log / job).
- `app/seed.py`, `app/console.py` — сид промптов/банка вопросов; REPL-консоль.
- `app/evals/` — eval-стенд (runner + метрики retrieval + generation через judge).
- `tests/` — unit-тесты (чистая логика, без DB/Ollama): `docker compose exec rag-lab pytest -q`.

## Статус

Учебный проект: цель — освоить RAG/LLM-инженерию руками, из примитивов. RAG из примитивов (гибрид, ltree, FTS) + FastAPI-сервер + 4-осевой eval-стенд на LLM-judge + production-слой (pyproject/uv, централизованный config, SQLAlchemy ORM sync+async, OpenAI-совместимый клиент, role-keyed lifecycle моделей, версионирование промптов, async job-queue, банк вопросов, reranking, route-driven eval-платформа).
