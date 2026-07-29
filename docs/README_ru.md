# rag-lab

RAG-система над личной базой знаний и внешними IT-репозиториями. Отвечает на технические вопросы и возвращает источники, использованные как контекст. Собрана **из примитивов** (без LangChain/LlamaIndex) на **локальном инференсе** (Ollama на своей GPU, через OpenAI-совместимый протокол).

Это **витрина-лаборатория**: стенд, чтобы руками отрабатывать инженерные подходы LLM/RAG (retrieval, eval на LLM-судье, lifecycle моделей, reranking, async-очереди) и показывать результат на цифрах. Не продакшн-сервис, а полигон подходов.

> English: [../README.md](../README.md) · Сценарии и полный справочник команд: [use_cases.md](use_cases.md)

## Что это

- **Гибридный retrieval**: векторный поиск (pgvector) + полнотекстовый (Postgres FTS с per-language стеммингом), слияние через **RRF** (Reciprocal Rank Fusion). Фильтр по иерархическим категориям (ltree), distance-порог с честным отказом на вопросах вне корпуса.
- **Локальные модели** через Ollama по ролям: генерация, эмбеддинги (`bge-m3`), LLM-судья. Роль отвязана от конкретной модели: какая модель обслуживает роль, хранится в БД и меняется на лету.
- **Мультиисточник**, единая таксономия: личные заметки (rus) + interview-репозитории Devinterview-io (eng, ~170 репо). Per-source стратегии ingestion.
- **Eval-стенд, 5 осей качества**: retrieval (hit@k / MRR), faithfulness (грунтован ли ответ в контексте), relevance (отвечает ли по существу), completeness (покрывает ли эталонный ответ), refusal-accuracy - через **LLM-as-judge** со structured output.
- **Асинхронный job-queue**: тяжёлые операции (pull/delete моделей, индексация корпуса, эмбеддинг банка вопросов) уходят в очередь на Postgres, их разбирает воркер. Сервис зависит только от Postgres: Ollama может быть недоступна на старте, джобы дефёрятся/ретраятся, приложение не падает.
- **Reranking (opt-in)**: cross-encoder `bge-reranker-v2-m3` поверх гибрида (retrieve-wide → rerank → narrow), включается флагом (per-request / per-run); A/B-протестирован на кросс-язычном наборе.
- **Агент (ReAct, из примитивов)**: рукописный tool-calling цикл, где модель сама решает когда искать по корпусу, может переформулировать запрос и сделать несколько хопов, затем отвечает. Выбирается как eval-пайплайн (`pipeline: agent`) и меряется в лоб против single-shot RAG (см. лог экспериментов).
- **Route-driven eval-платформа**: генерация не-циркулярных наборов (LLM-парафраз interview-вопросов + перевод на ru), импорт вопросов файлом, прогоны и оценка судьёй - всё булково через очередь; наблюдаемость через лог запросов (`question-log`) и джобы с `elapsed`.
- **Слоёвка**: транспортно-нейтральные `use_cases` → тонкие адаптеры (CLI / FastAPI REST).

## Стек

Python · PostgreSQL + pgvector · SQLAlchemy 2.0 (sync psycopg + async asyncpg) · Ollama (GPU, OpenAI-совместимый API) · FastAPI · dbmate (миграции) · uv/pyproject · Docker Compose.

## Архитектура моделей и промптов

- **Model / ModelRole**: `Model` (имя + статус: available/loading/ready), `ModelRole` (role как PK → одна модель на роль по построению, FK `ON DELETE RESTRICT` = БД не даст удалить назначенную модель). Резолвер `llm.resolve_name(role)` берёт имя из БД, параметры инференса - из `config.yaml`.
- **Prompt**: версионирование в БД (`purpose` + `version`, ровно один `active` на назначение). Промпты-исходники лежат файлами в `prompts/` (формат `<purpose>.v<N>.txt`), сид заливает их в БД, активной становится свежая версия.
- **Bootstrap на старте** (idempotent): завести Model-строки из конфига → засидить роли → сверить с Ollama (что не скачано → статус `loading` + джоба pull) → заэнкьюить индексацию, если корпус пуст → заэнкьюить эмбеддинг вопросов без вектора.

## Конфигурация

Всё тюнингуемое вынесено в **`config.yaml`** (монтируется в контейнер):

- `llm.roles` - модель + `options` на каждую роль (`generation` / `embedding` / `judging` / `paraphrasing`); `llm.candidates` - модели, которые тоже скачать, но не назначить.
- `service.retrieval` - `distance_threshold`, `results_limit`, `rrf_k`, лимиты кандидатов.
- `service.rerank` - `enabled`, `model`, `candidates`, `top`.
- `service.agent` - `max_hops` (кап хопов ReAct).
- `service.ingestion` - `chunk_max_size`, `batch_size`, `commit_size`.
- `service.sources` - источники (interview-репозитории и их base_url).
- `postgres` - подключение к БД.

В коде остаются source-специфичные деревья категорий.

## Быстрый старт

```bash
docker compose up -d
curl localhost:8000/readiness            # pg обязателен, ollama мягко
curl -X POST localhost:8000/v1/chat/question \
  -H 'Content-Type: application/json' -d '{"text": "What is a hash table?"}'
# Swagger: http://localhost:8000/docs
```

Аутентификации нет намеренно (REST, `/mcp`, `/mcp-ops` открыты): это локальная лаборатория, порты привязаны к 127.0.0.1. Не выставляйте её в сеть как есть.

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
- `POST /v1/eval/paraphrase` (сгенерить парафраз-набор), `POST /v1/eval/run` (прогнать набор → судья; `pipeline: single_shot|agent`, per-run override'ы `rerank`, `k` (ширина ретривала), `max_hops` (кап хопов), `model` (генератор); конфиг задаёт только дефолты)
- `POST /v1/eval/experiment` (серия прогонов: параметр `param` (`k`, `max_hops` или `model`) варьируется по списку `values`, один авто-именованный прогон на значение, каждый судится; набор/пайплайн/язык фиксированы для чистого сравнения по одной переменной; значение `model`, которого нет в реестре, создаётся и скачивается, прогон ждёт готовности)
- `GET /v1/eval/misses?run_name=X` (retrieval-промахи прогона: in-corpus вопросы, где ожидаемый источник не найден, expected vs retrieved)
- `POST /v1/questions/import` (залить файл вопросов, ≤5 МБ; опц. цепочка run)

Эксперименты (полноценная сущность над сырым роутом серии):
- `POST /v1/experiment` (создаёт эксперимент - dataset + детерминированная выборка по seed / снапшот процедуры / варьируемый параметр - и ставит серию прогонов), `GET /v1/experiment` (список с фильтрами), `GET /v1/experiment/{id}`, `PUT /v1/experiment/{id}/conclusion`
- стейт-машина `draft → running → aggregated → concluded` (+ `failed`); после последней judge-джобы серии агрегатор считает метрики по каждому значению + RRF-композит по трём генеративным осям и пишет в `results` (retrieval hit@k/MRR отдаются per value, но в фьюжен не входят: hit@k монотонен по `k` и конфаундил бы композит)
- в results лежат **парные статистики значимости**, а не только точечные оценки: для победителя против каждого другого значения, по осям - средняя парная дельта (один и тот же вопрос в обоих прогонах), bootstrap 95% CI и p-value Уилкоксона, плюс флаги значимости с поправкой Бонферрони на всё семейство тестов - JSON сам говорит, что пережило поправку на множественные сравнения

Один вызов = один вопрос стенду; генерация, судейство и агрегация происходят в фоне. Вопросы, которые так можно задавать:

```bash
# "Сколько чанков кормить генератору?" - свип ширины ретривала
curl -sX POST localhost:8000/v1/experiment -H 'Content-Type: application/json' -d '{
  "name": "k_sweep", "dataset": "paraphrased_ru", "sample_size": 100,
  "pipeline": "agent", "language": "ru", "param": "k", "param_values": [1, 3, 5, 7, 10]}'

# "Окупает ли себя модель побольше на моём корпусе?" - A/B моделей
# (модели, которой нет в реестре, скачается автоматически, прогон подождёт)
curl -sX POST localhost:8000/v1/experiment -H 'Content-Type: application/json' -d '{
  "name": "model_ab", "dataset": "paraphrased_ru", "sample_size": 100,
  "param": "model", "param_values": ["llama3.1:8b", "gemma2:9b"]}'

# "Дают ли что-то лишние хопы агента?" - свип капа хопов на дешёвой выборке из 10 вопросов
curl -sX POST localhost:8000/v1/experiment -H 'Content-Type: application/json' -d '{
  "name": "hops", "dataset": "paraphrased_ru", "sample_size": 10, "sample_seed": 42,
  "pipeline": "agent", "param": "max_hops", "param_values": [2, 4, 6]}'
```

Когда серия отсужена, `GET /v1/experiment/{id}` возвращает метрики по каждому значению и композитный вердикт:

```json
{
  "status": "aggregated",
  "results": {
    "per_value": {"5": {"faithfulness": 7.18, "relevance": 8.9, "completeness": 6.16, "hit_at_k": 0.9, "mrr": 0.757}, "...": "..."},
    "composite": {
      "method": "rrf", "winner": "5",
      "ranking": [{"value": "5", "rrf": 0.0487}, {"value": "10", "rrf": 0.0484}],
      "pairwise": {
        "comparisons": {"5_vs_10": {"faithfulness": {"mean_delta": 0.19, "ci95": [-0.17, 0.57], "p": 0.3669, "n": 100, "significant_raw": false, "significant_bonferroni": false}, "...": "..."}},
        "method": "bonferroni", "alpha": 0.05, "tests": 15, "threshold": 0.00333
      }
    }
  }
}
```

Блок pairwise - то, что держит выводы честными: ранняя версия стенда «заключила», что k=5 лучше k=10, по разнице композита в третьем знаке - парный тест показывает, что это сравнение монетка (p=0.37), а флаги с поправкой отмечают, какие из 15 тестов решётки вообще выживают. Случаи, где это развернуло наши собственные вердикты - в [experiments.md](experiments.md).

Вывод фиксируется через `PUT /v1/experiment/{id}/conclusion`, и эксперимент становится самодостаточным артефактом: что варьировали, на каких данных, цифры, вердикт.

Наблюдаемость:
- `GET /v1/question-log`, `GET /v1/question-log/{id}` (логи ответов; фильтры вкл. `pipeline`, `faithfulness`/`relevance`/`completeness`, `run_name`; детально с context)
- `GET /v1/job`, `GET /v1/job/{id}` (джобы + elapsed), `POST /v1/job/{id}/cancel` (отменяет джобу и её зависимый judge, кооперативный стоп для запущенного прогона)

## MCP

MCP-сервер (Model Context Protocol) примонтирован на `/mcp` (streamable HTTP) и отдаёт корпус любому MCP-клиенту (Claude Desktop, Cursor, IDE-агенты). Построен на standalone `fastmcp`, переиспользует те же примитивы поиска, что REST/agent. Тулы:
- `search_corpus(query, category?)` - гибридный поиск, отдаёт чанки с маркерами `[source]`; опц. фильтр по категории.
- `answer_question(text, pipeline?, category?, language?)` - полный RAG-ответ, возвращает `{answer, retrieved, sources}` (`agent` или `single_shot`; `category` только с `single_shot`).
- `list_categories(category?, only_top?)` - пути категорий с количеством чанков, для discovery валидных значений фильтра перед поиском.

Подключение: `claude mcp add --transport http rag-lab http://127.0.0.1:8000/mcp/`, или MCP Inspector на тот же URL.

Второй, отдельный ops-сервер примонтирован на `/mcp-ops` - control plane eval-платформы, вынесенный с продуктовой поверхности (внешний клиент получает search/answer, а не админские глаголы):
- `run_metrics(run_name)` - агрегированные метрики прогона (генеративные оси + retrieval hit@k/MRR).
- `compare_runs(run_names)` - сравнение прогонов бок о бок с RRF-композитным ранжированием (только генеративные оси).
- `list_jobs(status?, type?, run_name?)` / `cancel_job(id)` - управление очередью джоб, cancel снимает и зависимый judge.

### MCP-клиент: агент потребляет внешние серверы

Лаборатория теперь обе стороны протокола: свой MCP-сервер выше и MCP-*клиент* ниже. Внешние hosted-серверы регистрируются как `McpIntegration`, их тулы попадают в тулбокс агента рядом с `search_corpus` под неймспейсом `integration__tool` (например `deepwiki__ask_question`). Агент сам решает на каждом хопе, идти ли за пределы корпуса; успешный внешний вызов пишется источником `mcp:` (провенанс), упавший деградирует в строку ошибки.

Жизненный цикл через `/v1/mcp_integration`: CRUD с фильтрами; новые интеграции стартуют `disabled`, стейт-машина `disabled/active/unreachable` разделяет намерение оператора и наблюдаемое здоровье (пробы переключают только `active <-> unreachable`). `POST /{id}/discover` кэширует снапшот схем тулов в БД - агент собирает тулы из замороженного кэша (ноль сети на старте рана, подмена описания на сервере не долетает до промпта молча). `POST /{id}/probe` - живой пинг, `GET /{id}/health` - дешёвое чтение состояния; health-джоба на каждый create/update уходит в io-лейн очереди и не ждёт GPU-джобы. `allowed_tools` - явный allowlist: discover показывает каталог, человек выбирает, что реально видит 8B-модель.

Auth интеграции описывается как `{"type": "bearer", "token_env": "HF_TOKEN"}` или header-вариант: в БД только ИМЕНА env-переменных, значения берутся из окружения и только для переменных из allowlist в `config.yaml` (`mcp_integrations.secret_env`).

Про границы доверия: auth нет намеренно; любой с доступом к API может зарегистрировать интеграцию на произвольный url, и разрешённые секреты уйдут туда. Не выставляйте сервис наружу.

Сиды (все disabled до явного включения): DeepWiki (без auth), Hugging Face (`HF_TOKEN`), Context7 (`CONTEXT7_API_KEY`).

## Как это устроено

- `app/config.py` - loader `config.yaml`.
- `app/orm/` - SQLAlchemy: `base` (declarative), `sync_db` (psycopg), `async_db` (asyncpg).
- `app/models/` - ORM-модели: `registry` (Model/ModelRole/Prompt), `eval` (Question/QuestionLog), `jobs` (Job), `corpus` (DataSource/DataChunk), `experiment` (Experiment + стейт-машина).
- `app/llm.py` - клиент Ollama через OpenAI SDK (генерация / эмбеддинги / structured output) + резолвер роль→модель.
- `app/rerank.py` - cross-encoder реранкер (sentence-transformers, CPU, ленивая загрузка).
- `app/job_queue.py`, `app/worker.py`, `app/job_handlers/` - очередь на Postgres (FOR UPDATE SKIP LOCKED) и воркер с ретраями/дефёром; хендлеры разнесены по тематике.
- `app/bootstrap.py` - idempotent-инициализация на старте.
- `app/sources/` - per-source ingestion (reader-паттерн: `Base` ABC + источники).
- `app/db.py` - гибридный поиск (сырой SQL: pgvector `<=>`, FTS, ltree, RRF).
- `app/use_cases/` - `chat` (retrieve/answer), `agent` (ReAct tool-calling цикл), `index` (сбор корпуса), `judge` (оценка ответа), `experiment` (агрегатор серии + RRF-композит).
- `app/agent_tools.py` - реестр тулов + `dispatch` + тул `search_corpus` поверх гибридного поиска.
- `app/mcp_server.py` - FastMCP-сервер (примонтирован на `/mcp`): тулы `search_corpus` / `answer_question` / `list_categories` поверх примитивов поиска.
- `app/mcp_ops.py` - ops MCP-сервер (примонтирован на `/mcp-ops`): `run_metrics` / `compare_runs` / `list_jobs` / `cancel_job` поверх eval-платформы.
- `app/api/` - REST-адаптеры (health + v1: chat / agent / categories / model / role / source / prompt / eval / experiment / questions / question-log / job).
- `app/seed.py`, `app/console.py` - сид промптов/банка вопросов; REPL-консоль.
- `app/evals/` - eval-стенд (runner + метрики retrieval + generation через judge).
- `tests/` - unit-тесты (чистая логика, без DB/Ollama): `docker compose exec rag-lab pytest -q`.

## Статус

Учебный проект: цель - освоить RAG/LLM-инженерию руками, из примитивов. RAG из примитивов (гибрид, ltree, FTS) + FastAPI-сервер + 5-осевой eval-стенд на LLM-judge + production-слой (pyproject/uv, централизованный config, SQLAlchemy ORM sync+async, OpenAI-совместимый клиент, role-keyed lifecycle моделей, версионирование промптов, async job-queue, банк вопросов, reranking, route-driven eval-платформа, MCP-сервер).
