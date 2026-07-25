GVL это что? | notes/languages/ruby/concurrency-gvl.md
Локи в PG? | notes/databases/postgresql/locks.md
Разница между руби  2 и 3 | notes/languages/ruby/ruby-2-vs-3.md
разница между кафкой и кроликом? | notes/messaging/comparison.md, notes/messaging/kafka.md, notes/messaging/rabbitmq.md
топ 3 фишки PG? | notes/databases/postgresql/pg-tips.md
кеширование в PG? | notes/databases/postgresql/pg-tips.md
расскажи о кликхаусе | developer-roadmap/src/data/roadmaps/backend/content/clickhouse

# --- in-corpus, PG (должны отвечаться по заметкам) ---
как работает MVCC в постгресе? | notes/databases/postgresql/mvcc-vacuum.md, developer-roadmap/src/data/roadmaps/postgresql-dba/content/mvcc
зачем нужен vacuum? | notes/databases/postgresql/mvcc-vacuum.md, developer-roadmap/src/data/roadmaps/postgresql-dba/content/vacuums, developer-roadmap/src/data/roadmaps/postgresql-dba/content/vacuum-processing
объясни оконные функции в sql | notes/databases/postgresql/window-functions.md, developer-roadmap/src/data/roadmaps/bi-analyst/content/window-functions
что такое CTE и зачем? | notes/databases/postgresql/cte.md, developer-roadmap/src/data/roadmaps/postgresql-dba/content/cte, developer-roadmap/src/data/roadmaps/sql/content/common-table-expressions
уровни изоляции транзакций в pg | notes/databases/postgresql/isolation-levels.md, developer-roadmap/src/data/roadmaps/sql/content/transaction-isolation-levels
как накатить миграцию без блокировки таблицы? | notes/databases/postgresql/migrations-without-locks.md
как устроен полнотекстовый поиск в postgres? | notes/databases/postgresql/full-text-search.md
зачем нужен ltree? | notes/databases/postgresql/ltree.md

# --- in-corpus, Ruby ---
метапрограммирование в ruby на пальцах | notes/languages/ruby/metaprogramming.md
чем proc отличается от lambda? | notes/languages/ruby/blocks-procs-lambdas.md

# --- in-corpus, messaging / protocols ---
чем nats отличается от кафки? | notes/messaging/nats.md, notes/messaging/comparison.md
что делает sidekiq? | notes/messaging/sidekiq.md
grpc против rest, в чём разница | notes/protocols/grpc.md, developer-roadmap/src/data/roadmaps/api-design/content/grpc-apis, developer-roadmap/src/data/roadmaps/server-side-game-developer/content/rpc--rest
зачем нужен protobuf? | notes/protocols/protobuf.md, developer-roadmap/src/data/roadmaps/cpp/content/protobuf, developer-roadmap/src/data/roadmaps/server-side-game-developer/content/protobuf
websocket или http, когда что? | notes/protocols/websockets.md, notes/protocols/http.md, developer-roadmap/src/data/roadmaps/flutter/content/web-sockets, developer-roadmap/src/data/roadmaps/aspnet-core/content/web-sockets

# --- in-corpus, твой LLM-домен (заметки про RAG) ---
что такое RAG простыми словами? | notes/llm/rag-concepts.md, developer-roadmap/src/data/roadmaps/ai-agents/content/understand-the-basics-of-rag, developer-roadmap/src/data/roadmaps/ai-agents/content/rag-and-vector-databases, developer-roadmap/src/data/roadmaps/forward-deployed-engineer/content/rags
как работает гибридный поиск и RRF? | notes/llm/hybrid-retrieval.md

# --- cross-source (cheatsheets/roadmap/sdp, местами тонко) ---
как задизайнить твиттер по системному дизайну? | system-design-primer/solutions/system_design/twitter/README.md
как масштабировать сервис под миллионы пользователей? | system-design-primer/README.md, system-design-primer/solutions/system_design/scaling_aws/README.md
базовые команды docker | cheatsheets/docker.md, developer-roadmap/src/data/roadmaps/docker/content/basics-of-docker
что такое pod в kubernetes? | developer-roadmap/src/data/roadmaps/kubernetes/content/pods
как работают хуки в react? | developer-roadmap/src/data/roadmaps/react/content/hooks, cheatsheets/react.md

# --- его домен: заметки про собесы ---
как честно говорить про свои факапы на собеседовании? | notes/interview/honesty-line.md

# --- out-of-corpus: должен ЧЕСТНО отказаться (faithfulness) ---
как приготовить карбонару? | NONE
что такое язык zig? | NONE
объясни квантовую запутанность | NONE
