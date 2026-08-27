# Что смена разборщика делает с множеством разделов `clean_1024`

Сухая нарезка новым кодом против 12 108 строк, лежащих в базе (нарезаны регэкспом).

**Исчезает 16 разделов в 5 файлах, появляется 6.**

| файл | раздел, которого больше нет |
|---|---|
| `cheatsheets/circle.md` | CircleCI > Custom notifications |
| `cheatsheets/circle.md` | CircleCI > Customize checkout |
| `cheatsheets/circle.md` | CircleCI > Customize database setup |
| `cheatsheets/circle.md` | CircleCI > Customize dependencies |
| `cheatsheets/circle.md` | CircleCI > Customize deployment commands |
| `cheatsheets/circle.md` | CircleCI > Customize test commands |
| `cheatsheets/circle.md` | CircleCI > Customize the test machine |
| `cheatsheets/markdown.md` | Markdown > h2 |
| `cheatsheets/mysql.md` | MySQL > Switch back to the mysqld_safe terminal and kill the process using Control + \ |
| `cheatsheets/ronn.md` | Ronn > COPYRIGHT |
| `cheatsheets/ronn.md` | Ronn > DESCRIPTION |
| `cheatsheets/ronn.md` | Ronn > EXAMPLES |
| `cheatsheets/ronn.md` | Ronn > OPTIONS |
| `cheatsheets/ronn.md` | Ronn > SEE ALSO |
| `cheatsheets/ronn.md` | Ronn > SYNOPSIS |
| `cheatsheets/textile.md` | Textile > ordered list |

| файл | раздел, которого раньше не было |
|---|---|
| `notes/llm/agent_systems_market_map.md` | Рынок «построение агентных систем» — что хотят + roadmap под rag-lab |
| `notes/llm/agent_systems_market_map.md` | Рынок «построение агентных систем» — что хотят + roadmap под rag-lab > ROADMAP: что достроить в rag-lab / выучить, чтобы стать целевым |
| `notes/llm/agent_systems_market_map.md` | Рынок «построение агентных систем» — что хотят + roadmap под rag-lab > Кластер вакансий (тот же тип роли, ссылки) |
| `notes/llm/agent_systems_market_map.md` | Рынок «построение агентных систем» — что хотят + roadmap под rag-lab > Стратегический вывод |
| `notes/llm/agent_systems_market_map.md` | Рынок «построение агентных систем» — что хотят + roadmap под rag-lab > Что строят — ПО ПРОЕКТАМ |
| `notes/llm/agent_systems_market_map.md` | Рынок «построение агентных систем» — что хотят + roadmap под rag-lab > Что хотят — ПО ТЕХНОЛОГИЯМ (по частоте) |

## Сверка с `false_headings.md` по уровню `##`

Строк уровня `##`, которые разборщик не считает заголовком, всего 18, а разделов исчезает 16. Разница в тех, кто разделом никогда и не был: под ними нет текста, а раздел без текста старый код тоже выбрасывал, поэтому в базе его нет и исчезать нечему. Появившиеся разделы это `notes`, живой каталог владельца, он дрейфует.

| файл | строка | заголовок | причина | раздел исчез |
|---|---|---|---|---|
| `cheatsheets/circle.md` | 20 | Customize the test machine | внутри забора | да |
| `cheatsheets/circle.md` | 41 | Customize checkout | внутри забора | да |
| `cheatsheets/circle.md` | 47 | Customize dependencies | внутри забора | да |
| `cheatsheets/circle.md` | 64 | Customize database setup | внутри забора | да |
| `cheatsheets/circle.md` | 71 | Customize test commands | внутри забора | да |
| `cheatsheets/circle.md` | 81 | Customize deployment commands | внутри забора | да |
| `cheatsheets/circle.md` | 88 | Custom notifications | внутри забора | да |
| `cheatsheets/graphql.md` | 1 | Intro | разборщик не считает | нет |
| `cheatsheets/mysql.md` | 257 | Switch back to the mysqld_safe terminal and kill the process using Control + \ | внутри забора | да |
| `cheatsheets/ronn.md` | 29 | SYNOPSIS | внутри забора | да |
| `cheatsheets/ronn.md` | 33 | DESCRIPTION | внутри забора | да |
| `cheatsheets/ronn.md` | 38 | OPTIONS | внутри забора | да |
| `cheatsheets/ronn.md` | 46 | EXAMPLES | внутри забора | да |
| `cheatsheets/ronn.md` | 53 | COPYRIGHT | внутри забора | да |
| `cheatsheets/ronn.md` | 58 | SEE ALSO | внутри забора | да |
| `cheatsheets/markdown.md` | 8 | h2 | внутри забора | да |
| `cheatsheets/textile.md` | 46 | ordered list | внутри забора | да |
| `redis-doc/docs/data-types/bitfields.md` | 14 | Examples | разборщик не считает | нет |

## Сухая нарезка обоих вариантов новым кодом

```
clean_1024   size_cut 0.883
prefix_1024  size_cut 0.281
```

`size_cut` это доля кусков, границу которых поставил счётчик, а не структура автора.
