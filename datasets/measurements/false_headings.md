# Строки, которые старый регэксп считал заголовком, а разборщик не считает

Всего 23, по уровням {'##': 18, '###': 5}, файлов 11.

Две разные причины, и путать их не надо:

- **внутри забора** (17): строка стоит в блоке кода, и заголовком она никогда не была. Это то, ради чего брали разборщик;
- **разборщик не считает заголовком** (6): строка не внутри забора с тройной кавычкой, и почему именно разборщик её отверг, из этих данных не видно. Проверенные случаи: блок кода с отступом в четыре пробела (это тоже код по markdown, и наш счётчик заборов его не видит) и заголовок, под которым ничего не написано. Причину каждой строки надо смотреть глазами, а не подписывать одним словом: первая версия этого файла подписала все строки заборами и была неправа.

Попали бы в набор вопросов (`^## \d+\.`): **0**.

| файл | строка | уровень | текст | причина |
|---|---|---|---|---|
| `cheatsheets/circle.md` | 20 | `##` | ## Customize the test machine | внутри забора |
| `cheatsheets/circle.md` | 41 | `##` | ## Customize checkout | внутри забора |
| `cheatsheets/circle.md` | 47 | `##` | ## Customize dependencies | внутри забора |
| `cheatsheets/circle.md` | 64 | `##` | ## Customize database setup | внутри забора |
| `cheatsheets/circle.md` | 71 | `##` | ## Customize test commands | внутри забора |
| `cheatsheets/circle.md` | 81 | `##` | ## Customize deployment commands | внутри забора |
| `cheatsheets/circle.md` | 88 | `##` | ## Custom notifications | внутри забора |
| `cheatsheets/graphql.md` | 1 | `##` | ## Intro | разборщик не считает |
| `cheatsheets/mysql.md` | 257 | `##` | ## Switch back to the mysqld_safe terminal and kill the process using  | внутри забора |
| `cheatsheets/ronn.md` | 29 | `##` | ## SYNOPSIS | внутри забора |
| `cheatsheets/ronn.md` | 33 | `##` | ## DESCRIPTION | внутри забора |
| `cheatsheets/ronn.md` | 38 | `##` | ## OPTIONS | внутри забора |
| `cheatsheets/ronn.md` | 46 | `##` | ## EXAMPLES | внутри забора |
| `cheatsheets/ronn.md` | 53 | `##` | ## COPYRIGHT | внутри забора |
| `cheatsheets/ronn.md` | 58 | `##` | ## SEE ALSO | внутри забора |
| `cheatsheets/markdown.md` | 8 | `##` | ## h2 | внутри забора |
| `cheatsheets/markdown.md` | 9 | `###` | ### h3 | внутри забора |
| `cheatsheets/textile.md` | 46 | `##` | ## ordered list | внутри забора |
| `cheatsheets/firefox.md` | 106 | `###` | ### Firefox 8 (Nov 2011) | разборщик не считает |
| `redis-doc/docs/data-types/bitfields.md` | 14 | `##` | ## Examples | разборщик не считает |
| `redis-doc/docs/connect/clients/java/jedis.md` | 180 | `###` | ### Production usage | разборщик не считает |
| `redis-doc/docs/management/optimization/cpu-profiling.md` | 161 | `###` | ### Visualizing the recorded profile information using Flame Graphs | разборщик не считает |
| `devops-interview-questions/README.md` | 542 | `###` | ### Continuous Delivery / Deployment (CD) Tools | разборщик не считает |
