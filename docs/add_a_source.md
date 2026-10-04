# Adding a source

A source is anything the corpus reads: a PDF book, a scan, a documentation site, a repository of markdown. This page walks one source from its declaration to questions about it, and marks at every step who acts:

- **person**: a decision the stand does not take for you;
- **agent**: reading and proposing through the MCP tools (an LLM client such as Claude Code); it never accepts a source on its own judgement;
- **stand**: code and config, through a job in the queue.

How a file is converted and how it checks itself is in [intake.md](intake.md); this page is the path around it.

![A source's life: declared, raw with its verdict, accepted with a candidate run beside it, indexed per variant, active](diagrams/source_life.svg)

## The doors

Every step has a REST route and an MCP tool of the `rag-lab-ops` server ([mcp.md](mcp.md)); the routes are listed in [api.md](api.md).

| step | REST | MCP | who |
|---|---|---|---|
| declare | `POST /v1/source` | `add_source` | person or agent |
| onboard | `POST /v1/source/{id}/onboard` | `onboard_source` | stand (job) |
| where every source stands, by who it waits for | | `intake_board` | agent reads |
| one source in a screen | `GET /v1/source/{id}` | `source_trail`, `source` | agent reads |
| try a knob on a few pages | job `probe_intake` | `probe_intake` | agent proposes |
| set a source's knobs | `PUT /v1/source/{id}/intake` | `set_source_intake` | person |
| set a declared field in place | `PUT /v1/source/{id}/skip_paths`, `/markup`, `/section_roots` | `set_source_fields` | person |
| accept | `POST /v1/source/{id}/accept` | `accept_source` | person, or stand for `ok` |
| index | job `index_data` | | stand |
| turn on for search | `PUT /v1/source/{id}` | `set_source_active` | person |
| remove | `DELETE /v1/source/{id}` | `remove_source` | person |

`intake_board` answers "what waits for whom": every source by stage and verdict, and the ones waiting for a person, for a door or for the queue, each with its next step. `source_trail` shows one source: its stage, verdict and reasons, the files it left out and why, chunks per variant, its last jobs and what waits next.

## 1. Declare (person or agent)

A declaration names the source, one origin and, optionally, its language, licence and trust:

```bash
curl -X POST localhost:8000/v1/source -H 'content-type: application/json' -d '{
  "name": "react-docs", "language": "en", "licence": "CC BY 4.0",
  "folder": "datasets/inbox/git/react-docs", "categories": ["react"],
  "skip_paths": ["src/content/blog/*"]}'
```

- Origins: `urls` (files to fetch), `folder` (already on disk), `git`, `git_family`, or a site's `pages` (with the CSS `main` that holds the text and `drop` for its furniture, or a `sitemap` with address patterns).
- `skip_paths` leaves whole paths under the root out, each named in the report; `skip` matches file stems only.
- `trust` (`official`, `book`, `notes`) decides which copy stays when two sources hold a text word for word. Left out, it is read from the origin: the book shelf is `book`, the interview banks are `notes`, the rest `official`.
- Fields that replace a source-specific reader: `tags`, `tag_from_name`, `tags_by_path` and `tags_from_frontmatter` (tags in place of the file's folders), `skip_when_frontmatter` (a page whose frontmatter key holds the value is not read), `section_root_from_filename` and `section_root_by_path` (where a page's heading path starts), `markup` (`hugo` or `mdn`, rendered before the cut) with `markup_values` (the site parameters it prints). `skip_paths`, `markup`, `markup_values` and `section_root_by_path` can be set on an existing row in place; the rest go through a new declaration or the source file.
- The door refuses a malformed declaration before any job: a missing folder, an unknown category, an origin it cannot read.

The row starts `declared` and inactive. Sources the stand ships with come from `sources/*.yaml` through the same model.

## 2. Onboard (stand)

One `onboard_source` job reads every file through the engine its kind picks: markdown as it is, a PDF with a text layer and HTML to Docling, a scan to MinerU. You never choose the engine. The output is a raw folder: one markdown per file, the pieces and a report. Nothing is indexed yet.

## 3. The verdict (stand)

The report carries a verdict, `ok`, `dirty` or `bad`, computed from the conversion's signals per piece and the chunker's gates per chapter (`intake.quality` in `config/intake.yaml`).

- `ok` is accepted by the stand when `intake.quality.auto_accept_ok` is on, marked `accepted_by: auto`.
- `dirty` and `bad` wait for a person.
- A source already accepted gets the new run as a candidate beside the one in use. An `ok` candidate replaces it at once, unless an `index_data` job for that source waits in the queue: then it stays a candidate until a person accepts it. Do not queue a re-onboard and an index of the same source in one batch.

## 4. When it comes out dirty or bad (agent reads, person decides)

The agent's order of reading:

1. `intake_board`: who waits for a person, with the bad share and the top reasons.
2. `source_trail <name>`: the reasons, the files left out, the last jobs.
3. The report next to the source's measurement file: `raw_source_<name>_<date>_sections.json.gz` lists every chapter with its words and the gates it breached. Group the breaching words by folder: in practice the breach sits in one place.
4. The markdown of the worst places, read in the raw folder.

Then one of four outcomes, each a person's decision:

| what the breach is | what to do | door |
|---|---|---|
| a part of the source that is not text worth retrieving (a generated API reference, release notes, a blog) | leave the paths out and onboard again | remove and declare with `skip_paths` |
| a conversion defect a knob fixes | try the knob on the pages first, then set it on this source | `probe_intake`, then `set_source_intake`, then `onboard_source` |
| the source's own shape (a wide table, a tutorial that repeats its code) | accept with the reason | `accept_source` with `reason` |
| not worth the corpus | remove | `remove_source` |

A knob set on one source stays that source's. Making it a stand default is a separate decision with its own measured run ([intake.md](intake.md), "How a default is chosen").

Worked example, the corpus build of 02.10.2026:

| source | verdict | where | decision |
|---|---|---|---|
| kubernetes-website | bad | the generated API reference: every resource repeats the same block of HTTP parameters | `skip_paths` on `content/en/docs/reference/kubernetes-api/*` |
| altinity-docs | dirty | release notes, whose headings are commit links longer than the text under them | `skip_paths` on `releasenotes/*` |
| react-docs | dirty | the blog, plus tutorials that show the same code step by step | `skip_paths` on the blog, the rest accepted with a reason |
| sre-workbook | dirty | one wide table whose header repeats in every chunk | accepted with a reason |

## 5. Accept (person, or stand for `ok`)

```bash
curl -X POST localhost:8000/v1/source/<id>/accept -H 'content-type: application/json' \
  -d '{"reason": "one wide table, the prose reads fine"}'
```

A `bad` verdict needs the reason, and any reason given is kept on the row. Accepting a candidate run drops the run it replaces; the variants cut from that run read as moved in the source's `drift` until it is indexed again.

## 6. Index (stand) and turn on (person)

`index_data` cuts the accepted sources into a variant (`corpus.variants` in `config.yaml`) and embeds them. A text another, more trusted source holds word for word is dropped from this one, and the job's result names how many chunks went (`lower_copies_dropped`). Which variant search reads is `corpus.variant`, a person's choice; the source answers only once `set_source_active` turns it on.

## 7. Questions (stand, with a person's set name)

A question set has two entrances:

- the stand writes pairs: `generate_questions` asks a cloud model for an English and a Russian question per section;
- pairs written outside (by an agent, by hand) are saved as `sources/questions/<set>.jsonl` and poured in by `load_questions`.

Either way the same checks follow: `accept_questions` has a reader answer each question from its gold section by a quote and settles what it can, `judge_questions` decides what the reader left open, and `anchor_questions` ties each row to the identifiers its section holds. How a set is built and read is in [question_sets.md](question_sets.md).

## 8. Measure

A closing number is preregistered first, the preflight runs, then `eval_run` and its judge. The method is in [measurement.md](measurement.md), the commands in [use_cases.md](use_cases.md).
