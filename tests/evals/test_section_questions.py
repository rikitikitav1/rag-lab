import json

from evals import section_questions as sq

ROW = {
    "file": "redis/commands/expire.md",
    "section": "EXPIRE > Options",
    "chapter": "EXPIRE > Options",
    "versions": ["7.4", "7.2"],
    "words": 120,
    "text": (
        "The NX option sets the expiry only when the key has none.\n"
        "The GT option sets it only when the new expiry is greater."
    ),
}


def _reply(*pairs):
    return "Here you go:\n```json\n" + json.dumps({"pairs": list(pairs)}) + "\n```"


def _pair(**over):
    base = {
        "en": "How do I set a key's timeout only if it has no timeout yet?",
        "ru": "Как задать срок жизни ключа, только если его ещё нет?",
        "answer": "Use the NX option.",
        "evidence": "The NX option sets the expiry  only when the key has none.",
    }
    return {**base, **over}


# a pair the section stands behind is kept; each refusal names its reason
def test_a_reply_is_kept_only_where_the_section_stands_behind_it():
    reply = _reply(
        _pair(),
        _pair(en="What does GT do when the new expiry is smaller?", evidence="GT is great"),
        _pair(en="Options", ru="Как работает GT?", evidence="The GT option sets it only when"),
        _pair(en="Когда срабатывает GT?", ru="Когда срабатывает GT?", evidence="The GT option"),
        {"en": "no answer"},
    )
    kept, refused = sq.parse(reply, ROW, wanted=5)

    assert [p["en"] for p in kept] == [_pair()["en"]]
    assert [r["why"] for r in refused] == [
        "the evidence is not the section's own words",
        "a question repeats the heading",
        "a question is not in the language of its place",
        "a field of en, ru, answer, evidence is missing or empty",
    ]


def test_a_reply_without_json_or_short_of_pairs_says_so():
    assert sq.parse("sorry", ROW, 2) == ([], [{"why": "the reply holds no pairs as JSON", "count": 2}])
    kept, refused = sq.parse(_reply(_pair()), ROW, 2)
    assert len(kept) == 1 and refused == [{"why": "fewer pairs than asked", "count": 1}]
    _, refused = sq.parse(_reply(_pair(), _pair(en="Another one about GT?")), ROW, 1)
    assert refused == [{"why": "pairs past the number asked", "count": 1}]


# the two languages of a fact are two rows with one pair id and the section's exact gold, newest version first
def test_a_kept_pair_is_two_rows_of_one_pair_and_one_gold():
    rows = sq.question_rows(_pair(), ROW, "smoke")

    assert [r["language"] for r in rows] == ["en", "ru"]
    assert rows[0]["pair_id"] == rows[1]["pair_id"]
    assert rows[0]["gold"] == {"file": "redis/commands/expire.md", "section": "EXPIRE > Options", "version": "7.4"}
    assert rows[0]["reference_answer"] == "Use the NX option." and rows[1]["kind"] == "in_corpus"


# a chapter's pairs go to its sections by words; a stub section gets none
def test_a_chapters_pairs_go_to_its_sections_by_words_and_a_stub_gets_none():
    rows = [
        {"file": "a.md", "section": "A > x", "chapter": "A > x", "words": 900},
        {"file": "a.md", "section": "A > y", "chapter": "A > y", "words": 300},
        {"file": "a.md", "section": "A > z", "chapter": "A > z", "words": 20},
    ]
    assert sq.section_pairs(rows, {"A > x": 3, "A > y": 1, "A > z": 2}) == {
        ("a.md", "A > x", None): 3, ("a.md", "A > y", None): 1}


# one path in two version streams is two sections, and a question's gold names the stream it was written from
def test_a_versioned_source_keeps_each_stream_of_a_section_apart():
    rows = [{"file": "a.md", "section": "A > x", "stream": v, "chapter": "A > x", "words": 500, "versions": [v],
             "text": "t"} for v in ("3.12", "3.11")]
    assert sq.section_pairs(rows, {"A > x": 2}) == {("a.md", "A > x", "3.12"): 1, ("a.md", "A > x", "3.11"): 1}
    pair = {"en": "q?", "ru": "в?", "answer": "a", "evidence": "t"}
    gold = sq.question_rows(pair, rows[1], "smoke")[0]["gold"]
    assert sq.gold_key(gold) == sq.section_key(rows[1]) == ("a.md", "A > x", "3.11")


# a section's pairs go out two a call, and a later call is told what the earlier kept
def test_a_sections_ask_is_split_and_a_later_call_hears_the_earlier():
    assert sq.asks(5) == [2, 2, 1] and sq.asks(1) == [1] and sq.asks(0) == []
    assert "Already asked" not in sq.user_turn(ROW)
    assert sq.user_turn(ROW, ["How do I keep the timeout?"]).endswith("other facts:\n- How do I keep the timeout?")


# a quote without the section's emphasis and backticks is still its words; other words are not
def test_evidence_is_read_past_the_section_s_markup():
    row = {**ROW, "text": "Setting the *`size`* to 0 disables checking of the client_max_body_size."}
    kept, refused = sq.parse(_reply(
        _pair(evidence="Setting the size to 0 disables checking of the client_max_body_size."),
        _pair(en="And what does 1 do?", evidence="Setting the size to 1 disables checking"),
    ), row, 2)
    assert [p["evidence"] for p in kept] == ["Setting the size to 0 disables checking of the client_max_body_size."]
    assert [r["why"] for r in refused] == ["the evidence is not the section's own words"]


# a link is quoted by its words; a quote that joins two places with an ellipsis is still not the section's
def test_a_link_reads_as_its_words_and_an_ellipsis_stays_refused():
    row = {**ROW, "text": "Also enable [directio](#directio), or reading will block. Zero turns the limit off."}
    kept, refused = sq.parse(_reply(
        _pair(evidence="Also enable directio, or reading will block."),
        _pair(en="And zero?", evidence="Also enable directio, ... Zero turns the limit off."),
    ), row, 2)
    assert [p["evidence"] for p in kept] == ["Also enable directio, or reading will block."]
    assert [r["why"] for r in refused] == ["the evidence is not the section's own words"]


# a second pair on a quote the section already gave is the same fact asked again, in this reply or an earlier one
def test_a_pair_on_a_quote_already_taken_is_refused():
    first = _pair()
    again = _pair(en="How do I set a timeout only when the key has none?", ru="Как задать срок, если его нет?")
    kept, refused = sq.parse(_reply(first, again), ROW, 2)
    assert len(kept) == 1 and [r["why"] for r in refused] == ["the evidence repeats another pair of the section"]
    kept, refused = sq.parse(_reply(again), ROW, 1, taken=[first["evidence"]])
    assert kept == [] and [r["why"] for r in refused] == ["the evidence repeats another pair of the section"]


# a quote across a directive's table, an escaped bar and list dashes is still the section's words in order
def test_evidence_is_read_through_tables_escapes_and_list_dashes():
    row = {**ROW, "text": (
        "| Syntax: | **`keepalive_requests`** `number` ; |\n|---|---|\n| Default: | `keepalive_requests 1000;` |\n"
        "- `on \\| off` - turns it on\n- --prefix= *path* - defines a directory\n"
    )}
    kept, refused = sq.parse(_reply(
        _pair(evidence="keepalive_requests number ; Default: keepalive_requests 1000;"),
        _pair(en="Which values?", ru="Какие значения?", evidence="on | off turns it on"),
        _pair(en="Where does it go?", ru="Куда он ставит?", evidence="--prefix=path defines a directory"),
    ), row, 3)
    assert len(kept) == 3 and refused == []


# a section's pairs go to its blocks by length, the longest taking the remainder
def test_pairs_are_spread_over_blocks_by_length():
    assert sq.pairs_by_block(3, ["x" * 100, "x" * 300, "x" * 100]) == [1, 2, 0]
    assert sq.pairs_by_block(2, ["x" * 100, "x" * 300]) == [0, 2]
    assert sq.pairs_by_block(1, ["x" * 10]) == [1]


# a question that leans on its page is refused in either language; one that names the technology is kept
def test_a_question_leaning_on_its_page_is_refused():
    kept, refused = sq.parse(_reply(
        _pair(en="In the provided dataset, how is the expiry set?", ru="Как задать срок ключа?"),
        _pair(en="How do I set a timeout only when none exists?", ru="Как в данном примере задать срок?",
              evidence="The GT option sets it only when the new expiry is greater."),
    ), ROW, 2)
    assert kept == [] and {r["why"] for r in refused} == {"a question leans on the page it was written from"}
    said = sq._LEANS_ON_THE_PAGE.search
    assert said("Which model is mentioned as being able to accept raw CSV files?")
    assert said("What is the purpose of the code in Listing 4.1?") and said("Что делает код в листинге 4.1?")
    assert not said("How do I list tables in PostgreSQL 17?")
    assert not said("Which responses count as failed, whether or not they are listed in the directive?")


# a set asked in one language keeps one question a pair; the reply's keys and the rows follow the set's languages
def test_a_set_of_one_language_asks_and_writes_that_language_alone():
    reply = json.dumps({"pairs": [{"ru": "Как задать срок, только если его ещё нет?", "answer": "NX",
                                   "evidence": ROW["text"].split(".")[0] + "."}]})
    kept, refused = sq.parse(reply, ROW, 1, languages=("ru",))
    assert refused == [] and set(kept[0]) == {"ru", "answer", "evidence"}
    rows = sq.question_rows(kept[0], ROW, "smoke", languages=("ru",))
    assert [r["language"] for r in rows] == ["ru"]
    assert '"ru": "..."' in sq.prompt("{languages} {keys}", "redis-doc", 1, ("ru",))
    _, refused = sq.parse(reply, ROW, 1)
    assert refused[0]["why"].startswith("a field of en, ru")
