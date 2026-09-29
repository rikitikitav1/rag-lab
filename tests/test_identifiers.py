from corpus_keys import anchors, identifiers, spaceless_key


# an identifier by its shape or its backticks, never an ordinary word; the Russian half anchors on the same token
def test_identifiers_are_read_by_their_shape():
    assert identifiers("I'm using the `rank_names` list in the `Card` class.") == ["rank_names", "Card"]
    assert identifiers("Why does copy.copy not clone nested lists?") == ["copy.copy"]
    assert identifiers("How do I pass --prefix to configure?") == ["--prefix"]
    assert identifiers("What does TypeError mean when calling len()?") == ["TypeError", "len()"]
    assert identifiers("Как работает proxy_cache_valid в nginx?") == ["proxy_cache_valid"]
    assert identifiers("How do I open a list or an int file, and is it pre-set?") == []


# an anchor is an identifier its gold section holds, counted over the source's sections by the one quote key
def test_anchors_count_the_sections_that_hold_them():
    sections = ["Sets `proxy_cache_valid` for 5m.", "The proxy_cache_valid directive again.", "Nothing here."]
    keys = [spaceless_key(s) for s in sections]
    assert anchors("How long does proxy_cache_valid keep it?", keys[0], keys) == {"proxy_cache_valid": 2}
    assert anchors("What does zlib_level do?", keys[0], keys) == {}


# a written pair carries the anchors of each half; the stratum is read from them at the declared threshold
def test_rows_carry_anchors_and_the_stratum_is_read_at_its_threshold():
    from types import SimpleNamespace

    from evals import columns, section_questions

    sections = [{"file": "n/p.md", "section": "Proxy", "text": "Sets `proxy_cache_valid` for 5m."},
                {"file": "n/c.md", "section": "Core", "text": "Unrelated text."}]
    pair = {"en": "How long does proxy_cache_valid keep it?", "ru": "Сколько держит proxy_cache_valid?",
            "answer": "5m", "evidence": "Sets proxy_cache_valid for 5m."}
    rows = section_questions.question_rows(pair, sections[0], "s", section_questions.anchors_for(sections)(sections[0]))
    assert [r["anchors"] for r in rows] == [{"proxy_cache_valid": 1}, {"proxy_cache_valid": 1}]

    def read(anchors):
        return columns.read("anchored_by_identifier", SimpleNamespace(question=SimpleNamespace(anchors=anchors)))

    assert read({"proxy_cache_valid": 1}) == 1.0 and read({"list": 40}) == 0.0 and read(None) is None


# the backfill finds each row's section by its gold and writes the rule's anchors; a gone section is counted
def test_a_set_is_anchored_by_its_gold_sections(monkeypatch):
    from types import SimpleNamespace

    from job_handlers import questions

    sections = [{"file": "n/p.md", "section": "Proxy", "text": "Sets `proxy_cache_valid` for 5m."}]
    rows = [SimpleNamespace(gold={"file": "n/p.md", "section": "Proxy"}, original_text="What is proxy_cache_valid?",
                            anchors=None),
            SimpleNamespace(gold={"file": "n/gone.md", "section": "Gone"}, original_text="x_y?", anchors=None)]

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scalars(self, stmt):
            return rows

        def commit(self):
            pass

    monkeypatch.setattr(questions.section_export, "of_source", lambda name: {"sections": sections})
    monkeypatch.setattr(questions, "Session", FakeSession)
    out = questions.anchor_questions({"source": "nginx-org-en", "set_name": "s"})
    assert rows[0].anchors == {"proxy_cache_valid": 1} and out["anchored"] == 1 and out["section_gone"] == 1


# a question repeating a heading word is its own stratum; a five-letter stem meets Russian endings
def test_a_question_that_shares_a_heading_word_is_told_apart():
    from corpus_keys import shares_heading_word

    assert shares_heading_word("How do I set the proxy buffering size?", "Module > Proxy buffering")
    assert shares_heading_word("Как работает анализ тональности текста?", "Книга > Анализ тональности")
    assert not shares_heading_word("How do I set the size?", "Module > Proxy buffering")
    assert not shares_heading_word("Is it on?", "Module > It is on")


# a reference page is named by a code shape in its leaf, or by its source's own pattern; a product name is neither
def test_a_reference_page_is_read_from_its_leaf():
    from corpus_keys import reference_by

    assert reference_by("Functions > String functions > NTH()") == "heading"
    assert reference_by("Module ngx_http_proxy_module > proxy_pass") == "heading"
    assert reference_by("Commands > EXPIRE") is None
    assert reference_by("Commands > EXPIRE", r"^[A-Z][A-Z0-9_]+( [A-Z][A-Z0-9_]+)?$") == "knob"
    plain = ("Guide > WebSocket proxying", "Book > FAQ", "Анализ > ВЫВОД", "Lists")
    assert [reference_by(s) for s in plain] == [None] * 4
