import pytest
from evals import build_paraphrased


class _Q:
    def __init__(self, qid):
        self.id = qid
        self.original_text = f"q{qid}"
        self.marked_sources = ["repo/README.md"]


class _Session:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def _rows(self):
        order = self.store.get("answers")
        if order is None:
            rows = self.store["originals"] if self.store["asked"] == 0 else self.store["done"]
        else:
            rows = order[min(self.store["asked"], len(order) - 1)]
        self.store["asked"] += 1
        return rows

    def scalars(self, stmt):
        rows = self._rows()

        class _R:
            @staticmethod
            def all():
                return rows

        return _R()

    # the done-sets come back as (original id, stored text) so a resumed run can derive
    # the missing half from the half that is already there
    def execute(self, stmt):
        rows = [(qid, f"p:q{qid}") for qid in self._rows()]

        class _R:
            @staticmethod
            def all():
                return rows

        return _R()

    def commit(self):
        pass


def _wire(monkeypatch, store, made):
    monkeypatch.setattr(build_paraphrased, "Session", lambda: _Session(store))
    monkeypatch.setattr(
        build_paraphrased, "_pick", lambda *a, **kw: store.setdefault("picked", []).append(1) or []
    )
    monkeypatch.setattr(build_paraphrased, "_paraphrase", lambda text: f"p:{text}")
    monkeypatch.setattr(build_paraphrased, "_translate_ru", lambda text: f"ru:{text}")
    monkeypatch.setattr(
        build_paraphrased,
        "_insert",
        lambda session, text, set_name, lang, original: made.append(original.id) or True,
    )


def test_a_fixed_list_of_originals_survives_a_restart(monkeypatch):
    store = {"originals": [_Q(1), _Q(2)], "done": [], "asked": 1}
    _wire(monkeypatch, store, [])
    build_paraphrased.build(None, set_name="s", seed="x", originals=[1, 2])
    assert "picked" not in store, "originals were given, nothing may be picked again"


def test_without_a_list_the_run_picks_for_itself(monkeypatch):
    store = {"originals": [], "done": [], "asked": 1}
    _wire(monkeypatch, store, [])
    build_paraphrased.build(None, set_name="s", seed="x")
    assert store["picked"] == [1]


def test_a_restart_finishes_the_list_instead_of_repeating_it(monkeypatch):
    # the first attempt got through question 1 before the worker took the job back
    made = []
    # the three queries build makes, in order: the originals, done in en, done in ru
    store = {"answers": [[_Q(1), _Q(2), _Q(3)], [1], [1]], "asked": 0}
    _wire(monkeypatch, store, made)
    build_paraphrased.build(None, set_name="s", seed="x", originals=[1, 2, 3])
    assert sorted(set(made)) == [2, 3], "question 1 was already done and must not be redone"


def test_the_missing_half_is_derived_from_the_stored_one(monkeypatch):
    # a fresh paraphrase would be a different question, and the pair exists to be one
    # question in two languages
    texts = []
    store = {"answers": [[_Q(1)], [1], []], "asked": 0}
    monkeypatch.setattr(build_paraphrased, "Session", lambda: _Session(store))
    monkeypatch.setattr(build_paraphrased, "_pick", lambda *a, **kw: [])
    monkeypatch.setattr(
        build_paraphrased, "_paraphrase", lambda text: pytest.fail("re-paraphrased a stored half")
    )
    monkeypatch.setattr(build_paraphrased, "_translate_ru", lambda text: f"ru:{text}")
    monkeypatch.setattr(
        build_paraphrased,
        "_insert",
        lambda session, text, set_name, lang, original: texts.append(text) or True,
    )
    build_paraphrased.build(None, set_name="s", seed="x", originals=[1])
    assert texts == ["ru:p:q1"]


def test_an_original_with_only_a_russian_half_is_left_alone(monkeypatch):
    # the english half cannot be recovered from it, and inventing one writes a pair that
    # is two different questions
    inserted = []
    store = {"answers": [[_Q(1)], [], [1]], "asked": 0}
    monkeypatch.setattr(build_paraphrased, "Session", lambda: _Session(store))
    monkeypatch.setattr(build_paraphrased, "_pick", lambda *a, **kw: [])
    monkeypatch.setattr(build_paraphrased, "_paraphrase", lambda text: f"p:{text}")
    monkeypatch.setattr(build_paraphrased, "_translate_ru", lambda text: f"ru:{text}")
    monkeypatch.setattr(
        build_paraphrased,
        "_insert",
        lambda session, text, set_name, lang, original: inserted.append(text) or True,
    )
    build_paraphrased.build(None, set_name="s", seed="x", originals=[1])
    assert inserted == []


def test_a_run_that_died_between_the_pair_adds_only_the_missing_half(monkeypatch):
    # one original makes two rows: the paraphrase, then the translation. Counting "done"
    # across both sets would drop question 1 entirely and leave it without a translation
    inserted = []
    store = {"answers": [[_Q(1), _Q(2)], [1], []], "asked": 0}
    monkeypatch.setattr(build_paraphrased, "Session", lambda: _Session(store))
    monkeypatch.setattr(build_paraphrased, "_pick", lambda *a, **kw: [])
    monkeypatch.setattr(build_paraphrased, "_paraphrase", lambda text: f"p:{text}")
    monkeypatch.setattr(build_paraphrased, "_translate_ru", lambda text: f"ru:{text}")
    monkeypatch.setattr(
        build_paraphrased,
        "_insert",
        lambda session, text, set_name, lang, original: inserted.append(
            (original.id, set_name)
        )
        or True,
    )
    build_paraphrased.build(None, set_name="s", seed="x", originals=[1, 2])
    assert (1, "s") not in inserted, "the english half of question 1 was already there"
    assert (1, "s_ru") in inserted, "its translation was not, and must be added"
    assert sorted(inserted) == [(1, "s_ru"), (2, "s"), (2, "s_ru")]
