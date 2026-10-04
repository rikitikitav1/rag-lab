import config
import pytest
import query_translation


@pytest.fixture
def translation_on(monkeypatch):
    monkeypatch.setattr(config.settings.retrieval.keyword.translation, "enabled", True)
    monkeypatch.setattr(query_translation, "_translate", lambda text, model_dir: f"<{text}>")


def test_names_spelled_in_latin_are_kept_out_of_the_translation():
    question = "Как в PostgreSQL задать max_connections флагом --config или `SET x = 1`?"
    words, kept = query_translation.split_kept(question)
    assert kept == ["PostgreSQL", "max_connections", "--config", "SET x = 1"]
    assert "PostgreSQL" not in words and "config" not in words and words.startswith("Как в")


def test_a_russian_question_gets_its_words_in_english_and_its_names_as_they_are(translation_on):
    assert query_translation.keyword_translation("Как настроить pgbouncer пул?") == "<Как настроить пул?> pgbouncer"


def test_an_english_question_and_a_switched_off_translation_leave_the_search_alone(translation_on, monkeypatch):
    assert query_translation.keyword_translation("How to set max_connections?") is None
    monkeypatch.setattr(config.settings.retrieval.keyword.translation, "enabled", False)
    assert query_translation.keyword_translation("Как настроить пул?") is None


def test_a_translation_that_fails_leaves_the_question_unanswered_by_it_not_broken(translation_on, monkeypatch):
    def broken(text, model_dir):
        raise OSError("no model")

    monkeypatch.setattr(query_translation, "_translate", broken)
    assert query_translation.keyword_translation("Как настроить пул?") is None
