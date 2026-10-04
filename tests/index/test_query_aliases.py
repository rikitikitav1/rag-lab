import config
import pytest
import query_aliases


@pytest.fixture
def aliases(monkeypatch):
    monkeypatch.setattr(config.settings.retrieval.keyword.aliases, "enabled", True)
    entries = {"postgresql": {"canonical": "PostgreSQL", "aliases": ["постгрес", "pg", "postgresql"]},
               "kubernetes": {"canonical": "Kubernetes", "aliases": ["k8s", "кубер", "куб"], "unsure": ["куб"]},
               "mssql": {"canonical": "Microsoft SQL Server", "aliases": ["ms sql"]}}
    monkeypatch.setattr(config.settings, "aliases", {k: config.AliasCfg(**v) for k, v in entries.items()})


def test_a_short_or_two_word_russian_alias_declines_too(aliases, monkeypatch):
    entries = {"git": {"canonical": "Git", "aliases": ["гит"]}, "node": {"canonical": "Node.js", "aliases": ["нод жс"]}}
    monkeypatch.setattr(config.settings, "aliases", {k: config.AliasCfg(**v) for k, v in entries.items()})
    assert query_aliases.reword("Ветки в гите и в нод жсе")[1] == ["гите=Git", "нод жсе=Node.js"]
    assert query_aliases.reword("Гитара и гитлер")[1] == []


def test_a_russian_alias_is_found_under_its_case_ending(aliases):
    assert query_aliases.reword("Как в постгресе включить vacuum?") == (
        "Как в PostgreSQL включить vacuum?", ["постгресе=PostgreSQL"])


def test_a_short_alias_matches_only_as_a_whole_word(aliases):
    assert query_aliases.reword("Что делает pg_dump и pgbouncer?")[1] == []
    assert query_aliases.reword("Бэкап pg через k8s")[0] == "Бэкап PostgreSQL через Kubernetes"


def test_an_unsure_alias_and_the_name_itself_leave_the_question_as_it_is(aliases):
    assert query_aliases.reword("Куб на столе и PostgreSQL") == ("Куб на столе и PostgreSQL", [])


def test_an_alias_of_two_words_is_replaced_whole(aliases):
    reworded = query_aliases.reword("Индексы в MS SQL")
    assert reworded == ("Индексы в Microsoft SQL Server", ["ms sql=Microsoft SQL Server"])


def test_the_switch_off_leaves_every_question_alone(aliases, monkeypatch):
    monkeypatch.setattr(config.settings.retrieval.keyword.aliases, "enabled", False)
    assert query_aliases.reword("Бэкап pg через k8s") == ("Бэкап pg через k8s", [])


def test_one_alias_naming_two_technologies_refuses_to_load():
    with pytest.raises(ValueError, match="name several"):
        config.AppConfig._one_owner_per_alias({"a": config.AliasCfg(canonical="A", aliases=["x"]),
                                               "b": config.AliasCfg(canonical="B", aliases=["x"])})


def test_the_stand_dictionary_loads_with_numbers_as_words():
    entries = config.settings.aliases
    assert entries["postgresql"].canonical == "PostgreSQL" and "404" in entries["http"].aliases
