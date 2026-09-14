import outcomes

# every string here is a real answer from the crag3 runs, trimmed
REFUSALS = (
    "I'm sorry, but I couldn't find any information on why pasta water needs salt. "
    "A cookbook or culinary expert might have the answer to this question.",
    "Извлечение информации о заживлении растяжения связок голеностопа не удалось из "
    "доступных источников. Нужны медицинские справочники.",
    "I cannot answer this from the available sources.",
    outcomes.NO_RESULTS,
)

ANSWERS = (
    "A catalyst speeds up a chemical reaction without being consumed, by lowering the "
    "activation energy required for the reaction to occur.",
    "If the key is not found, the lookup returns nil rather than raising.",
    "Индекс отсутствует, поэтому планировщик выбирает seq scan по всей таблице.",
    "The corpus is sharded by source, and every shard has its own index.",
)


def test_the_phrases_the_model_actually_refuses_with_are_recognised():
    for text in REFUSALS:
        assert outcomes.refusal(text), text[:60]


def test_technical_prose_is_not_a_refusal():
    for text in ANSWERS:
        assert not outcomes.refusal(text), text[:60]


def test_a_weak_phrase_needs_the_sources_to_be_blamed():
    assert not outcomes.refusal("The row is not found and the cache stays cold.")
    assert outcomes.refusal("The answer is not found in the available sources.")


def test_sources_that_do_not_contain_the_answer_refuse():
    for text in (
        "Контекст не содержит информации об ионной и ковалентной связи. В предоставленных материалах "
        "рассматриваются только вопросы по Ionic и CNN, и они не касаются химических типов связи.",
        "Предоставленные фрагменты не содержат ответа на этот вопрос.",
        "The provided context does not contain information about Delphi 7.",
    ):
        assert outcomes.refusal(text), text[:60]


def test_a_thing_that_does_not_contain_something_is_prose():
    for text in (
        "Индекс не содержит NULL, поэтому документ попадает в выборку целиком.",
        "The array does not contain duplicates, so the source order is kept.",
    ):
        assert not outcomes.refusal(text), text[:60]


def test_a_long_essay_is_never_a_refusal():
    essay = "Replication is asynchronous, so a failover can lose the tail. " * 12
    assert len(essay) > outcomes.REFUSAL_MAX_CHARS
    assert not outcomes.refusal(essay + " no information in sources")


def test_a_narrated_call_wins_over_a_refusal_phrase():
    text = '{"name": "deepwiki__ask_question", "parameters": {"q": "x"}} no information found'
    assert outcomes.classify(text, False) == outcomes.Outcome.narrated_call
