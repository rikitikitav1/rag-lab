from functools import lru_cache

import config
import logging_setup
from corpus_keys import language_by_alphabet
from langdetect import DetectorFactory, LangDetectException, detect
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

DetectorFactory.seed = 0
log = logging_setup.get_logger(__name__)


# every config knows its own function words, so ask them instead of guessing the language
FUNCTION_WORDS = """
SELECT cfg, coalesce(array_length(tsvector_to_array(to_tsvector(cfg::regconfig, :q)), 1), 0) AS kept
FROM unnest(CAST(:configs AS text[])) cfg
ORDER BY kept, cfg
"""


def _code_of(cfg: str, fts) -> str:
    return next((code for code, name in fts.languages.items() if name == cfg), "en")


def _by_alphabet(text_, fts) -> str:
    return language_by_alphabet(text_)


# the config that drops the most tokens recognised them as its own stopwords
def _by_function_words(text_, fts) -> str | None:
    configs = sorted(set(fts.languages.values()) | {fts.fallback})
    if len(configs) < 2:
        return None
    from orm.sync_db import engine

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(FUNCTION_WORDS), {"q": text_, "configs": configs}).all()
    except SQLAlchemyError as e:
        # picking a language must not need a database: the alphabet rule answers on its own
        log.warning("db.function_words_unavailable", error=str(e))
        return None
    # a tie means no function word of any candidate showed up: this rule has nothing to say
    return None if rows[0][1] == rows[1][1] else _code_of(rows[0][0], fts)


# one rule for the search config and for the language the answer comes back in
def detect_language(text_, mode=None) -> str:
    return _detect(text_, mode or config.settings.retrieval.keyword.query_lang)


# a hop asks for the same question several times, and function_words costs a round trip
@lru_cache(maxsize=4096)
def _detect(text_: str, mode: str) -> str:
    fts = config.settings.fts
    if mode == "function_words":
        return _by_function_words(text_, fts) or _by_alphabet(text_, fts)
    if mode == "cyrillic_ratio":
        # langdetect misreads short mixed-script questions, and a wrong config kills the match
        return _by_alphabet(text_, fts)
    fallback = _code_of(fts.fallback, fts)
    try:
        # a language we cannot search is a language we should not answer in either
        code = detect(text_)
    except LangDetectException:
        return fallback
    return code if code in fts.languages else fallback


# the full-text config a text is searched with
def ts_config(text_, mode=None):
    fts = config.settings.fts
    return fts.languages.get(detect_language(text_, mode), fts.fallback)


# which language an answer belongs in: asked for, else the one the question is written in
def resolve_language(question: str, language: str | None) -> str:
    return language or detect_language(question)
