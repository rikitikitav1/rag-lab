import hashlib
import json
import re

import config

_TOKEN = re.compile(r"[0-9a-zа-яё][\w+#.-]*[0-9a-zа-яё+#]|[0-9a-zа-яё]", re.IGNORECASE)
_CYRILLIC = re.compile(r"[а-яё]")
# a Russian alias takes a case ending (постгресе, кубере); a short stem or a free tail swallows words (рубить, монголия)
STEM_MIN = 4
CASE_ENDINGS = frozenset(("а", "я", "у", "ю", "е", "и", "ы", "ом", "ем", "ой", "ей", "ою", "ам", "ям", "ами", "ями",
                          "ах", "ях", "ов", "ев"))


def words_of(text: str) -> tuple[str, ...]:
    return tuple(t.lower() for t in _TOKEN.findall(text))


def active() -> dict[tuple[str, ...], str]:
    entries = config.settings.aliases
    if _CACHE.get("of") is not entries:
        _CACHE.update({"of": entries, "table": _active(entries)})
    return _CACHE["table"]


_CACHE: dict = {}


def _active(entries: dict) -> dict[tuple[str, ...], str]:
    out = {}
    for entry in entries.values():
        for alias in set(entry.aliases) - set(entry.unsure):
            words = words_of(alias)
            if words and words != words_of(entry.canonical):
                out[words] = entry.canonical
    return out


# a stem match that lands on a word the dictionary called unsure (дельфин under дельфи) is that word, not the alias
def _unsure() -> frozenset[str]:
    return frozenset(w for entry in config.settings.aliases.values() for u in entry.unsure for w in words_of(u))


def digest() -> str:
    pairs = sorted((" ".join(words), name) for words, name in active().items())
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode()).hexdigest()[:12]


def _matches(word: str, token: str, last: bool, unsure: frozenset) -> bool:
    if token == word:
        return True
    if not last or _CYRILLIC.search(word) is None or len(word) < STEM_MIN or token in unsure:
        return False
    # кафка declines as кафке: the final vowel is the ending, not the stem
    stem = word[:-1] if word[-1] in "аяоьй" else word
    return token.startswith(stem) and token[len(stem):] in CASE_ENDINGS


# the question with each alias replaced by its technology's name, and the aliases that fired
def reword(question: str) -> tuple[str, list[str]]:
    if not config.settings.retrieval.keyword.aliases.enabled:
        return question, []
    table, unsure = active(), _unsure()
    longest = max((len(words) for words in table), default=0)
    tokens = list(_TOKEN.finditer(question))
    out, fired, at, i = [], [], 0, 0
    while i < len(tokens):
        for n in range(min(longest, len(tokens) - i), 0, -1):
            span = [t.group(0).lower() for t in tokens[i:i + n]]
            hit = next((name for words, name in table.items() if len(words) == n and all(
                _matches(w, t, k == n - 1, unsure) for k, (w, t) in enumerate(zip(words, span, strict=True)))), None)
            if hit:
                out += [question[at:tokens[i].start()], hit]
                fired.append(f"{' '.join(span)}={hit}")
                at, i = tokens[i + n - 1].end(), i + n
                break
        else:
            i += 1
    return "".join([*out, question[at:]]), fired
