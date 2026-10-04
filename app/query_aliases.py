import hashlib
import json
import re
from functools import lru_cache

import config

_TOKEN = re.compile(r"[0-9a-zа-яё][\w+#.-]*[0-9a-zа-яё+#]|[0-9a-zа-яё]", re.IGNORECASE)
_CYRILLIC = re.compile(r"[а-яё]")
# a Russian alias takes a case ending (постгресе, кубере), a short one would also swallow ordinary words
STEM_MIN, ENDING_MAX = 4, 3


def active() -> dict[tuple[str, ...], str]:
    return _active(json.dumps({k: v.model_dump() for k, v in config.settings.aliases.items()}, sort_keys=True))


@lru_cache(maxsize=4)
def _active(entries: str) -> dict[tuple[str, ...], str]:
    out = {}
    for entry in json.loads(entries).values():
        for alias in set(entry["aliases"]) - set(entry["unsure"]):
            words = tuple(t.lower() for t in _TOKEN.findall(alias))
            if words and " ".join(words) != entry["canonical"].lower():
                out[words] = entry["canonical"]
    return out


def digest() -> str:
    pairs = sorted((" ".join(words), name) for words, name in active().items())
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode()).hexdigest()[:12]


def _matches(word: str, token: str, last: bool) -> bool:
    if token == word:
        return True
    return (last and _CYRILLIC.search(word) is not None and len(word) >= STEM_MIN
            and token.startswith(word) and len(token) - len(word) <= ENDING_MAX)


# the question with each alias replaced by its technology's name, and the aliases that fired
def reword(question: str) -> tuple[str, list[str]]:
    if not config.settings.retrieval.keyword.aliases.enabled:
        return question, []
    table = active()
    longest = max((len(words) for words in table), default=0)
    tokens = list(_TOKEN.finditer(question))
    out, fired, at, i = [], [], 0, 0
    while i < len(tokens):
        for n in range(min(longest, len(tokens) - i), 0, -1):
            span = [t.group(0).lower() for t in tokens[i:i + n]]
            hit = next((name for words, name in table.items() if len(words) == n and all(
                _matches(w, t, k == n - 1) for k, (w, t) in enumerate(zip(words, span, strict=True)))), None)
            if hit:
                out += [question[at:tokens[i].start()], hit]
                fired.append(f"{' '.join(span)}={hit}")
                at, i = tokens[i + n - 1].end(), i + n
                break
        else:
            i += 1
    return "".join([*out, question[at:]]), fired
