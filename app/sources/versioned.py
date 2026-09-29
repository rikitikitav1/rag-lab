import re
from collections import Counter

import config
from corpus_keys import body_hash
from sources.base import Base


# «PostgreSQL 18» and «PostgreSQL 17» are one text for the merge; a bare number is never touched, «limit 17» stays apart
def version_neutral(text: str | None, category: str, versions: list[str]) -> str:
    if not text:
        return text or ""
    name = re.escape(config.settings.categories[category].name)
    alternatives = "|".join(re.escape(v) for v in sorted(versions, key=len, reverse=True))
    return re.sub(rf"\b({name})\s+(?:{alternatives})\b", r"\1 {version}", text)


# the k-th copy of a text in one version meets the k-th copy in another; copies inside a version stay rows, as today
def merge_versions(per_version: list[tuple[str, list]], category: str) -> tuple[list, int]:
    versions = [v for v, _ in per_version]
    kept, by_key, merged = [], {}, 0
    for version, docs in per_version:
        seen = Counter()
        for doc in docs:
            base = (
                version_neutral(doc.section, category, versions),
                body_hash(version_neutral(doc.body or doc.content, category, versions)),
            )
            key = (*base, seen[base])
            seen[base] += 1
            if key in by_key:
                by_key[key].versions.append(version)
                merged += 1
                continue
            doc.versions = [version]
            by_key[key] = doc
            kept.append(doc)
    return _one_address_each(kept), merged


# an older version's own chunk numbers past the file's newest, or two texts would share `source#chunk_index`
def _one_address_each(kept: list) -> list:
    taken, top = set(), {}
    for doc in kept:
        if (doc.source, doc.chunk_index) in taken:
            doc.chunk_index = top[doc.source] + 1
        taken.add((doc.source, doc.chunk_index))
        top[doc.source] = max(top.get(doc.source, -1), doc.chunk_index)
    return kept


# several released versions under one row: documents() reads them all, any other attribute is the newest's reader
class Versioned:
    def __init__(self, readers: list[Base]):
        self.readers = readers
        self.newest = readers[0]
        self.merged = 0

    def __getattr__(self, attr):
        return getattr(self.newest, attr)

    def documents(self, policy=None):
        per_version = [(reader.version, reader.documents(policy)) for reader in self.readers]
        docs, self.merged = merge_versions(per_version, self.newest.settings.categories[0])
        self.left_out_as_matter = sorted({left for reader in self.readers for left in reader.left_out_as_matter})
        return docs
