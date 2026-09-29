import re

from corpus_keys import chapter_of

# what the generator reads in one call: the index's own chunks joined, about 1300 words, one window of acceptance
BLOCK_CHARS = 8000
# a subsection starts a new block once the block holds this share of its ceiling; smaller ones join their neighbour
BLOCK_BREAK_SHARE = 0.5
_SUBHEADING = re.compile(r"#{3,6} ")


# a source's sections as the index cuts them, keyed as its chunks are: what a question generator reads and marks
def export(docs) -> dict:
    sections: dict[tuple, dict] = {}
    runs: dict[tuple, int] = {}
    refused: dict[tuple, str] = {}
    last: dict[str | None, tuple] = {}
    for doc in docs:
        # a versioned source hands each version's own texts after the newest one's: a section is read within its stream
        stream = doc.versions[0] if doc.versions else None
        key = (doc.source, doc.section, stream)
        if doc.section is None:
            refused[key] = "no heading path: a gold names its section"
            continue
        # a path that comes back after another one is two sections under one name, and a gold could not tell them apart
        if key != last.get(stream):
            runs[key] = runs.get(key, 0) + 1
        last[stream] = key
        row = sections.setdefault(
            key, {"file": doc.source, "section": doc.section, "stream": stream, "chapter": chapter_of(doc.section),
                  "versions": []},
        )
        row.setdefault("text", [])
        row["text"].append(doc.body if doc.body is not None else doc.content)
        row["versions"] = sorted(set(row["versions"]) | set(doc.versions), reverse=True)
    for key, times in runs.items():
        if times > 1:
            refused[key] = f"its whole path comes back {times} times in the file"
            sections.pop(key)
    rows = []
    for row in sections.values():
        bodies = row.pop("text")
        text = "\n\n".join(bodies)
        rows.append({**row, "words": len(text.split()), "text": text, "blocks": blocks(bodies)})
    return {
        "sections": rows,
        "refused": [{"file": f, "section": s, "version": v, "why": why} for (f, s, v), why in refused.items()],
    }


# a section's chunks joined in order into the generator's blocks, broken at a subheading or at the ceiling
def blocks(bodies: list[str]) -> list[str]:
    out, held, size = [], [], 0
    for body in bodies:
        starts_a_subsection = bool(_SUBHEADING.match(body.lstrip()))
        full = size + len(body) > BLOCK_CHARS
        if held and (full or (starts_a_subsection and size >= BLOCK_CHARS * BLOCK_BREAK_SHARE)):
            out.append("\n\n".join(held))
            held, size = [], 0
        held.append(body)
        size += len(body) + 2
    if held:
        out.append("\n\n".join(held))
    return out


# the questions each chapter is drawn to: the source's ceiling spread by words, no chapter above its own ceiling
def chapter_quota(rows: list[dict], per_source: int, per_chapter: int) -> dict[str | None, int]:
    words: dict[str | None, int] = {}
    for row in rows:
        words[row["chapter"]] = words.get(row["chapter"], 0) + row["words"]
    quota = {chapter: 0 for chapter in words}
    # highest averages: each question goes to the chapter with the most words per question it would hold
    for _ in range(min(per_source, per_chapter * len(words))):
        best = max((c for c in words if quota[c] < per_chapter), key=lambda c: words[c] / (quota[c] + 1))
        quota[best] += 1
    return quota


# an accepted source read by the index's own readers, so a question's gold is spelled as its chunks are
def of_source(name: str) -> dict:
    import config
    from sources import factory

    policy = config.settings.corpus.policy(config.settings.corpus.variant)
    out = export(doc for reader in factory.sources(names=[name]) for doc in reader.documents(policy))
    limits = config.settings.evals.question_set
    return {"source": name, **out, "quota": chapter_quota(out["sections"], limits.per_source, limits.per_chapter)}
