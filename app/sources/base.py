import re
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import ingest
import logging_setup

log = logging_setup.get_logger(__name__)

# the largest markdown in the corpus is 110 KB; anything far past that is not a document
MAX_FILE_BYTES = 8 * 1024 * 1024

HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)


ROOTED = "rooted"
STRUCTURED = "structured"
HYGIENIC_CHUNKERS = frozenset({ROOTED, STRUCTURED})


# one explicit key decides which cut a variant gets. Deriving it from a side effect like
# "boilerplate: dropped" would give a variant that declares an unusual mix of rules a cut
# nobody asked for, silently
def hygienic(policy) -> bool:
    return bool(policy) and policy.get("chunker") in HYGIENIC_CHUNKERS


# from collections.abc import Iterable


# a block repeated across half a source's files carries rank and no answer. The exception
# is what the rule is worth: the only carrier of its section stays, because hygiene that
# removes the answer is not hygiene. On this corpus it saved six gold sections
def drop_wide_boilerplate(docs: list["Doc"], policy: dict | None = None) -> list["Doc"]:
    if not (policy or {}).get("drop_boilerplate"):
        return docs
    share, min_files = ingest.BOILERPLATE_FILE_SHARE, ingest.BOILERPLATE_MIN_FILES
    bodied = [d for d in docs if d.body is not None]
    files = {d.source for d in bodied}
    if len(files) < min_files:
        return docs
    spread: dict[str, set[str]] = {}
    for doc in bodied:
        spread.setdefault(doc.body, set()).add(doc.source)
    wide = {body for body, seen in spread.items() if len(seen) / len(files) >= share}
    if not wide:
        return docs
    carried = {
        (d.source, d.section) for d in docs if d.body is None or d.body not in wide
    }
    return [
        doc
        for doc in docs
        if doc.body is None
        or doc.body not in wide
        or (doc.source, doc.section) not in carried
    ]


@dataclass
class Doc:
    content: str
    source: str
    category: str
    language: str
    chunk_index: int
    title: str
    links: list[str]
    tags: list[str]
    # what the dedup hash is taken from: the answer, never the heading path prefix
    body: str | None = None
    section: str | None = None
    root: str | None = None
    cut_by: str | None = None


@dataclass
class Parsed:
    content: str
    category: str
    title: str | None
    links: list[str]
    tags: list[str]


class Base(ABC):
    name: str
    url: str | None
    root: Path
    language: str = "unknown"
    SKIP_FILE_NAMES: frozenset[str] = frozenset()
    # skipped only by a variant that asks for the hygienic cut; SKIP_FILE_NAMES above
    # predates it and always applies
    HYGIENIC_SKIP_FILE_NAMES: frozenset[str] = frozenset()
    _registry: dict[str, type["Base"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None):
            Base._registry[cls.name] = cls

    def __init__(self, root: Path):
        self.root = root

    def files(self):
        return self.root.rglob("*.md")

    # @abstractmethod
    # a variant that declares the old cut gets the old file set too: baseline has to be
    # intact because the code says so, not because nobody re-indexes it
    def skips(self, policy=None) -> frozenset[str]:
        if not hygienic(policy):
            return self.SKIP_FILE_NAMES
        return self.SKIP_FILE_NAMES | self.HYGIENIC_SKIP_FILE_NAMES

    def discover(self, policy=None):
        skip = self.skips(policy)
        return (
            f for f in self.files() if f.stem not in skip and self._inside_root(f)
        )

    # a .md symlink pointing out of the corpus reads whatever the worker can read, and
    # what a source yields ends up quoted in answers
    def _inside_root(self, file) -> bool:
        try:
            resolved = Path(file).resolve()
            resolved.relative_to(Path(self.root).resolve())
        except (OSError, ValueError):
            log.warning("source.outside_root", file=str(file), source=self.name)
            return False
        return True

    def category_for(self, rel_path: Path):
        return ingest.path_to_category(rel_path)

    # a byte order mark hides the frontmatter fence from the parser and the first
    # heading from the regexp: one redis page shipped with one
    def text_of(self, file) -> str:
        if file.stat().st_size > MAX_FILE_BYTES:
            log.warning("source.file_too_large", file=str(file), bytes=file.stat().st_size)
            return ""
        return file.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")

    def read(self, file, rel, policy=None):
        content = self.text_of(file) if hygienic(policy) else self.legacy_text_of(file)
        title = (
            self.title_from(content)
            if hygienic(policy)
            else self.legacy_title_from(content)
        )
        return Parsed(content, self.category_for(rel), title, [], [])

    def title_from(self, content):
        # the first heading, not the first line: primer opens with a translation
        # banner and redis docs open with the frontmatter fence
        found = HEADING.search(content or "")
        return found.group(1).strip() or None if found else None

    def legacy_text_of(self, file) -> str:
        if file.stat().st_size > MAX_FILE_BYTES:
            log.warning("source.file_too_large", file=str(file), bytes=file.stat().st_size)
            return ""
        return file.read_text(encoding="utf-8", errors="ignore")

    def legacy_title_from(self, content):
        if not (content or "").strip():
            return None
        return content.splitlines()[0].lstrip("#").strip() or None

    # where the heading path of this file starts. Markdown headings for most, but a
    # source is free to declare it from anywhere: frontmatter, the file name, a config
    def section_root_for(self, file, parsed) -> str | None:
        # frontmatter is yaml: `title: 101` arrives as an int
        title = parsed.title
        return None if title is None else str(title).strip() or None

    # the whole source at once: a rule that has to see every file cannot be applied one
    # file at a time. Every reader comes through here, or the ingest and the dry run end
    # up cutting the corpus two different ways
    def documents(self, policy=None):
        policy = policy or {}
        docs = [
            doc
            for file in self.discover(policy)
            for doc in self.to_documents(file, policy)
        ]
        return drop_wide_boilerplate(docs, policy)

    def to_documents(self, file, policy=None):
        policy = policy or {}
        rel = str(file.relative_to(self.root))
        parsed = self.read(file, rel, policy)
        if parsed is None:
            return []
        docs = list(self._docs(file, rel, parsed, policy))
        return self.postprocess(docs, policy)

    def _docs(self, file, rel, parsed, policy):
        cuts = self._cuts(file, parsed, policy)
        for i, (content, body, section, root, cut_by) in enumerate(cuts):
            yield Doc(
                content=content,
                body=body,
                root=root,
                cut_by=cut_by,
                source=f"{self.root.name}/{rel}",
                category=parsed.category,
                language=self.language,
                title=parsed.title,
                links=parsed.links,
                tags=parsed.tags,
                chunk_index=i,
                section=section,
            )

    def _cuts(self, file, parsed, policy):
        if not hygienic(policy):
            # chunk_markdown copies the file's H1 onto every section chunk. That copy is
            # the legacy prefix, and naming it is what lets the same body metrics run on
            # baseline: a variant that abstains everywhere can never be called dirty
            head = parsed.content.lstrip().split("\n", 1)[0]
            h1 = f"{head}\n" if head.startswith("# ") else ""
            section = None
            ceiling = policy.get("max_chunk_size")
            for i, chunk in enumerate(ingest.chunk_markdown(parsed.content, ceiling=ceiling)):
                section = ingest.heading_path(chunk) or section
                body = chunk[len(h1) :] if h1 and i and chunk.startswith(h1) else chunk
                yield chunk, body, section, None, None
            return
        root = self.section_root_for(file, parsed)
        ceiling = policy.get("max_chunk_size")
        cut_by = (
            ingest.cut_structured
            if policy.get("chunker") == STRUCTURED
            else ingest.cut_with_root
        )
        on = policy.get("ceiling_on", ingest.BODY)
        for cut in cut_by(parsed.content, root, ceiling=ceiling, ceiling_on=on, file=str(file)):
            yield cut.prefix + cut.body, cut.body, cut.section, root, cut.cut_by

    # a share-of-symbols rule used to live here and caught nothing on this corpus:
    # ascii art drawn with letters and spaces reads as prose to every ratio we tried,
    # and its one target is dropped by name instead. A rule that gates nothing tells
    # the next reader that symbol junk is handled, so it is gone rather than kept
    def postprocess(self, docs: list[Doc], policy: dict | None = None) -> list[Doc]:
        return docs
