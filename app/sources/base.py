import fnmatch
import re
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import ingest
import logging_setup
from sources.declaration import DEFAULT_INCLUDE

log = logging_setup.get_logger(__name__)

# the largest markdown in the corpus is 110 KB; anything far past that is not a document
MAX_FILE_BYTES = 8 * 1024 * 1024

HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)


ROOTED = "rooted"
STRUCTURED = "structured"
HYGIENIC_CHUNKERS = frozenset({ROOTED, STRUCTURED})


# one explicit key decides the cut: a side effect would hand an odd mix a cut nobody asked
def hygienic(policy) -> bool:
    return bool(policy) and policy.get("chunker") in HYGIENIC_CHUNKERS


# the only carrier of its section stays: on this corpus that saved six gold sections
def drop_wide_boilerplate(docs: list["Doc"], policy: dict | None = None) -> list["Doc"]:
    if not (policy or {}).get("drop_boilerplate"):
        return docs
    min_files = ingest.BOILERPLATE_MIN_FILES
    bodied = [d for d in docs if d.body is not None]
    files = {d.source for d in bodied}
    if len(files) < min_files:
        return docs
    wide = ingest.wide_bodies(((d.body, d.source) for d in bodied), len(files))
    if not wide:
        return docs
    carried = {(d.source, d.section) for d in docs if d.body is None or d.body not in wide}
    return [doc for doc in docs if doc.body is None or doc.body not in wide or (doc.source, doc.section) not in carried]


# the first heading, not the first line: primer opens with a banner, redis with frontmatter
def first_heading(content) -> str | None:
    found = HEADING.search(content or "")
    return found.group(1).strip() or None if found else None


# one file's text as the variant's chunker cuts it, for the index and for a raw report alike
def cuts_of(content: str, root: str | None, policy: dict, file: str):
    ceiling = policy.get("max_chunk_size")
    if not hygienic(policy):
        # naming the H1 copy is what lets the same body metrics run on baseline
        head = content.lstrip().split("\n", 1)[0]
        h1 = f"{head}\n" if head.startswith("# ") else ""
        section = None
        for i, chunk in enumerate(ingest.chunk_markdown(content, ceiling=ceiling)):
            section = ingest.heading_path(chunk) or section
            body = chunk[len(h1) :] if h1 and i and chunk.startswith(h1) else chunk
            yield chunk, body, section, None, None
        return
    cut_by = ingest.cut_structured if policy.get("chunker") == STRUCTURED else ingest.cut_with_root
    on = policy.get("ceiling_on", ingest.BODY)
    for cut in cut_by(content, root, ceiling=ceiling, ceiling_on=on, file=file):
        yield cut.prefix + cut.body, cut.body, cut.section, root, cut.cut_by


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


# the chunk rows and the text-search trigger still spell the language in three letters
_CHUNK_LANGUAGE = {"en": "eng", "ru": "rus"}


class Base(ABC):
    name: str
    root: Path
    # the key a source file names in `reader`; the class parses what its file cannot say
    reader: str | None = None
    _registry: dict[str | None, type["Base"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "reader", None):
            Base._registry[cls.reader] = cls

    def __init__(self, root: Path, settings=None, name: str | None = None):
        from sources import files

        self.root = root
        self.settings = settings or files.of_reader(self.reader)
        self.name = name or self.settings.name

    @property
    def language(self) -> str:
        return _CHUNK_LANGUAGE[self.settings.language]

    def _include(self) -> list[str]:
        origin = self.settings.git or self.settings.git_family
        return origin.include if origin else DEFAULT_INCLUDE

    # overlapping patterns must not cut one file twice, and a directory named like a document is not one
    def files(self):
        seen = {}
        for pattern in self._include():
            seen.update((f, None) for f in self.root.glob(pattern) if f.is_file())
        return iter(seen)

    # baseline is intact because the file says which rules the hygienic cut adds
    def _skipped(self, stem: str, policy=None) -> bool:
        rules = [*self.settings.skip, *(self.settings.skip_when_hygienic if hygienic(policy) else [])]
        return any(fnmatch.fnmatchcase(stem, rule) for rule in rules)

    def discover(self, policy=None):
        return (f for f in self.files() if not self._skipped(f.stem, policy) and self._inside_root(f))

    # a .md symlink out of the corpus reads whatever the worker can, and it ends up quoted
    def _inside_root(self, file) -> bool:
        try:
            resolved = Path(file).resolve()
            resolved.relative_to(Path(self.root).resolve())
        except (OSError, ValueError):
            log.warning("source.outside_root", file=str(file), source=self.name)
            return False
        return True

    def category_for(self, rel_path: Path):
        return (self.settings.categories.prefix or "") + ingest.path_to_category(rel_path)

    # a byte order mark hides the frontmatter fence and the first heading: one redis page had one
    def text_of(self, file) -> str:
        return self.legacy_text_of(file).lstrip("\ufeff")

    def read(self, file, rel, policy=None):
        content = self.text_of(file) if hygienic(policy) else self.legacy_text_of(file)
        title = self.title_from(content) if hygienic(policy) else self.legacy_title_from(content)
        return Parsed(content, self.category_for(rel), title, [], [])

    def title_from(self, content):
        return first_heading(content)

    def legacy_text_of(self, file) -> str:
        if file.stat().st_size > MAX_FILE_BYTES:
            log.warning("source.file_too_large", file=str(file), bytes=file.stat().st_size)
            return ""
        return file.read_text(encoding="utf-8", errors="ignore")

    def legacy_title_from(self, content):
        if not (content or "").strip():
            return None
        return content.splitlines()[0].lstrip("#").strip() or None

    # where the heading path starts: markdown for most, but a source may declare it anywhere
    def section_root_for(self, file, parsed) -> str | None:
        # frontmatter is yaml: `title: 101` arrives as an int
        title = parsed.title
        return None if title is None else str(title).strip() or None

    # the one door onto a source's files, for the index, the quality report and the digest
    def documents(self, policy=None):
        policy = policy or {}
        docs = [doc for file in self.discover(policy) for doc in self.to_documents(file, policy)]
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
        root = self.section_root_for(file, parsed) if hygienic(policy) else None
        return cuts_of(parsed.content, root, policy, str(file))

    # a share-of-symbols rule caught nothing: ascii art reads as prose to every ratio we tried
    def postprocess(self, docs: list[Doc], policy: dict | None = None) -> list[Doc]:
        dropped = self.settings.drop_docs_containing
        if not (hygienic(policy) and dropped):
            return docs
        kept = [d for d in docs if not any(text in d.content for text in dropped)]
        for i, doc in enumerate(kept):
            doc.chunk_index = i
        return kept
