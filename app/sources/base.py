import fnmatch
import re
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path

import config
import frontmatter
import ingest
import logging_setup
from book_matter import is_matter
from corpus_keys import language_by_alphabet
from sources.declaration import DEFAULT_INCLUDE, skipped_path
from use_cases import markdown_cleanup

log = logging_setup.get_logger(__name__)

# the largest markdown in the corpus is 110 KB; anything far past that is not a document
MAX_FILE_BYTES = 8 * 1024 * 1024

HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)


STRUCTURED = "structured"


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
    cut_by = ingest.cut_structured if policy.get("chunker") == STRUCTURED else ingest.cut_with_root
    on = policy.get("ceiling_on", ingest.BODY)
    for cut in cut_by(ingest.without_index(content, file), root, ceiling=ceiling, ceiling_on=on, file=file):
        yield cut.prefix + cut.body, cut.body, cut.section, root, cut.cut_by


# a tag in the label alphabet the filter takes: «Java & JVM» as written would be stored and never asked for
def labels(tags) -> list[str]:
    return list(dict.fromkeys(re.sub(r"[^\w-]", "_", str(t).strip().lower()) for t in tags if str(t).strip()))


@dataclass
class Doc:
    content: str
    source: str
    category: str | None
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
    # the released versions this text stands for; one reader writes its own, the merge of versions unions them
    versions: list[str] = field(default_factory=list)


@dataclass
class Parsed:
    content: str
    category: str | None
    title: str | None
    links: list[str]
    tags: list[str]


def _as_list(value) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


class Base(ABC):
    name: str
    root: Path
    # the key a source file names in `reader`; the class parses what its file cannot say
    reader: str | None = None
    # a reader of code-defined rules has one source file; a shared one reads any source that names it
    one_file: bool = True
    _registry: dict[str | None, type["Base"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "reader", None):
            Base._registry[cls.reader] = cls

    def __init__(
        self, root: Path, settings=None, name: str | None = None, version: str | None = None, onboarded: bool = False
    ):
        from sources import files

        self.root = root
        self.left_out_as_matter: list[tuple] = []
        self._index_left: set[tuple[str, str]] = set()
        self.settings = settings or files.of_reader(self.reader)
        self.name = name or self.settings.name
        self.version = version
        self.onboarded = onboarded

    @property
    def language(self) -> str:
        return self.settings.language

    def _include(self) -> list[str]:
        origin = self.settings.git or self.settings.git_family
        return origin.include if origin else DEFAULT_INCLUDE

    # overlapping patterns must not cut one file twice, and a directory named like a document is not one
    def files(self):
        seen = {}
        for pattern in self._include():
            seen.update((f, None) for f in self.root.glob(pattern) if f.is_file())
        return iter(seen)

    def _skipped(self, stem: str, policy=None) -> bool:
        rules = [*self.settings.skip, *self.settings.skip_when_hygienic]
        return any(fnmatch.fnmatchcase(stem, rule) for rule in rules)

    def discover(self, policy=None):
        return (f for f in self.files() if not self._skipped(f.stem, policy) and not self._path_skipped(f)
                and self._inside_root(f))

    def _path_skipped(self, file) -> bool:
        return bool(self.settings.skip_paths) and skipped_path(self.rel_of(file), self.settings.skip_paths) is not None

    # a .md symlink out of the corpus reads whatever the worker can, and it ends up quoted
    def _inside_root(self, file) -> bool:
        try:
            resolved = Path(file).resolve()
            resolved.relative_to(Path(self.root).resolve())
        except (OSError, ValueError):
            log.warning("source.outside_root", file=str(file), source=self.name)
            return False
        return True

    # what marks, veto prefixes and the model's [source] read: a checkout or an onboarded tree is named by its row
    @property
    def spelled_as(self) -> str:
        return self.name if self.version or self.onboarded else self.root.name

    # a key of the map: the longest path prefix the file names, else the source's one category, else none declared
    def category_for(self, rel_path) -> str | None:
        rel = str(rel_path)
        by_path = [p for p in self.settings.category_by_path if rel.startswith(p)]
        if by_path:
            return self.settings.category_by_path[max(by_path, key=len)]
        if len(self.settings.categories) > 1:
            raise ValueError(f"{self.name}: {rel} is under no category_by_path and the source names several categories")
        return self.settings.categories[0] if self.settings.categories else None

    # the declared tags, else the file's folders and stem, the labels the old category path carried
    def tags_for(self, rel_path) -> list[str]:
        declared = self.settings
        if not (declared.tags or declared.tag_from_name is not None or declared.tags_by_path):
            return list(Path(rel_path).with_suffix("").parts)
        parts = Path(rel_path).parts
        named = [self.name.removesuffix(declared.tag_from_name)] if declared.tag_from_name is not None else []
        by_path = [parts[r.step] for r in declared.tags_by_path
                   if str(rel_path).startswith(r.prefix) and len(parts) > r.step]
        return [*declared.tags, *named, *by_path]

    # a byte order mark hides the frontmatter fence and the first heading: one redis page had one
    def text_of(self, file) -> str:
        if file.stat().st_size > MAX_FILE_BYTES:
            log.warning("source.file_too_large", file=str(file), bytes=file.stat().st_size)
            return ""
        return markdown_cleanup.markdown_of(file).lstrip("\ufeff")

    def read(self, file, rel, policy=None):
        text = self.text_of(file)
        declared = self.settings
        tags = self.tags_for(rel)
        if declared.skip_when_frontmatter or declared.tags_from_frontmatter:
            meta = frontmatter.loads(text).metadata
            if any(str(meta.get(key)) == value for key, value in declared.skip_when_frontmatter.items()):
                return None
            if declared.tags_from_frontmatter:
                tags = [str(t) for key in declared.tags_from_frontmatter for t in _as_list(meta.get(key))]
        content = markdown_cleanup.as_indexed(text, declared.markup)
        return Parsed(content, self.category_for(rel), self.title_from(content), [], tags)

    def title_from(self, content):
        return first_heading(content)

    # where the heading path starts: markdown for most, but a source may declare it anywhere
    def section_root_for(self, file, parsed) -> str | None:
        named_by_file = self.settings.section_root_from_filename
        if named_by_file and Path(file).parent.name == named_by_file:
            return Path(file).stem.replace("-", " ").upper()
        # frontmatter is yaml: `title: 101` arrives as an int
        title = parsed.title
        return None if title is None else str(title).strip() or None

    # the one door onto a source's files, for the index, the quality report and the digest
    def documents(self, policy=None):
        policy = policy or config.settings.corpus.policy()
        found = list(self.discover(policy))
        self._refuse_uncategorised(found)
        docs, matter = [], set()
        self._index_left = set()
        for file in found:
            for doc in self.to_documents(file, policy):
                if is_matter(doc.source, doc.section):
                    matter.add((doc.source, doc.section))
                else:
                    docs.append(doc)
        # a section left out by its heading and a back index cut by position are named, so a report shows what went
        self.left_out_as_matter = sorted(matter | self._index_left)
        if self.left_out_as_matter:
            log.info("source.book_matter_left_out", source=self.name, sections=len(self.left_out_as_matter),
                     first=[f"{f} # {s}" for f, s in self.left_out_as_matter[:5]])
        return drop_wide_boilerplate(docs, policy)

    # every file without a category named at once, before the first cut, not the first of them halfway through
    def _refuse_uncategorised(self, found) -> None:
        if len(self.settings.categories) < 2:
            return
        prefixes = tuple(self.settings.category_by_path)
        loose = [rel for f in found if not (rel := self.rel_of(f)).startswith(prefixes)]
        if loose:
            raise ValueError(f"{self.name}: {len(loose)} files under no category_by_path, e.g. {loose[:5]}")

    # the path a document names its file by, in the source as a reader would find it
    def rel_of(self, file) -> str:
        return str(file.relative_to(self.root))

    def to_documents(self, file, policy=None):
        policy = policy or {}
        rel = self.rel_of(file)
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
                source=f"{self.spelled_as}/{rel}",
                category=parsed.category,
                # a chunk's own alphabet: an English footnote or SQL in a Russian book is searched as English
                language=language_by_alphabet(body or content),
                title=parsed.title,
                links=parsed.links,
                tags=labels(parsed.tags),
                chunk_index=i,
                section=section,
                versions=[self.version] if self.version else [],
            )

    def _cuts(self, file, parsed, policy):
        root = self.section_root_for(file, parsed)
        source = f"{self.spelled_as}/{self.rel_of(file)}"
        self._index_left.update((source, left) for left in ingest.index_left_out(parsed.content, str(file)))
        return cuts_of(parsed.content, root, policy, str(file))

    # a share-of-symbols rule caught nothing: ascii art reads as prose to every ratio we tried
    def postprocess(self, docs: list[Doc], policy: dict | None = None) -> list[Doc]:
        dropped = self.settings.drop_docs_containing
        if not dropped:
            return docs
        kept = [d for d in docs if not any(text in d.content for text in dropped)]
        for i, doc in enumerate(kept):
            doc.chunk_index = i
        return kept
