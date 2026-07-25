from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import ingest

# from collections.abc import Iterable


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
    def discover(self):
        return (f for f in self.files() if f.stem not in self.SKIP_FILE_NAMES)

    def category_for(self, rel_path: Path):
        return ingest.path_to_category(rel_path)

    def read(self, file, rel):
        content = file.read_text(encoding="utf-8", errors="ignore")
        return Parsed(content, self.category_for(rel), self.title_from(content), [], [])

    def title_from(self, content):
        if not content.strip():
            return None
        return content.splitlines()[0].lstrip("#").strip() or None

    def to_documents(self, file):
        rel = str(file.relative_to(self.root))
        parsed = self.read(file, rel)
        if parsed is None:
            return
        for i, chunk in enumerate(ingest.chunk_markdown(parsed.content)):
            yield Doc(
                content=chunk,
                source=f"{self.root.name}/{rel}",
                category=parsed.category,
                language=self.language,
                title=parsed.title,
                links=parsed.links,
                tags=parsed.tags,
                chunk_index=i,
            )

    def postprocess(self, docs: list[Doc]):
        return docs
