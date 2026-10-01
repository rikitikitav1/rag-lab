import re
from dataclasses import dataclass

import config

# a door's list of sources to search, the same bound at every door
MAX_SOURCES = 100


# one rule for both doors: a literal label; the old dotted path refuses rather than finds nothing
CATEGORY_RE = re.compile(r"^[\w-]+$")
VERSION_RE = re.compile(r"^[\w.-]{1,32}$")


def refuse_bad_category(category: str | None) -> None:
    if category is not None and not CATEGORY_RE.fullmatch(category):
        raise ValueError(f"invalid category filter: {category[:60]!r}")


# a label is a category of the map, a group of them, or a tag; the filter takes any of the three
def categories_of(label: str) -> list[str]:
    cats = config.settings.categories
    return [key for key, row in cats.items() if key == label or row.group == label]


class ScopeRefused(ValueError):
    pass


# what a search may read: a label (a category, a group or a tag), sources by name, one version of one category
@dataclass(frozen=True)
class Scope:
    label: str | None = None
    sources: tuple[str, ...] = ()
    version: str | None = None

    # tags are stored lowercased, so a label asked as «Redis» reads what the index wrote as «redis»
    def __post_init__(self):
        if self.label:
            object.__setattr__(self, "label", self.label.lower())

    # the one way a door builds it: a list of sources, possibly empty or None, becomes the tuple the search reads
    @classmethod
    def of(cls, label: str | None, sources: list[str] | tuple | None, version: str | None) -> "Scope":
        return cls(label=label, sources=tuple(sources or ()), version=version)

    @property
    def narrowed(self) -> bool:
        return bool(self.label or self.sources or self.version)


def as_scope(scope) -> Scope:
    if scope is None:
        return Scope()
    return scope if isinstance(scope, Scope) else Scope(label=scope)


# the rules a scope keeps by itself: a version is a category's, and a category lacking it is said, not emptied
def refuse_malformed_scope(scope: Scope) -> None:
    refuse_bad_category(scope.label)
    if scope.version is None:
        return
    if not scope.label:
        raise ScopeRefused(f"version {scope.version} names no category; add the category it belongs to")
    cats = categories_of(scope.label)
    if len(cats) != 1:
        raise ScopeRefused(f"a version belongs to one category, and {scope.label} names {len(cats)}")
    listed = config.settings.categories[cats[0]].versions
    if not listed:
        raise ScopeRefused(f"{cats[0]} has no versions declared")
    if scope.version not in listed:
        raise ScopeRefused(f"{cats[0]} has no version {scope.version}; listed: {listed}")
