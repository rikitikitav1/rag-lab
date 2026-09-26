from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Language = Literal["en", "ru"]


# what a repository or a folder gives the index when a source names no files of its own
DEFAULT_INCLUDE = ["**/*.md"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GitOrigin(_Strict):
    # a url, never a word git could read as its own option
    repo: str = Field(pattern=r"^(https://|git@)")
    ref: str | None = None
    path: str | None = None
    include: list[str] = DEFAULT_INCLUDE


# where a site keeps its own text, what inside it is the site's furniture, which pages it builds itself
class SiteSettings(_Strict):
    main: str
    drop: list[str] = []
    generated: list[str] = []


# an added source as its owner declares it: what, where from, what language and licence; never which engine
class Declaration(_Strict):
    ORIGINS: ClassVar[tuple[str, ...]] = ("urls", "folder", "git", "pages")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$")
    language: Language
    licence: str = Field(min_length=1)
    urls: list[str] | None = None
    folder: str | None = None
    git: GitOrigin | None = None
    pages: list[str] | None = None
    site: SiteSettings | None = None

    @model_validator(mode="after")
    def _one_origin(self):
        given = [k for k in self.ORIGINS if getattr(self, k)]
        if len(given) != 1:
            raise ValueError(f"a source comes from exactly one of {', '.join(self.ORIGINS)}; given {given or 'none'}")
        if self.site is not None and not self.pages:
            raise ValueError("site settings belong to pages of a site")
        return self


# one organisation's repositories read the same way, each a source row of its own
class GitFamily(_Strict):
    base_url: str = Field(pattern=r"^https://")
    repos: list[str] = Field(min_length=1)
    include: list[str] = DEFAULT_INCLUDE


# a category from the front matter's field, else from the file's stem, under a prefix the source owns
class Categories(_Strict):
    prefix: str | None = None
    by_front_matter: dict[str, str] = {}
    by_file: dict[str, str] = {}


# the veto build joins families over every source file; a family is a path prefix, not the whole source
class VetoFamily(_Strict):
    name: str
    prefix: str


# a worked-out source as the stand keeps it in `sources/<name>.yaml`: the declaration plus the rules only it needs
class SourceFile(Declaration):
    ORIGINS: ClassVar[tuple[str, ...]] = ("folder", "git", "git_family")
    git_family: GitFamily | None = None
    # the class that parses what the rules cannot say; none reads plain markdown
    reader: str | None = None
    categories: Categories = Categories()
    # stems skipped always, and by the hygienic cut only with a reason; `fnmatch` patterns, so `[` and `?` match
    skip: list[str] = []
    skip_when_hygienic: dict[str, str] = {}
    drop_docs_containing: list[str] = []
    veto_families: list[VetoFamily] = []
    # a folder that moves on its own, so its fingerprint drifting is not a fault
    drifts: bool = False

    # the index reads a folder, a whole repository or a family; the rest is onboarded by hand first, so refused at load
    @model_validator(mode="after")
    def _what_the_index_reads(self):
        if self.urls or self.pages or self.site:
            raise ValueError(f"{self.name}: urls, pages and site are onboarded through the door, not read by the index")
        if self.git is not None and (self.git.ref or self.git.path):
            raise ValueError(f"{self.name}: a source file clones the default branch whole; ref and path are not read")
        return self
