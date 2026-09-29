import copy
from typing import ClassVar, Literal

from config import SOURCE_KNOBS, RouteCfg
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator
from tool_names import settings_refusal

Language = Literal["en", "ru"]
SOURCE_NAME = r"^[a-z0-9][a-z0-9_-]{0,62}$"
# one vocabulary for the kind column: a folder is `local`, as the code-defined sources already say
KINDS = {"urls": "urls", "folder": "local", "git": "git", "git_family": "git", "pages": "pages"}


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
    # the product's release the pages document, the key of the site's version; the fetch date is only its label
    release: str | None = Field(default=None, min_length=1)


# a row's site settings read through their model, so every reader sees the same validated fields
def site_of(origin: dict) -> SiteSettings | None:
    return SiteSettings.model_validate(origin["site"]) if origin.get("site") else None


# the intake's knobs one source sets for itself; each left out keeps the stand's own in `config/intake.yaml`
def _optional(field):
    optional = copy.copy(field)
    optional.default, optional.annotation = None, field.annotation | None
    return optional.annotation, optional


# every route knob of the stand, optional here: a key left out keeps the stand's own
_Knobs = create_model(
    "_Knobs", __base__=_Strict, **{name: _optional(RouteCfg.model_fields[name]) for name in SOURCE_KNOBS}
)


class IntakeOverride(_Knobs):
    # a tool's settings file for this source, as `docling/pypdfium2`
    settings: dict[Literal["docling", "mineru"], str] = {}

    # a misspelt settings file is refused at the door, not found missing inside a job hours later
    @field_validator("settings")
    @classmethod
    def _settings_files(cls, settings):
        for tool, name in settings.items():
            if refusal := settings_refusal(tool, name):
                raise ValueError(refusal)
        return settings

    @field_validator("reread_settings")
    @classmethod
    def _reread_file(cls, name):
        if name is not None and (refusal := settings_refusal("docling", name)):
            raise ValueError(refusal)
        return name


# one organisation's repositories read the same way, each a source row of its own
class GitFamily(_Strict):
    base_url: str = Field(pattern=r"^https://")
    repos: list[str] = Field(min_length=1)
    include: list[str] = DEFAULT_INCLUDE

    # a family's row is one repository of it, named by the row
    def repo_of(self, name: str) -> str:
        return f"{self.base_url}/{name}"


# where one version of a source lives: a branch or tag of its repository, or a folder of its own
class VersionOrigin(_Strict):
    ref: str | None = None
    folder: str | None = None

    @model_validator(mode="after")
    def _one_place(self):
        if (self.ref is None) == (self.folder is None):
            raise ValueError("a version lives at a ref or in a folder, exactly one")
        return self


# the veto build joins families over every source file; a family is a path prefix, not the whole source
class VetoFamily(_Strict):
    name: str
    prefix: str


# a question the owner gives with the source, so the source skips question generation; its gold is a file for now
class DeclaredQuestion(_Strict):
    text: str = Field(min_length=5, max_length=2000)
    file: str | None = None


# a source as the stand keeps it in its row, from the door or the seed: where it comes from and the rules only it needs
class Declaration(_Strict):
    ORIGINS: ClassVar[tuple[str, ...]] = ("urls", "folder", "git", "git_family", "pages")
    name: str = Field(pattern=SOURCE_NAME)
    # optional: onboarding reads it from the source's text when undeclared; each chunk takes its own alphabet
    language: Language | None = None
    licence: str | None = Field(default=None, min_length=1)
    urls: list[str] | None = None
    folder: str | None = None
    git: GitOrigin | None = None
    git_family: GitFamily | None = None
    pages: list[str] | None = None
    site: SiteSettings | None = None
    intake: IntakeOverride | None = None
    # the class that parses what the rules cannot say; none reads plain markdown
    reader: str | None = None
    # stems skipped always, and by the hygienic cut only with a reason; `fnmatch` patterns, so `[` and `?` match
    skip: list[str] = []
    skip_when_hygienic: dict[str, str] = {}
    drop_docs_containing: list[str] = []
    veto_families: list[VetoFamily] = []
    # a folder that moves on its own, so its fingerprint drifting is not a fault
    drifts: bool = False
    # rows of `config/categories.yaml` the source covers; a path prefix picks one where a source covers several
    categories: list[str] = []
    category_by_path: dict[str, str] = {}
    # released versions side by side, newest first as the map lists them; empty is one rolling version
    versions: dict[str, VersionOrigin] = {}
    questions: list[DeclaredQuestion] = []

    @model_validator(mode="after")
    def _one_origin(self):
        given = [k for k in self.ORIGINS if getattr(self, k)]
        if len(given) != 1:
            raise ValueError(f"a source comes from exactly one of {', '.join(self.ORIGINS)}; given {given or 'none'}")
        if self.site is not None and not self.pages:
            raise ValueError("site settings belong to pages of a site")
        return self


# a worked-out source as the stand keeps it in `sources/<name>.yaml`: the declaration plus the rules only it needs
class SourceFile(Declaration):
    # a file clones a branch whole and names each version's own place; the door's declarations need neither rule
    @model_validator(mode="after")
    def _what_the_index_reads(self):
        if self.git is not None and (self.git.ref or self.git.path):
            raise ValueError(
                f"{self.name}: a source file clones a branch whole; a ref is named per version, a path never"
            )
        wrong = "ref" if self.folder is not None else "folder"
        if self.git_family is not None and self.versions:
            raise ValueError(
                f"{self.name}: a family's repositories are rows of their own, versions belong to one source"
            )
        if self.folder is not None and self.versions and self.folder != next(iter(self.versions.values())).folder:
            raise ValueError(f"{self.name}: a folder source with versions names its newest version's folder as its own")
        if any(getattr(v, wrong) is not None for v in self.versions.values()):
            raise ValueError(
                f"{self.name}: a {'folder' if wrong == 'ref' else 'git'} source's versions name no {wrong}"
            )
        return self
