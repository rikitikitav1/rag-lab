from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Language = Literal["en", "ru"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GitOrigin(_Strict):
    # a url, never a word git could read as its own option
    repo: str = Field(pattern=r"^(https://|git@)")
    ref: str | None = None
    path: str | None = None
    include: list[str] = ["**/*.md"]


# where a site keeps its own text, what inside it is the site's furniture, which pages it builds itself
class SiteSettings(_Strict):
    main: str
    drop: list[str] = []
    generated: list[str] = []


# an added source as its owner declares it: what, where from, what language and licence; never which engine
class Declaration(_Strict):
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
        given = [k for k in ("urls", "folder", "git", "pages") if getattr(self, k)]
        if len(given) != 1:
            raise ValueError(f"a source comes from exactly one of urls, folder, git, pages; given {given or 'none'}")
        if self.site is not None and not self.pages:
            raise ValueError("site settings belong to pages of a site")
        return self
