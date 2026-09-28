import json
from functools import cached_property

from sources.base import Base
from use_cases import fetch


# an onboarded source's raw folder: each file's markdown whole, cut by the stand's one cut as its report cut it
class Converted(Base):
    reader = "converted"
    one_file = False

    # the raw folder carries its settings' hash in its name; documents name the source
    @property
    def spelled_as(self) -> str:
        return self.name

    def _include(self) -> list[str]:
        return ["files/*.md"]

    # a file's markdown is named by a slug of its path; the record gives the path back
    @cached_property
    def _originals(self) -> dict[str, str]:
        record = json.loads((self.root / "record.json").read_text())
        return {fetch.file_stem(row["file"]): row["file"] for row in record["units"].values()}

    def rel_of(self, file) -> str:
        return self._originals.get(file.stem) or super().rel_of(file)
