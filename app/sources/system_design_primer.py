from pathlib import Path

from sources.base import Base


class SystemDesignPrimerSource(Base):
    name = "system-design-primer"
    language = "eng"
    kind = "git"
    url = "https://github.com/donnemartin/system-design-primer"

    def files(self):
        return self.root.rglob("README.md")

    def category_for(self, rel_path):
        parts = Path(rel_path).parts
        if parts[0] == "solutions":
            return f"system-design.{parts[2]}"
        return "system-design"
