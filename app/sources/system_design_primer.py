from pathlib import Path

from sources.base import Base


class SystemDesignPrimerSource(Base):
    reader = "system-design-primer"

    def category_for(self, rel_path):
        parts = Path(rel_path).parts
        if parts[0] == "solutions":
            return f"system-design.{parts[2]}"
        return "system-design"
