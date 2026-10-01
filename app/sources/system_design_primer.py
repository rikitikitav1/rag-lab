from pathlib import Path

from sources.base import Base


class SystemDesignPrimerSource(Base):
    reader = "system-design-primer"

    def tags_for(self, rel_path):
        parts = Path(rel_path).parts
        return ["system-design", parts[2]] if parts[0] == "solutions" else ["system-design"]
