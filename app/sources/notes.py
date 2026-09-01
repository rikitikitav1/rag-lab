from sources.base import Base


class NotesSource(Base):
    name = "notes"
    language = "rus"
    path = "/notes"
    # a hub of links answers nothing, and filtering it in search decided composition twice
    HYGIENIC_SKIP_FILE_NAMES = frozenset({"index"})
