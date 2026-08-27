from sources.base import Base


class NotesSource(Base):
    name = "notes"
    language = "rus"
    path = "/notes"
    # a hub of links to other notes, it answers nothing on its own. Used to be filtered
    # on the search side, which left the corpus composition decided in two places
    HYGIENIC_SKIP_FILE_NAMES = frozenset({"index"})
