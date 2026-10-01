import re

from corpus_keys import leaf_of

# a book's front and back matter: what a reader skips, a search should not return and a question should not ask about
BOOK_MATTER = re.compile(
    r"(brief |table of )?contents|index|acknowledge?ments|about the authors?|colophon|bibliography|references"
    r"|preface|foreword|about this book|about the cover( illustration)?"
    r"|содержание|оглавление|(предметный |алфавитный )?указатель|благодарности|об авторах?|список литературы"
    r"|литература|предисловие|об этой книге|о книге|от издательства|(иллюстрация|об иллюстрации) на обложке"
    r"|о (научном )?редакторе.*|о рецензентах?.*|о переводчике.*",
    re.I,
)
# a contents line a converter took for a heading: its dot leader and the page it points to
_CONTENTS_LINE = re.compile(r"\.{5,}\s*\d+\s*$")


# the number a converter or a file split puts before a heading: «8 Оглавление», «126 - 05: Index»
_LEADING_NUMBER = re.compile(r"^[\d\s.:\-–]+(?=\D)")


# the leaf alone is read, so a chapter named «Index» is left out too
def is_book_matter(section: str | None) -> bool:
    if not section:
        return False
    leaf = _LEADING_NUMBER.sub("", leaf_of(section).strip())
    return bool(BOOK_MATTER.fullmatch(leaf.strip(" .:")) or _CONTENTS_LINE.search(leaf))


_BOOK_FILE = re.compile(r"\.(pdf|epub)$", re.I)
_NOT_A_WORD = re.compile(r"[\W_]+")


def _words(text: str) -> str:
    return _NOT_A_WORD.sub(" ", text).strip().casefold()


# a book's title page is headed by the title its file is named after; a markdown project's first section is text
def is_title_page(source: str, section: str | None) -> bool:
    if not section or not _BOOK_FILE.search(source):
        return False
    leaf = _words(leaf_of(section)).split()
    named = set(_words(_BOOK_FILE.sub("", source.rsplit("/", 1)[-1])).split())
    # two words at least: a chapter named «Redis» in redis-in-action.pdf is text
    return len(leaf) >= 2 and set(leaf) <= named


# the one test the index and the suitability report read: a section that is the book's matter, not its text
def is_matter(file: str, section: str | None) -> bool:
    return is_book_matter(section) or is_title_page(file, section)
