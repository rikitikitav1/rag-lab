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
# a book as the raw folder names it too: «kafka.pdf-bb367b09.md», «ref.pdf_251-268-93d626e8.md»
_BOOK_NAME = re.compile(r"\.(pdf|epub)(?![a-z0-9])", re.I)
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


_INDEX = re.compile(r"(предметный |алфавитный )?указатель|index( of [\w ]+)?", re.I)
# the letter groups an index is split into: «D», «A-B», «Symbols», «Цифры»
_INDEX_LETTERS = re.compile(r"\W*(\w|\w\s*[-–]\s*\w|symbols|numbers|numerics|digits|символы|цифры)\W*", re.I)


def _is_index(text: str) -> bool:
    return bool(_INDEX.fullmatch(_LEADING_NUMBER.sub("", text).strip(" .:")))


# a contents or index line: a dot leader and the page it points to, in a table row too
_LEADER_LINE = re.compile(r"(\.\s?){4,}\s*\d{1,4}(\s*\|)?\s*$|(\.{4,}|(\.\s){4,})\s*\d{1,4}\b")
_HEADING_LINE = re.compile(r"^#{1,6}\s")
_PAGE_ROW = re.compile(r"^\s*\|.*\d{1,4}\s*\|?\s*$")
# prose lines a run of leader lines bridges: a part title or a wrapped entry, not a paragraph
_PROSE_BRIDGED = 3


# contents and indexes by their lines, not their heading: a converter drops the heading or makes a contents row one
def leader_spans(lines: list[str], runs_from: int) -> list[tuple[int, int]]:
    if runs_from <= 0:
        return []
    leader = [bool(_LEADER_LINE.search(line)) for line in lines]
    rows, k = [], 0
    while k < len(lines):
        if not leader[k]:
            k += 1
            continue
        last, count, prose, run = k, 0, 0, []
        for j in range(k, len(lines)):
            if leader[j] or _PAGE_ROW.match(lines[j]):
                run.append(j)
                if leader[j]:
                    last, count, prose = j, count + 1, 0
            elif lines[j].strip() and not _HEADING_LINE.match(lines[j]):
                prose += 1
                if prose > _PROSE_BRIDGED:
                    break
        # only the contents rows go: a book that interleaves its preface with its contents keeps the preface
        if count >= runs_from:
            rows += [j for j in run if j <= last]
        k = last + 1
    return _ranges(rows)


def _ranges(rows: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for row in rows:
        if spans and spans[-1][1] == row:
            spans[-1] = (spans[-1][0], row + 1)
        else:
            spans.append((row, row + 1))
    return spans


# a back index stands in the book's last quarter: a LaTeX manual's «12.5. Предметный указатель» is a chapter on indexes
BACK_SHARE = 0.75


# a back index runs from its heading to the next one of its level or above that is neither a letter group nor matter
def index_spans(file: str, headings: list[tuple[int, int, str]], line_count: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    # a docs page headed «Index» is about indexes, and the file's first heading is its root
    if not _BOOK_NAME.search(file.rsplit("/", 1)[-1]):
        return spans
    for n, (line, level, text) in enumerate(headings):
        if n == 0 or line < BACK_SHARE * line_count or not _is_index(text) or any(a <= line < b for a, b in spans):
            continue
        end = next(
            (
                at
                for at, at_level, after in headings[n + 1 :]
                if at_level <= level and not _INDEX_LETTERS.fullmatch(after) and not is_book_matter(after)
            ),
            line_count,
        )
        spans.append((line, end))
    return spans
