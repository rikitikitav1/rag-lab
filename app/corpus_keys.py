import hashlib
import re
from dataclasses import dataclass

# keys the corpus is stored and found by, and the gold rule over them with its sql twin; no import of their own


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


_SLUG = re.compile(r"[^\w.-]+")


# a file or piece key as a flat file name, readable and unique: two keys that slug alike differ by the hash
def file_stem(key: str) -> str:
    return f"{_SLUG.sub('_', key)[:120]}-{short_hash(key)}"


# the name reaches DDL as a literal; 22 chars of prefix + 36 + 4 of suffix fit in 63
VARIANT_RE = re.compile(r"^[a-z0-9_]{1,36}$")


# fullmatch: `$` admits a trailing newline, and the name reaches DDL twice
def check_variant(name: str) -> str:
    if not VARIANT_RE.fullmatch(name or ""):
        raise ValueError(f"corpus variant '{name}' must match {VARIANT_RE.pattern}")
    return name


VECTOR_INDEX_PREFIX = "data_chunks_embedding_"


def vector_index_name(variant: str) -> str:
    return f"{VECTOR_INDEX_PREFIX}{check_variant(variant)}_idx"


def body_hash(body: str) -> str:
    normalised = re.sub(r"\s+", " ", body).strip().encode()
    # a content fingerprint for deduplication, never a credential
    return hashlib.md5(normalised, usedforsecurity=False).hexdigest()


# an older mark is a path fragment a chunk's file contains; `Gold.mark_of` is the same test in python
GOLD_SQL = "position({mark} in {source}) > 0"
# a questions row carries a gold of either kind
HAS_GOLD_SQL = "(cardinality({q}.marked_sources) > 0 OR {q}.gold IS NOT NULL)"
# a run reads a question once it is accepted; `models.eval.READ_BY_RUNS` is the same test in the orm
READ_BY_RUNS_SQL = "{q}.status = 'accepted'"

SECTION_SEP = " > "
# a chapter is a section path's first two steps, the grain the coverage report reads and a question set is spread over
CHAPTER_STEPS = 2
# the variant a veto build cuts its headings from when its job names none; with no variants it reads baseline too
VETO_CUT_FROM = "clean_1024"


def chapter_of(section: str | None) -> str | None:
    return SECTION_SEP.join((section or "").split(SECTION_SEP)[:CHAPTER_STEPS]) or None


# a section path under another: itself or one of its sub-sections, never a parent or a sibling sharing a prefix
def section_under(section: str | None, gold: str) -> bool:
    return section == gold or (section or "").startswith(gold + SECTION_SEP)


# a question's gold: a file, its section path and version; an older question has path fragments only, its marks
@dataclass(frozen=True)
class Gold:
    marks: tuple[str, ...]
    section: str | None = None
    version: str | None = None

    # marks alone are the older gold, as the candidate rows and the tests still hand them
    @classmethod
    def coerce(cls, gold) -> "Gold":
        return gold if isinstance(gold, Gold) else cls(tuple(gold or ()))

    @classmethod
    def of(cls, marked_sources, gold: dict | None) -> "Gold | None":
        if gold:
            # a gold with no section would read as a containment mark, the older rule, with nobody told
            if not gold.get("file") or not gold.get("section"):
                raise ValueError(f"an exact gold names its file and section: {gold}")
            return cls((gold["file"],), gold["section"], gold.get("version"))
        return cls(tuple(marked_sources)) if marked_sources else None

    # the one reading of a question's gold every door shares, from a row or a stand-in: exact, older marks, or none
    @classmethod
    def of_question(cls, question) -> "Gold | None":
        if question is None:
            return None
        return cls.of(getattr(question, "marked_sources", None), getattr(question, "gold", None))

    # an exact gold names one file and one section; the older one names fragments and leaves the heading to its text
    @property
    def exact(self) -> bool:
        return self.section is not None

    # the mark a retrieved file answers to, or None: an exact gold's file is the file itself, not a part of it
    def mark_of(self, source: str) -> str | None:
        if self.exact:
            return self.marks[0] if source == self.marks[0] else None
        return next((m for m in self.marks if m in source), None)

    def holds_file(self, source: str) -> bool:
        return self.mark_of(source) is not None

    # a chunk of no version holds for every version, as search reads it
    def holds_section(self, source: str, section: str | None, versions=None) -> bool:
        return (
            self.exact
            and self.holds_file(source)
            and section_under(section, self.section)
            and (self.version is None or not versions or self.version in versions)
        )

    # the same exact test inside a query, on a chunk's own columns; one test, its python and its sql read alike
    def section_sql(self, source: str, section: str, versions: str) -> tuple[str, dict]:
        clause = exact_gold_sql(":gold_file", ":gold_section", "CAST(:gold_version AS text)", source, section, versions)
        return clause, {"gold_file": self.marks[0], "gold_section": self.section, "gold_version": self.version}


# the exact gold test as sql over any expressions: bound values for one question, a question's own columns for many
def exact_gold_sql(file: str, gold_section: str, version: str, source: str, section: str, versions: str) -> str:
    under = f"{gold_section} || '{SECTION_SEP}'"
    return (
        f"{source} = {file} AND ({section} = {gold_section} OR left({section}, length({under})) = {under})"
        f" AND ({version} IS NULL"
        f" OR cardinality({versions}) = 0 OR {version} = ANY({versions}))"
    )


# the share of Cyrillic among a text's letters from which it reads as Russian; a Russian text carries Latin terms
CYRILLIC_SHARE = 0.2


# one rule for a question, a chunk and a source: the stand has two languages, and their alphabets tell them apart
def language_by_alphabet(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    cyrillic = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    return "ru" if letters and cyrillic / len(letters) >= CYRILLIC_SHARE else "en"


_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


# a link read by its words, without the address: how a model quotes it
def unlinked(text: str) -> str:
    return _LINK.sub(r"\1", text)


# a section path's last step, the one that names the section itself
def leaf_of(section: str | None) -> str:
    return (section or "").split(SECTION_SEP)[-1]


# a quote is a text's words in their order: a table's cells, list dashes, escapes and emphasis around them are layout
_LAYOUT = re.compile(r"[\s*_`|\\-]")


def spaceless_key(text: str) -> str:
    return _LAYOUT.sub("", unlinked(text)).casefold()


_BACKTICKED = re.compile(r"`([^`\n]+)`")
_TOKEN = re.compile(r"[\w.()\-]+")
# snake_case, module.name, call(), --option, a digit among letters; a product name in camelCase is not code
_CODE_SHAPES = (
    re.compile(r"[A-Za-z0-9]_[A-Za-z0-9]"),
    re.compile(r"[A-Za-z]\.[A-Za-z]"),
    re.compile(r"\w\(\)$"),
    re.compile(r"^--[A-Za-z]"),
    re.compile(r"[A-Za-z]\d|\d[A-Za-z]"),
)
# a token that names a thing of code by its shape, a case change inside too; backticks are the generator's habit
_SHAPES = (*_CODE_SHAPES, re.compile(r"[a-z][A-Z]"))


def identifiers(question: str) -> list[str]:
    found = [t.strip() for t in _BACKTICKED.findall(question)]
    for token in _TOKEN.findall(_BACKTICKED.sub(" ", question)):
        token = token.rstrip(".-").lstrip(".")
        if any(shape.search(token) for shape in _SHAPES):
            found.append(token)
    return list(dict.fromkeys(t for t in found if t))


# the identifiers of a question its gold section holds, each with the number of the source's sections that hold it
def anchors(question: str, gold_key: str, section_keys: list[str]) -> dict[str, int]:
    out = {}
    for token in identifiers(question):
        key = spaceless_key(token)
        if len(key) > 1 and key in gold_key:
            out[token] = sum(1 for k in section_keys if key in k)
    return out


_WORDS = re.compile(r"[^\W\d_]{5,}")


# a reference page: its leaf names one identifier by a code shape, or matches the source's own pattern
def reference_by(section: str | None, knob: str | None = None) -> str | None:
    leaf = leaf_of(section).strip().strip("`").rstrip(":")
    words = leaf.split()
    if words and len(words) <= 3 and any(shape.search(words[0].strip("`")) for shape in _CODE_SHAPES):
        return "heading"
    if knob and re.search(knob, leaf):
        return "knob"
    return None


# a question that repeats a word of its section's heading is found by the heading; five letters and a five-letter stem
def shares_heading_word(question: str, section: str | None) -> bool:
    if not section:
        return False
    stems = {w.casefold()[:5] for w in _WORDS.findall(leaf_of(section))}
    return any(w.casefold()[:5] in stems for w in _WORDS.findall(question))
