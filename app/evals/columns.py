import json
from dataclasses import dataclass

import outcomes
from corpus_keys import reference_by, shares_heading_word
from evals import gold_classes, grade_curve, measurements, pools
from evals.stats import score_of


# the first step of a gold's file path names the source's row; its pattern comes from the caller, read once a run
def _reference_page(gold: dict | None, leaves: dict) -> float | None:
    if gold is None:
        return None
    leaf = leaves.get((gold.get("file") or "").split("/", 1)[0])
    return float(reference_by(gold.get("section"), leaf) is not None)


# a column named in a preregistration and nowhere else is a column nobody can compute
SCHEMA = 3

# where a column's rows come from: the question logs of a run, or the rows of a recorded measurement
RUN, MEASUREMENT = "run", "measurement"
# a share is 0 or 1 a row and several can be joined; a score is a judge's number and stands alone
SHARE, SCORE = "share", "score"

CANDIDATES = measurements.FOLDER.parent / "candidates"
# the serving path: the top of the fusion, the arm every grader number of the stand is read on
ARM, TOP = "A", 5


# an identifier this rare in its source is what search finds by its token alone; the threshold is read here, not stored
ANCHOR_RARE = 2


def anchored(anchors: dict | None) -> bool:
    return any(held <= ANCHOR_RARE for held in (anchors or {}).values())


# which way is worse belongs to the declaration that uses a column, not to the column
@dataclass(frozen=True)
class Column:
    reads: str
    says: str
    source: str = RUN
    kind: str = SHARE


def _outcome(value: str):
    return lambda ql: 1.0 if pools.outcome(ql) == value else 0.0


def _judged(axis: str):
    return lambda ql: None if getattr(ql, axis) is None else float(score_of(getattr(ql, axis)))


def _grader(name: str):
    def read(row: dict):
        seats = grade_curve.shares(row["classes"], row["addresses"], row["arm"], set(row["kept"]))
        return seats[name]
    return read


# the characters read every chunk the strip was asked about (`kept_before_strip`), a fully cut one included
def _chars_removed(cls: str):
    def read(row: dict):
        chars = row.get("chars")
        if not chars:
            return None
        asked = row.get("kept_before_strip", row["kept"])
        # a chunk of fewer than two strips was decided by the chunk pass, and the strip never touched it
        touched = row.get("strips") or [2] * len(row["addresses"])
        seats = [n for n, address in enumerate(row["addresses"])
                 if address in row["arm"] and n in asked and row["classes"][n] == cls and touched[n] >= 2]
        before = sum(chars["before"][n] for n in seats)
        return (before - sum(chars["after"][n] for n in seats)) / before if before else None
    return read


# every column a run can be closed or guarded on, with the one reading each name has
REGISTRY: dict[str, Column] = {
    # two names, because the stand writes "exhausted" in two places and they are not the same row set
    "hops_exhausted": Column(
        reads="finished_by",
        says="the loop reached the hop ceiling and the forced final wrote the answer",
    ),
    "exhausted": Column(
        reads="outcome", says="the row was classified as having spent its hops with nothing to show"
    ),
    "answered_ungrounded": Column(
        reads="outcome", says="sources came back and nothing in the answer used them"
    ),
    "narrated_call": Column(reads="outcome", says="the answer describes a tool call instead of issuing one"),
    "refused": Column(reads="outcome", says="the row refused"),
    "unsupported_answer": Column(reads="outcome", says="an answer with no source behind it"),
    "answered": Column(reads="outcome", says="the row answered with sources"),
    "error": Column(reads="outcome", says="the row failed"),
    **{axis: Column(reads="judge", kind=SCORE,
                    says=f"our judge's {axis} score, 0 to 10; a row the judge abstained on has none")
       for axis in ("faithfulness", "relevance", "completeness")},
    "gold_retained": Column(
        reads="grader", source=MEASUREMENT,
        says="the grader kept at least one chunk of the gold section; rows whose gold never reached"
             " the arm have none",
    ),
    "strangers_dropped": Column(
        reads="grader", source=MEASUREMENT,
        says="the share of the row's stranger chunks the grader dropped; rows without a stranger have none",
    ),
    **{f"chars_removed_{short}": Column(
        reads="strip", source=MEASUREMENT,
        says=f"the share of characters the strip removed inside the {short} chunks the grader kept,"
             " a row's own share, averaged over rows rather than pooled over characters")
       for short in ("gold", "neighbour", "stranger")},
    "reader_answered": Column(
        reads="acceptance", source=MEASUREMENT,
        says="the reader quoted the section rather than saying the answer is not in it; an empty or cut reply is no",
    ),
    "evidence_held": Column(
        reads="acceptance", source=MEASUREMENT,
        says="the reader's quote holds the evidence's words to the sieve's share; rows it did not answer have none",
    ),
    "anchored_by_identifier": Column(
        reads="question",
        says="the question names an identifier its gold section holds, and at most ANCHOR_RARE sections of its source",
    ),
    "shares_heading_word": Column(
        reads="question",
        says="the question repeats a word of its gold section's heading, five letters or longer, by a five-letter stem",
    ),
    "reference_page": Column(
        reads="question",
        says="the question's gold section is a reference page: its leaf names one identifier, or matches its source's"
             " own `reference_leaf`",
    ),
    "left_for_judge": Column(
        reads="acceptance", source=MEASUREMENT,
        says="the question's pair was neither accepted nor refused by the reader and waits for the judge",
    ),
}

_CLASS_OF = {"gold": gold_classes.GOLD, "neighbour": gold_classes.NEIGHBOUR,
             "stranger": gold_classes.STRANGER}

# the outcome and the ceiling are read through the pools, which re-derive them for old rows
_READERS = {
    "hops_exhausted": lambda ql: 1.0 if pools.hops_exhausted(ql) else 0.0,
    **{name: _outcome(name) for name, col in REGISTRY.items() if col.reads == "outcome"},
    **{name: _judged(name) for name, col in REGISTRY.items() if col.reads == "judge"},
    "gold_retained": _grader("gold_any"),
    "strangers_dropped": _grader("strangers_dropped"),
    **{f"chars_removed_{short}": _chars_removed(cls) for short, cls in _CLASS_OF.items()},
    "reader_answered": lambda row: 1.0 if row["answerable"] else 0.0,
    "evidence_held": lambda row: None if not row["answerable"] else float(row["why"] is None),
    "left_for_judge": lambda row: float(row["outcome"] == "undecided"),
    "shares_heading_word": lambda ql: (
        None if getattr(ql.question, "gold", None) is None
        else float(shares_heading_word(ql.question.original_text, ql.question.gold.get("section")))
    ),
    # a question whose anchors were never read has no value, not a zero
    "anchored_by_identifier": lambda ql: (
        None if getattr(ql.question, "anchors", None) is None else float(anchored(ql.question.anchors))
    ),
}


def known() -> list[str]:
    return sorted(REGISTRY)


# the reason the registry exists: a predicate named in prose has two readings, a name here has one
def unknown(names) -> list[str]:
    return sorted({str(name) for name in names if name not in REGISTRY})


def source_of(names) -> set:
    return {REGISTRY[name].source for name in names}


# columns that read what the sources declare: the caller reads it once and passes it, so a column stays a function
_WITH_SOURCES = {"reference_page": lambda ql, leaves: _reference_page(getattr(ql.question, "gold", None), leaves)}


def needs_sources(name: str) -> bool:
    return name in _WITH_SOURCES


def read(name: str, row, leaves: dict | None = None) -> float | None:
    if name in _WITH_SOURCES:
        return _WITH_SOURCES[name](row, leaves or {})
    if name not in _READERS:
        raise KeyError(f"no column named {name!r}; known: {', '.join(known())}")
    return _READERS[name](row)


# a union joins shares; one score is read as it is, and a row with no value is left out of a pairing
def value(names, row) -> float | None:
    names = list(names)
    if len(names) == 1:
        return read(names[0], row)
    said = [read(name, row) for name in names]
    # a row where no member has a value is not a row where every member said no
    if all(v is None for v in said):
        return None
    return 1.0 if any(said) else 0.0


def read_any(names, row) -> float:
    return 1.0 if any(read(name, row) for name in names) else 0.0


# measurement rows by question, narrowed to the serving arm the way the grader curve reads it
def measurement_rows(name: str) -> list[dict]:
    path = measurements.FOLDER / name
    if "/" in name or not name.endswith(".json") or not path.exists():
        raise FileNotFoundError(f"no measurement named {name!r} in {measurements.FOLDER.name}")
    payload = json.loads(path.read_text())
    # an acceptance pass keys its rows by question already, with no arm to narrow to
    if "evidence_held" in payload:
        return measurements.rows_of(path)
    if "candidates_file" not in payload:
        raise FileNotFoundError(f"{name!r} is not a grader measurement: it names no candidates file")
    frozen = json.loads((CANDIDATES / payload["candidates_file"]).read_text())
    by_id = {row["id"]: row for row in frozen["rows"]}
    rows = measurements.rows_of(path)
    return [{**row, "arm": grade_curve.arm_of(by_id[row["question_id"]], TOP, ARM)}
            for row in rows if row["question_id"] in by_id]


def check_outcomes_are_all_registered() -> list[str]:
    # an outcome the stand can write and the registry cannot name would be unmeasurable
    named = {c for c, col in REGISTRY.items() if col.reads == "outcome"}
    return sorted({str(o) for o in outcomes.Outcome} - named)
