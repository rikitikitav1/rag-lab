from dataclasses import dataclass

import outcomes
from evals import pools

# a column named in a preregistration and nowhere else is a column nobody can compute
SCHEMA = 2


# which way is worse belongs to the declaration that uses a column, not to the column
@dataclass(frozen=True)
class Column:
    reads: str
    says: str


def _outcome(value: str):
    return lambda ql: 1.0 if pools.outcome(ql) == value else 0.0


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
}

# the outcome and the ceiling are read through the pools, which re-derive them for old rows
_READERS = {
    "hops_exhausted": lambda ql: 1.0 if pools.hops_exhausted(ql) else 0.0,
    **{name: _outcome(name) for name, col in REGISTRY.items() if col.reads == "outcome"},
}


def known() -> list[str]:
    return sorted(REGISTRY)


# the reason the registry exists: a predicate named in prose has two readings, a name here has one
def unknown(names) -> list[str]:
    return sorted({str(name) for name in names if name not in REGISTRY})


def read(name: str, ql) -> float:
    if name not in _READERS:
        raise KeyError(f"no column named {name!r}; known: {', '.join(known())}")
    return _READERS[name](ql)


# a column of several names at once, because a closing predicate is often a union of outcomes
def read_any(names, ql) -> float:
    return 1.0 if any(read(name, ql) for name in names) else 0.0


def check_outcomes_are_all_registered() -> list[str]:
    # an outcome the stand can write and the registry cannot name would be unmeasurable
    named = {c for c, col in REGISTRY.items() if col.reads == "outcome"}
    return sorted({str(o) for o in outcomes.Outcome} - named)
