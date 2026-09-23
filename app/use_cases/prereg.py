from evals import columns
from evals.loaders import load_logs
from evals.pools import Ambiguous, by_question
from evals.stats import bootstrap_ci
from models.eval import Question
from models.prereg import Preregistration
from orm.sync_db import Session
from sqlalchemy import select

SCHEMA = 2

# which way the arm is meant to move the closing columns, and which way a guard must not move
ARM_SHOULD = ("lower", "raise")
MUST_NOT = ("rise", "fall")


class Refused(ValueError):
    pass


def _known_or_refuse(names, where: str) -> list:
    names = list(names or [])
    if not names:
        raise Refused(f"{where}: name at least one column; known: {', '.join(columns.known())}")
    if not all(isinstance(name, str) for name in names):
        raise Refused(f"{where}: a column is named by a string; known: {', '.join(columns.known())}")
    unknown = columns.unknown(names)
    if unknown:
        raise Refused(
            f"{where}: no column named {', '.join(unknown)}."
            f" Known: {', '.join(columns.known())}."
            " A column the registry cannot name is one nobody can recompute"
        )
    return names


def _one_of_or_refuse(value, allowed: tuple, where: str):
    if value not in allowed:
        raise Refused(f"{where}: one of {', '.join(allowed)}, got {value!r}")
    return value


def _sets_or_refuse(session, names, where: str = "population") -> list:
    names = list(names or [])
    if not names:
        raise Refused(f"{where}: name at least one question set")
    here = set(session.scalars(select(Question.set_name).where(Question.set_name.in_(names))))
    missing = sorted(set(names) - here)
    if missing:
        raise Refused(f"{where}: no question set named {', '.join(missing)}")
    return names


def _guard_or_refuse(session, guard) -> dict:
    if not isinstance(guard, dict):
        raise Refused(f"guards: each guard is an object with `column` and `must_not`, got {guard!r}")
    _known_or_refuse([guard.get("column")], "guards")
    _one_of_or_refuse(guard.get("must_not"), MUST_NOT, "guards.must_not")
    margin = guard.get("margin", 0.0)
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin < 0:
        raise Refused(f"guards.margin: a share no smaller than zero, got {margin!r}")
    out = {**guard, "margin": float(margin)}
    if "sets" in guard:
        out["sets"] = _sets_or_refuse(session, guard["sets"], "guards.sets")
    return out


def exists(name: str) -> bool:
    with Session() as session:
        return session.scalar(select(Preregistration.id).where(Preregistration.name == name)) is not None


# written before any row of the run exists, which is the whole point of the door
def write(name: str, population: dict, arms: dict, closing: dict, guards: list, declared: dict):
    with Session() as session:
        if session.scalar(select(Preregistration).where(Preregistration.name == name)):
            raise Refused(f"{name!r} is already preregistered; a promise is not edited after the fact")
        sets = _sets_or_refuse(session, (population or {}).get("sets"))
        cols = _known_or_refuse((closing or {}).get("columns"), "closing")
        _one_of_or_refuse((closing or {}).get("arm_should"), ARM_SHOULD, "closing.arm_should")
        checked = [_guard_or_refuse(session, guard) for guard in guards or []]
        if not (arms or {}).get("control") or not (arms or {}).get("arm"):
            raise Refused("arms: name both `control` and `arm`")
        row = Preregistration(
            name=name,
            population={**(population or {}), "sets": sets},
            arms=arms,
            closing={**(closing or {}), "columns": cols},
            guards=checked,
            declared=declared or {},
        )
        session.add(row)
        session.commit()
        return read(name)


def read(name: str) -> dict:
    with Session() as session:
        row = session.scalar(select(Preregistration).where(Preregistration.name == name))
        if row is None:
            raise Refused(f"no preregistration named {name!r}")
        return {
            "schema": SCHEMA, "name": row.name, "created_at": str(row.created_at),
            "population": row.population, "arms": row.arms, "closing": row.closing,
            "guards": row.guards, "declared": row.declared, "closed_with": row.closed_with,
        }


def _question_ids(sets: list) -> set:
    with Session() as session:
        return set(session.scalars(select(Question.id).where(Question.set_name.in_(sets))))


# the same pairing the other doors use, so two rows on one question refuse instead of one winning
def _rows(run_name: str) -> dict:
    try:
        return by_question(load_logs(run_name))
    except Ambiguous as e:
        raise Refused(str(e)) from e


def _paired(a: dict, b: dict, ids: set, cols: list, sign: int) -> list:
    shared = sorted(set(a) & set(b) & ids)
    return [sign * (columns.read_any(cols, b[q]) - columns.read_any(cols, a[q])) for q in shared]


def _band(deltas: list) -> dict:
    low, high = bootstrap_ci(deltas)
    return {"n": len(deltas), "mean": round(sum(deltas) / len(deltas), 4),
            "ci95": [round(low, 4), round(high, 4)]}


def _band_or_refuse(deltas: list, what: str) -> dict:
    if not deltas:
        raise Refused(f"{what}: no question is shared by both runs on the declared population")
    return _band(deltas)


def _guard(guard: dict, rows: dict, ids: set) -> dict:
    must_not = guard.get("must_not")
    if must_not not in MUST_NOT:
        # a promise written before guards declared a direction cannot be read one way or the other
        return {"column": guard.get("column"), "state": "unreadable",
                "why": "the guard declares no `must_not`, so which way is worse was never said"}
    sign = 1 if must_not == "rise" else -1
    ids = _question_ids(guard["sets"]) if guard.get("sets") else ids
    column = [guard["column"]]
    band = _band_or_refuse(
        _paired(rows["control"], rows["arm"], ids, column, sign), f"guard {guard['column']}"
    )
    margin = guard.get("margin", 0.0)
    out = {"column": guard["column"], "must_not": must_not, "margin": margin, "worsening": band}
    # the margin is added on top of the night's own noise, so a guard at margin 0 does not break on it
    bar = margin
    if "floor" in rows:
        out["floor"] = _band_or_refuse(
            _paired(rows["control"], rows["floor"], ids, column, sign), f"guard {guard['column']} floor"
        )
        bar = max(margin, out["floor"]["ci95"][1])
    low, high = band["ci95"]
    out["bar"] = bar
    out["state"] = "holds" if high <= bar else "broken" if low > bar else "undecided"
    return out


def _verdict(closing: str | None, guards: list) -> tuple:
    broken = [g["column"] for g in guards if g["state"] == "broken"]
    if closing == "missed" or broken:
        why = "the closing band stays inside the floor" if closing == "missed" else ""
        why += ("; " if why and broken else "") + (f"guard broken: {', '.join(broken)}" if broken else "")
        return False, why
    if closing in ("no_floor", "no_direction"):
        return None, {"no_floor": "no floor run was named, so the bar has nothing to clear",
                      "no_direction": "the promise declares no `arm_should`, so the effect has no sign"}[closing]
    open_ = [f"{g['column']} {g['state']}" for g in guards if g["state"] != "holds"]
    if open_:
        return None, f"the bar is cleared and a guard is not decided: {', '.join(open_)}"
    return True, "the bar is cleared and every guard holds"


def _closed_with(name: str, runs: dict, result: dict) -> dict:
    with Session() as session:
        row = session.scalar(select(Preregistration).where(Preregistration.name == name))
        # an undecided close is a look, not a closing, so it leaves the promise open
        if row.closed_with is None and result["cleared"] is not None:
            row.closed_with = {"runs": runs, "cleared": result["cleared"],
                               "cleared_because": result["cleared_because"]}
            session.commit()
        elif row.closed_with is not None and row.closed_with.get("runs") != runs:
            raise Refused(
                f"{name!r} was closed with {row.closed_with.get('runs')}; a promise closes once"
            )
        return row.closed_with


# the door computes only what was declared, and says so when asked for anything else
def close(name: str, runs: dict) -> dict:
    promise = read(name)
    closing = promise["closing"]
    cols = closing["columns"]
    for role in ("control", "arm"):
        if role not in runs:
            raise Refused(f"name the run for {role!r}; the preregistration declared both arms")
    ids = _question_ids(promise["population"]["sets"])
    rows = {role: _rows(run) for role, run in runs.items()}
    shared = sorted(set(rows["control"]) & set(rows["arm"]) & ids)
    if not shared:
        raise Refused("no question is shared by both arms on the declared population")
    arm_should = closing.get("arm_should")
    sign = -1 if arm_should == "lower" else 1
    effect = _band(_paired(rows["control"], rows["arm"], ids, cols, sign))
    out = {
        "schema": SCHEMA, "name": name, "columns": cols, "arm_should": arm_should, "n": len(shared),
        # the shares sit on the paired questions, the same population the effect is read on
        "shares": {role: round(sum(columns.read_any(cols, rows[role][q]) for q in shared) / len(shared), 4)
                   for role in ("control", "arm")},
        "effect": effect,
        "reads": "the effect is the arm's move on the closing columns in the declared direction,"
                 " paired by question; positive is what the arm promised" if arm_should in ARM_SHOULD
                 else "no direction was declared, so the effect is arm minus control, paired by question",
    }
    if arm_should not in ARM_SHOULD:
        closing_state = "no_direction"
    elif "floor" in runs:
        out["floor"] = _band_or_refuse(
            _paired(rows["control"], rows["floor"], ids, cols, sign), "floor"
        )
        # the declared bar: the lower edge of the effect above the upper edge of the floor
        closing_state = "cleared" if effect["ci95"][0] > out["floor"]["ci95"][1] else "missed"
    else:
        closing_state = "no_floor"
    out["guards"] = [_guard(guard, rows, ids) for guard in promise["guards"]]
    out["cleared"], out["cleared_because"] = _verdict(closing_state, out["guards"])
    out["closed_with"] = _closed_with(name, runs, out)
    return out
