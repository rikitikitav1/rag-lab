from evals import columns
from evals.loaders import load_logs
from evals.pools import Ambiguous, by_question
from evals.stats import bootstrap_ci
from models.eval import Question
from models.prereg import Preregistration
from orm.sync_db import Session
from sqlalchemy import select

SCHEMA = 3

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
    out = {**guard, "margin": _margin_or_refuse(guard.get("margin", 0.0), "guards.margin")}
    if "sets" in guard:
        out["sets"] = _sets_or_refuse(session, guard["sets"], "guards.sets")
    return out


def _margin_or_refuse(margin, where: str) -> float:
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin < 0:
        raise Refused(f"{where}: a number no smaller than zero, got {margin!r}")
    return float(margin)


# a veto compares two columns inside one arm by point: the arm must not cut `column` more than `above`
def _veto_or_refuse(veto) -> dict:
    if not isinstance(veto, dict):
        raise Refused(f"vetoes: each veto is an object with `column`, `above` and `margin`, got {veto!r}")
    pair = _known_or_refuse([veto.get("column"), veto.get("above")], "vetoes")
    if len(columns.source_of(pair)) > 1:
        raise Refused("vetoes: `column` and `above` are read from one source, a run or a measurement")
    rows = veto.get("min_rows")
    # declared, so a smoke read through the door cannot fire a veto meant for the full pass
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise Refused(f"vetoes.min_rows: the rows a veto is read on, a positive integer, got {rows!r}")
    return {**veto, "margin": _margin_or_refuse(veto.get("margin", 0.0), "vetoes.margin"),
            "on": _one_of_or_refuse(veto.get("on", "arm"), ("arm", "control"), "vetoes.on")}


def _closing_or_refuse(closing: dict) -> dict:
    cols = _known_or_refuse(closing.get("columns"), "closing")
    if len(columns.source_of(cols)) > 1:
        raise Refused("closing: the columns are read from one source, a run or a measurement")
    if len(cols) > 1 and any(columns.REGISTRY[c].kind == columns.SCORE for c in cols):
        raise Refused("closing: a judge score closes alone; only shares join into a union")
    _one_of_or_refuse(closing.get("arm_should"), ARM_SHOULD, "closing.arm_should")
    out = {**closing, "columns": cols}
    if "floor_value" in closing:
        out["floor_value"] = _margin_or_refuse(closing["floor_value"], "closing.floor_value")
    return out


def exists(name: str) -> bool:
    with Session() as session:
        return session.scalar(select(Preregistration.id).where(Preregistration.name == name)) is not None


# written before any row of the run exists, which is the whole point of the door
def write(name: str, population: dict, arms: dict, closing: dict, guards: list, declared: dict,
          vetoes: list | None = None):
    with Session() as session:
        if session.scalar(select(Preregistration).where(Preregistration.name == name)):
            raise Refused(f"{name!r} is already preregistered; a promise is not edited after the fact")
        sets = _sets_or_refuse(session, (population or {}).get("sets"))
        named = (population or {}).get("question_ids")
        if named is not None:
            if not isinstance(named, list) or not all(isinstance(q, int) and not isinstance(q, bool)
                                                      for q in named):
                raise Refused("population.question_ids: a list of question ids")
            here = set(session.scalars(select(Question.id).where(
                Question.set_name.in_(sets), Question.id.in_(named))))
            if len(here) != len(set(named)):
                raise Refused(f"population.question_ids: {len(set(named) - here)} are not in {sets}")
        closing = _closing_or_refuse(closing or {})
        checked = [_guard_or_refuse(session, guard) for guard in guards or []]
        vetoed = [_veto_or_refuse(veto) for veto in vetoes or []]
        if not (arms or {}).get("control") or not (arms or {}).get("arm"):
            raise Refused("arms: name both `control` and `arm`")
        row = Preregistration(
            name=name,
            population={**(population or {}), "sets": sets},
            arms=arms,
            closing=closing,
            guards=checked,
            vetoes=vetoed,
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
            "guards": row.guards, "vetoes": row.vetoes, "declared": row.declared,
            "closed_with": row.closed_with,
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


def _measured(name: str) -> dict:
    try:
        rows = columns.measurement_rows(name)
    except FileNotFoundError as e:
        raise Refused(str(e)) from e
    out = {}
    for row in rows:
        if row["question_id"] in out:
            raise Refused(f"question {row['question_id']} appears twice in measurement {name!r}")
        out[row["question_id"]] = row
    return out


# a row with no value on the column (a judge that abstained, a gold that never reached the arm) is left out
def _pairs(a: dict, b: dict, ids: set, cols: list) -> list:
    out = []
    for q in sorted(set(a) & set(b) & ids):
        left, right = columns.value(cols, a[q]), columns.value(cols, b[q])
        if left is not None and right is not None:
            out.append((left, right))
    return out


def _paired(a: dict, b: dict, ids: set, cols: list, sign: int) -> list:
    return [sign * (right - left) for left, right in _pairs(a, b, ids, cols)]


def _band(deltas: list) -> dict:
    low, high = bootstrap_ci(deltas)
    return {"n": len(deltas), "mean": round(sum(deltas) / len(deltas), 4),
            "ci95": [round(low, 4), round(high, 4)]}


def _band_or_refuse(deltas: list, what: str) -> dict:
    if not deltas:
        raise Refused(f"{what}: no question is shared by both runs on the declared population")
    return _band(deltas)


def _source(rows: dict, cols: list) -> dict:
    return rows[next(iter(columns.source_of(cols)))]


def _guard(guard: dict, rows: dict, ids: set) -> dict:
    must_not = guard.get("must_not")
    if must_not not in MUST_NOT:
        # a promise written before guards declared a direction cannot be read one way or the other
        return {"column": guard.get("column"), "state": "unreadable",
                "why": "the guard declares no `must_not`, so which way is worse was never said"}
    column = [guard["column"]]
    by_role = _source(rows, column)
    if not {"control", "arm"} <= set(by_role):
        return {"column": guard["column"], "state": "not_run", "why": "both arms are not named yet"}
    sign = 1 if must_not == "rise" else -1
    ids = _question_ids(guard["sets"]) if guard.get("sets") else ids
    band = _band_or_refuse(
        _paired(by_role["control"], by_role["arm"], ids, column, sign), f"guard {guard['column']}"
    )
    margin = guard.get("margin", 0.0)
    out = {"column": guard["column"], "must_not": must_not, "margin": margin, "worsening": band}
    # the margin is added on top of the night's own noise, so a guard at margin 0 does not break on it
    bar = margin
    if "floor" in by_role:
        out["floor"] = _band_or_refuse(
            _paired(by_role["control"], by_role["floor"], ids, column, sign),
            f"guard {guard['column']} floor",
        )
        bar = max(margin, out["floor"]["ci95"][1])
    low, high = band["ci95"]
    out["bar"] = bar
    out["state"] = "holds" if high <= bar else "broken" if low > bar else "undecided"
    return out


def _mean(by_q: dict, ids: set, name: str) -> tuple:
    seen = [v for q, row in by_q.items() if q in ids and (v := columns.read(name, row)) is not None]
    return (round(sum(seen) / len(seen), 4) if seen else None), len(seen)


# read by point with its margin: a veto is a stop before the expensive part, not a closing number
def _veto(veto: dict, rows: dict, ids: set) -> dict:
    by_role = _source(rows, [veto["column"], veto["above"]])
    if veto["on"] not in by_role:
        return {**veto, "state": "not_run", "why": f"no {veto['on']} named for this source"}
    (cut, n_cut), (above, n_above) = (_mean(by_role[veto["on"]], ids, name)
                                       for name in (veto["column"], veto["above"]))
    out = {**veto, "points": {veto["column"]: cut, veto["above"]: above}, "n": [n_cut, n_above]}
    if cut is None or above is None:
        return out | {"state": "unreadable", "why": "one of the two columns has no value on these rows"}
    if min(n_cut, n_above) < veto.get("min_rows", 1):
        return out | {"state": "undecided", "why": f"read on fewer rows than the declared {veto['min_rows']}"}
    return out | {"state": "fired" if cut > above + veto["margin"] else "quiet"}


_OPEN = {
    "no_floor": "no floor was named, neither a floor run nor `floor_value`, so the bar has nothing to clear",
    "no_direction": "the promise declares no `arm_should`, so the effect has no sign",
    "not_run": "the closing arms are not named yet",
}


def _verdict(closing: str, guards: list, vetoes: list = ()) -> tuple:
    stops = [f"guard broken: {g['column']}" for g in guards if g["state"] == "broken"]
    stops += [f"veto fired: {v['column']} above {v['above']}" for v in vetoes if v["state"] == "fired"]
    if closing == "missed" or stops:
        return False, "; ".join((["the closing band does not clear its floor"] if closing == "missed" else [])
                                + stops)
    if closing in _OPEN:
        return None, _OPEN[closing]
    open_ = [f"{g['column']} {g['state']}" for g in guards if g["state"] != "holds"]
    open_ += [f"veto {v['column']} {v['state']}" for v in vetoes if v["state"] != "quiet"]
    if open_:
        return None, f"the bar is cleared and a check is not decided: {', '.join(open_)}"
    return True, "the bar is cleared, every guard holds and no veto fired"


# what a closing named; the first door to close with measurements nested both under `runs`
def _named(closed: dict) -> dict:
    said = closed.get("runs") or {}
    if "runs" in said or "measurements" in said:
        return {"runs": said.get("runs") or {}, "measurements": said.get("measurements") or {}}
    return {"runs": said, "measurements": closed.get("measurements") or {}}


def _closed_with(name: str, named: dict, result: dict) -> dict:
    with Session() as session:
        row = session.scalar(select(Preregistration).where(Preregistration.name == name))
        # an undecided close is a look, not a closing, so it leaves the promise open
        if row.closed_with is None and result["cleared"] is not None:
            row.closed_with = {**named, "cleared": result["cleared"],
                               "cleared_because": result["cleared_because"]}
            session.commit()
        elif row.closed_with is not None and _named(row.closed_with) != named:
            raise Refused(
                f"{name!r} was closed with {_named(row.closed_with)}; a promise closes once"
            )
        return row.closed_with


# the door computes only what was declared, and says so when asked for anything else
def close(name: str, runs: dict | None = None, measurements: dict | None = None) -> dict:
    promise = read(name)
    closing = promise["closing"]
    cols = closing["columns"]
    runs, measurements = runs or {}, measurements or {}
    for role in set(runs) | set(measurements):
        _one_of_or_refuse(role, ("control", "arm", "floor"), "runs")
    ids = _question_ids(promise["population"]["sets"])
    # a declared draw narrows the sets to the questions it named, and nothing else is read
    if promise["population"].get("question_ids") is not None:
        ids &= set(promise["population"]["question_ids"])
    rows = {columns.RUN: {role: _rows(run) for role, run in runs.items()},
            columns.MEASUREMENT: {role: _measured(m) for role, m in measurements.items()}}
    arm_should = closing.get("arm_should")
    sign = -1 if arm_should == "lower" else 1
    out = {"schema": SCHEMA, "name": name, "columns": cols, "arm_should": arm_should}
    by_role = _source(rows, cols)
    if {"control", "arm"} <= set(by_role):
        pairs = _pairs(by_role["control"], by_role["arm"], ids, cols)
        if not pairs:
            raise Refused("no question is shared by both arms on the declared population")
        out |= {
            "n": len(pairs),
            # the means sit on the paired questions, the same population the effect is read on
            "means": {"control": round(sum(c for c, _ in pairs) / len(pairs), 4),
                      "arm": round(sum(a for _, a in pairs) / len(pairs), 4)},
            "effect": _band([sign * (a - c) for c, a in pairs]),
            "reads": "the effect is the arm's move on the closing columns in the declared direction,"
                     " paired by question; positive is what the arm promised" if arm_should in ARM_SHOULD
                     else "no direction was declared, so the effect is arm minus control, paired by question",
        }
        if arm_should not in ARM_SHOULD:
            closing_state = "no_direction"
        elif "floor" in by_role or "floor_value" in closing:
            bar = closing.get("floor_value", float("-inf"))
            if "floor" in by_role:
                out["floor"] = _band_or_refuse(
                    _paired(by_role["control"], by_role["floor"], ids, cols, sign), "floor"
                )
                bar = max(bar, out["floor"]["ci95"][1])
            out["bar"] = bar
            # the declared bar: the lower edge of the effect above the floor's upper edge or its value
            closing_state = "cleared" if out["effect"]["ci95"][0] > bar else "missed"
        else:
            closing_state = "no_floor"
    else:
        closing_state = "not_run"
    out["guards"] = [_guard(guard, rows, ids) for guard in promise["guards"]]
    out["vetoes"] = [_veto(veto, rows, ids) for veto in promise.get("vetoes") or []]
    out["cleared"], out["cleared_because"] = _verdict(closing_state, out["guards"], out["vetoes"])
    out["closed_with"] = _closed_with(name, {"runs": runs, "measurements": measurements}, out)
    return out
