import hashlib
import itertools
import re

from models.eval import QuestionLog
from models.registry import MODEL_NAME_RE, Model, Prompt, Purpose, Status
from orm.sync_db import Session
from sqlalchemy import delete, func, insert, literal, select, text
from use_cases import judge, retrieval_compare
from use_cases.retrieval_compare import bootstrap_ci, half_of

# 1 had per-arm means and deltas against the first arm only
# 2 added `pairing`, every pair up to a cap, and the source read as an arm (`source_scored`)
# 3 added the judge that scored the source, because the source is compared like an arm
SCHEMA = 3

AXES = ("faithfulness", "relevance", "completeness")
# a copy is unjudged, so it must not carry the judge the original named: `models.judging`
# and `prompts.judge_*` are claims about a verdict this row does not have yet
JUDGE_MODEL_KEY = "judging"


def _stripped(column, keys):
    for key in keys:
        column = column.op("-")(literal(key))
    return column


# one run's answers under a new name, verdicts cleared, ready to be judged again
def copy_run(source: str, target: str) -> int:
    if not source or not target:
        raise ValueError("both the source run and the name of the copy are required")
    if source == target:
        raise ValueError("a copy under the same name would be judged as the original")

    refuse_oversized_fanout(source, 1)
    with Session() as session:
        _refuse_bad_pair(session, source, target)

        # RETURNING rather than rowcount: an INSERT ... SELECT reports -1 through this
        # driver, and a copy that cannot say how many rows it made is not a measurement
        copied = session.execute(copy_statement(source, target)).scalars().all()
        session.commit()
        return len(copied)


# all of them or none: a half-made fan-out commits its copies while the experiment row
# rolls back, and the names are then burned for the identical retry
def copy_runs(source: str, targets: list[str]) -> dict[str, int]:
    # the same refusal `copy_run` makes: a null source compiles to `run_name IS NULL` and
    # the copy then fans out over every orphan row in the table
    if not source or not targets or not all(targets):
        raise ValueError(f"a copy needs a source run and names, got {source!r} {targets}")
    if len(set(targets)) != len(targets):
        raise ValueError(f"the copies must have distinct names, got {targets}")
    refuse_oversized_fanout(source, len(targets))
    # a fixed order, not the caller's: two fan-outs sharing a name would otherwise take
    # their advisory locks in opposite orders and deadlock
    targets = sorted(targets)
    with Session() as session:
        for target in targets:
            _refuse_bad_pair(session, source, target)
        made = {
            target: len(session.execute(copy_statement(source, target)).scalars().all())
            for target in targets
        }
        session.commit()
        return made


# what a failure after the copies compensates with
def delete_runs(names: list[str]) -> int:
    if not names:
        return 0
    with Session() as session:
        done = session.execute(
            delete(QuestionLog).where(QuestionLog.run_name.in_(names))
        ).rowcount
        session.commit()
        return done


# the check below is a count then an insert, and the door is a sync handler FastAPI serves
# from a threadpool, so two identical requests can both find the name free. The lock is
# released by the commit or the rollback; a unique index would cost the whole schema a rule
def _claim(session, target: str) -> None:
    session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:name))"), {"name": target}
    )


def _refuse_bad_pair(session, source: str, target: str) -> None:
    _claim(session, target)
    # answered, not merely present: a run where nothing was answered copies cleanly, the
    # judge finds no targets, the series reads as complete on its first job, and the
    # report lands with n=0 everywhere and three digests over nothing agreeing
    if not session.scalar(
        select(func.count())
        .select_from(QuestionLog)
        .where(QuestionLog.run_name == source, QuestionLog.answered.is_(True))
    ):
        raise ValueError(f"run {source} has no answered rows to copy")
    if session.scalar(
        select(func.count()).select_from(QuestionLog).where(QuestionLog.run_name == target)
    ):
        raise ValueError(f"run {target} already has rows; a copy never merges into a name")


def carried_columns() -> list[str]:
    # taken from the model rather than listed by hand: a migration that adds a column
    # would otherwise produce a copy silently missing it
    return [c.name for c in QuestionLog.__table__.columns if c.name != "id"]


def copy_statement(source: str, target: str):
    carried = carried_columns()
    overrides = {
        "run_name": literal(target),
        "metrics": _stripped(QuestionLog.metrics, AXES),
        "prompts": _stripped(QuestionLog.prompts, [f"judge_{axis}" for axis in AXES]),
        "models": _stripped(QuestionLog.models, [JUDGE_MODEL_KEY]),
        **{axis: literal(None) for axis in AXES},
    }
    picked = select(
        *[overrides.get(name, getattr(QuestionLog, name)) for name in carried]
    ).where(QuestionLog.run_name == source)
    return insert(QuestionLog).from_select(carried, picked).returning(QuestionLog.id)


# what an arm may move; everything else is held by construction, the answers being copied
# rather than produced. `repeat` moves nothing and exists so the same run judged twice by
# the same judge has two arm names: its delta is the judge's own noise, not an effect
REPEAT = "repeat"
AXES_ALLOWED = (REPEAT, "judge_model", *[f"judge_{axis}" for axis in AXES])
# arms are capped by the grid, the work they make is not: one request over the 823-question
# sets can copy 26k rows and queue 79k judge calls. This caps what the fan-out may cost
MAX_ARM_ROWS = 4000
# what a `repeat` label may look like: it becomes part of a run name on every copied row
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


# what the copy will actually make, not what the judge will look at: the cap is about the
# rows written, and `copy_statement` copies every row of the source
def _source_rows(source: str) -> int:
    with Session() as session:
        return session.scalar(
            select(func.count())
            .select_from(QuestionLog)
            .where(QuestionLog.run_name == source)
        ) or 0


# `existing` is what the experiment already holds: the cap is a property of the experiment,
# not of one request, and counting only the new arms let a caller post them one at a time
# and reach the 26k rows the cap was written to refuse
def refuse_oversized_fanout(source: str, arm_count: int, existing: int = 0) -> None:
    rows = _source_rows(source)
    total = rows * (arm_count + existing)
    if total > MAX_ARM_ROWS:
        raise ValueError(
            f"{arm_count} new arms beside {existing} over {rows} rows is {total} rows to"
            f" judge, over the cap of {MAX_ARM_ROWS}: run fewer arms or a smaller source"
        )


# shape was checked and existence was not, the same hole the retrieval kind had with an
# undeclared variant: the arm dies per log, the job says done, the experiment waits for ever
def unseeded_prompt_versions(axes: dict) -> list[str]:
    missing = []
    with Session() as session:
        for name, versions in axes.items():
            if name in (REPEAT, "judge_model"):
                continue
            purpose = Purpose[name]
            known = set(
                session.scalars(
                    select(Prompt.version).where(Prompt.purpose == purpose)
                )
            )
            missing += [f"{name}=v{v}" for v in versions if v not in known]
    return sorted(missing)


def judges_not_ready(axes: dict) -> list[str]:
    named = axes.get("judge_model") or []
    if not named:
        return []
    with Session() as session:
        ready = set(
            session.scalars(
                select(Model.name).where(
                    Model.name.in_(named), Model.status == Status.ready
                )
            )
        )
    return sorted(set(named) - ready)


def validate_axes(axes: dict) -> None:
    unknown = sorted(set(axes) - set(AXES_ALLOWED))
    if unknown:
        raise ValueError(
            f"a rejudge cannot move {unknown}; its axes are {list(AXES_ALLOWED)}"
        )
    if not axes:
        raise ValueError("a rejudge with no axes compares an arm against itself")
    for name, values in axes.items():
        if not values:
            raise ValueError(f"axis {name} has no values")
        if name == "judge_model":
            # the same shape the generation door demands of a swept model: without it a
            # 300-character value with a newline in it reached psycopg as a 500
            bad = [
                v for v in values
                if not isinstance(v, str) or len(v) > 128 or not MODEL_NAME_RE.fullmatch(v)
            ]
            if bad:
                raise ValueError(f"judge_model takes model names, got {bad}")
            continue
        if name == REPEAT:
            labels = [v if isinstance(v, str | int) and not isinstance(v, bool) else None
                      for v in values]
            if None in labels:
                raise ValueError(f"repeat takes labels, got {values}")
            # the label rides into `run_name` on every copied row, so it is bounded the way
            # a model name is: unbounded, one request writes gigabytes of run names
            bad = [
                v for v in labels
                if not LABEL_RE.fullmatch(str(v))
            ]
            if bad:
                raise ValueError(
                    f"repeat labels are short names, got {bad}: {LABEL_RE.pattern}"
                )
            if len(set(labels)) != len(labels):
                raise ValueError(f"repeat labels must differ, got {values}")
            continue
        bad = [v for v in values if not isinstance(v, int) or isinstance(v, bool) or v < 1]
        if bad:
            raise ValueError(f"{name} takes prompt versions, got {bad}")


# from the record, not rebuilt from the axes: adding a value to one axis reshuffles the
# product, and pairing by position would relabel every arm with a neighbour's name while
# the lengths still agreed
def paired_arms(exp) -> list[tuple[dict, str]]:
    stored = (exp.procedure or {}).get("arms")
    if stored:
        return [(row["arm"], row["run"]) for row in stored]
    # experiments recorded before the mapping was stored: their grid was never extended,
    # so the product still reproduces the order their names were made in
    return list(zip(retrieval_compare.arms(exp.axes), exp.run_names, strict=True))


def stored_arms(pairs: list[tuple[dict, str]]) -> list[dict]:
    return [{"arm": arm, "run": name} for arm, name in pairs]


# arms named one by one rather than as a grid: the caller adding an arm to a finished
# experiment knows which arm it wants, and the product of the extended axes would also
# name the arms already run
def folded_axes(arms: list[dict]) -> dict:
    axes: dict[str, list] = {}
    for arm in arms:
        for name, value in arm.items():
            axes.setdefault(name, [])
            if value not in axes[name]:
                axes[name].append(value)
    return axes


def validate_arms(arms: list[dict]) -> None:
    if not arms:
        raise ValueError("no arms to add")
    if any(not isinstance(arm, dict) or not arm for arm in arms):
        raise ValueError("an arm is a mapping of axis to value, and it is not empty")
    validate_axes(folded_axes(arms))
    names = [retrieval_compare.arm_name(arm) for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"arms do not have distinct names: {sorted(names)}")


def _prompt_axes(arm: dict) -> dict:
    return {k: v for k, v in arm.items() if k not in ("judge_model", REPEAT)}


def arm_bench(arm: dict) -> judge.Bench:
    versions = {Purpose[name]: version for name, version in _prompt_axes(arm).items()}
    return judge.Bench(model=arm.get("judge_model"), versions=versions or None)


def arm_options(arm: dict, run_name: str) -> dict:
    return {
        "run_name": run_name,
        "judge_model": arm.get("judge_model"),
        "judge_prompts": _prompt_axes(arm),
    }


# what was judged, as a fact in the record rather than a promise in the description
def answers_digest(run_name: str) -> str:
    with Session() as session:
        rows = session.execute(
            select(QuestionLog.question_id, QuestionLog.answer)
            # the same rows `_scored` reads, ordered by something stable across copies:
            # `id` comes from an unordered INSERT ... SELECT, so it cannot break the tie
            .where(QuestionLog.run_name == run_name, QuestionLog.question_id.isnot(None))
            .order_by(QuestionLog.question_id, QuestionLog.answer)
        ).all()
    digest = hashlib.sha256(usedforsecurity=False)
    for question_id, answer in rows:
        digest.update(f"{question_id}\x00{answer or ''}\x00".encode())
    return f"sha256:{digest.hexdigest()[:16]}:{len(rows)}"


def _scored(run_name: str) -> dict[int, dict]:
    with Session() as session:
        rows = session.execute(
            select(
                QuestionLog.question_id,
                QuestionLog.faithfulness,
                QuestionLog.relevance,
                QuestionLog.completeness,
            ).where(QuestionLog.run_name == run_name, QuestionLog.question_id.isnot(None))
        ).all()
    return {
        r.question_id: {
            axis: int(getattr(r, axis))
            for axis in AXES
            if getattr(r, axis) and str(getattr(r, axis)).isdigit()
        }
        for r in rows
    }


def _paired(before: dict, after: dict, axis: str, which: str | None = None) -> dict | None:
    deltas = []
    for question_id in sorted(set(before) & set(after)):
        if which and half_of(question_id) != which:
            continue
        was, now = before[question_id].get(axis), after[question_id].get(axis)
        if was is not None and now is not None:
            deltas.append(now - was)
    if not deltas:
        return None
    # eight seeds, same rule as the model grid: a bound that changes sign when the
    # resampling changes is a parity, not a result
    bounds = [bootstrap_ci(deltas, seed=s) for s in range(8)]
    low, high = bounds[0]
    return {
        "n": len(deltas),
        "delta": round(sum(deltas) / len(deltas), 4),
        "ci95": [round(low, 4), round(high, 4)],
        "seed_shaky": (
            any(lo <= 0 for lo, _ in bounds) and any(lo > 0 for lo, _ in bounds)
            or any(hi >= 0 for _, hi in bounds) and any(hi < 0 for _, hi in bounds)
        ),
        "better": sum(1 for d in deltas if d > 0),
        "worse": sum(1 for d in deltas if d < 0),
    }


# every arm names the instrument that scored it: the arm dict says what was asked for, the
# rows say what actually ran, and a rejudge is the one kind where those can differ
def _judged_by(run_name: str) -> dict:
    with Session() as session:
        rows = session.execute(
            select(QuestionLog.models, QuestionLog.prompts).where(
                QuestionLog.run_name == run_name
            )
        ).all()
    models = {(m or {}).get(JUDGE_MODEL_KEY) for m, _ in rows}
    prompts = {
        axis: sorted({(p or {}).get(f"judge_{axis}") for _, p in rows} - {None})
        for axis in AXES
    }
    return {
        "model": sorted(models - {None}) or None,
        "prompts": {axis: seen for axis, seen in prompts.items() if seen} or None,
    }


def _mean(scored: dict, axis: str) -> float | None:
    values = [row[axis] for row in scored.values() if axis in row]
    return round(sum(values) / len(values), 4) if values else None


# every pair while the grid is small, first-against-the-rest once it is not: a pair costs
# nine bootstraps of eight seeds, about fifteen seconds over 823 rows, and the aggregation
# holds a transaction while it runs. Which rule was applied goes into the record
PAIR_EVERY_UP_TO = 6


def _couples(order: list[str]) -> tuple[list[tuple[str, str]], str]:
    if len(order) <= PAIR_EVERY_UP_TO:
        return list(itertools.combinations(order, 2)), "every pair"
    return [(order[0], name) for name in order[1:]], f"against {order[0]}"


def compute_results(source_run: str, param: str, pairs: list[tuple[dict, str]]) -> dict:
    run_names = [name for _, name in pairs]
    loaded = {name: _scored(name) for name in run_names}
    digests = {name: answers_digest(name) for name in run_names}
    source = answers_digest(source_run)

    per_arm = {
        name: {
            "arm": arm,
            "n": len(loaded[name]),
            "answers_digest": digests[name],
            "judge": _judged_by(name),
            **{axis: _mean(loaded[name], axis) for axis in AXES},
        }
        for arm, name in pairs
    }

    # the source carries verdicts already, so it is the cheapest arm any rejudge has: one
    # new arm against the record needs no paid twin to be compared against
    source_scored = _scored(source_run)
    order = list(run_names)
    if source_scored:
        loaded[source_run] = source_scored
        digests[source_run] = source
        order.insert(0, source_run)

    couples, pairing = _couples(order)
    deltas = {}
    for before, after in couples:
        deltas[f"{before}_vs_{after}"] = {
            axis: _paired(loaded[before], loaded[after], axis)
            for axis in AXES
        } | {
            "halves": {
                axis: {w: _paired(loaded[before], loaded[after], axis, w) for w in ("A", "B")}
                for axis in AXES
            },
            # the answers are the same by construction; the digest is what says so
            "same_answers": digests[before] == digests[after] == source,
        }

    return {
        # the shape changed three times in one week (`pairing`, `source_scored`, the
        # source's judge), and a record written before a field existed is indistinguishable
        # from one where the field is absent for a reason. A reader compares this, not dates
        "schema": SCHEMA,
        "param": param,
        "source_run": source_run,
        "source_digest": source,
        "source_scored": {
            "n": len(source_scored),
            "judge": _judged_by(source_run),
            **{axis: _mean(source_scored, axis) for axis in AXES},
        } if source_scored else None,
        "pairing": pairing,
        "per_arm": per_arm,
        "deltas": deltas,
    }
