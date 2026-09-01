import hashlib
import itertools
import re

import prompt_repo
from evals import sampling
from evals.stats import annotate_holm, deltas_over, mean_of, tally
from models.eval import QuestionLog
from models.registry import (
    MAX_MODEL_NAME,
    MODEL_NAME_RE,
    Model,
    ModelRole,
    Prompt,
    Purpose,
    Role,
    Status,
)
from orm.sync_db import Session
from scipy.stats import wilcoxon
from sqlalchemy import delete, func, insert, literal, select, text
from use_cases import judge, retrieval_compare
from use_cases.retrieval_compare import bootstrap_ci, half_of

# 1 means and deltas; 2 pairing and `source_scored`; 3 the source's judge; 4 p and Holm
SCHEMA = 4

AXES = ("faithfulness", "relevance", "completeness")
# a copy is unjudged, so it must not carry the judge the original named
JUDGE_MODEL_KEY = "judging"


def _stripped(column, keys):
    for key in keys:
        column = column.op("-")(literal(key))
    return column


# one run's answers under a new name, verdicts cleared, ready to be judged again
def copy_run(source: str, target: str, question_ids=None) -> int:
    if not source or not target:
        raise ValueError("both the source run and the name of the copy are required")
    if source == target:
        raise ValueError("a copy under the same name would be judged as the original")

    refuse_oversized_fanout(source, 1, question_ids=question_ids)
    with Session() as session:
        _refuse_bad_pair(session, source, target)

        # RETURNING, not rowcount: an INSERT ... SELECT reports -1 through this driver
        copied = session.execute(copy_statement(source, target, question_ids)).scalars().all()
        session.commit()
        return len(copied)


# all of them or none: half a fan-out burns the names for the identical retry
def copy_runs(source: str, targets: list[str], question_ids=None) -> dict[str, int]:
    # a null source compiles to `run_name IS NULL` and fans out over every orphan row
    if not source or not targets or not all(targets):
        raise ValueError(f"a copy needs a source run and names, got {source!r} {targets}")
    if len(set(targets)) != len(targets):
        raise ValueError(f"the copies must have distinct names, got {targets}")
    refuse_oversized_fanout(source, len(targets), question_ids=question_ids)
    # a fixed order: two fan-outs sharing a name would take their locks in opposite orders
    targets = sorted(targets)
    with Session() as session:
        for target in targets:
            _refuse_bad_pair(session, source, target)
        made = {
            target: len(
                session.execute(copy_statement(source, target, question_ids)).scalars().all()
            )
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


# a count then an insert from a threadpool: two requests both find the name free
def _claim(session, target: str) -> None:
    session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:name))"), {"name": target}
    )


def _refuse_bad_pair(session, source: str, target: str) -> None:
    _claim(session, target)
    # answered, not merely present: a run of nothing copies cleanly and reads as complete
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
    # taken from the model: a migration adding a column would leave the copy missing it
    return [c.name for c in QuestionLog.__table__.columns if c.name != "id"]


def copy_statement(source: str, target: str, question_ids=None):
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
    if question_ids is not None:
        picked = picked.where(QuestionLog.question_id.in_(list(question_ids)))
    return insert(QuestionLog).from_select(carried, picked).returning(QuestionLog.id)


# what an arm may move. `repeat` moves nothing: its delta is the judge's own noise
REPEAT = "repeat"
AXES_ALLOWED = (REPEAT, "judge_model", *[f"judge_{axis}" for axis in AXES])
# one request over the 823-question sets can copy 26k rows and queue 79k judge calls
MAX_ARM_ROWS = 4000
# what a `repeat` label may look like: it becomes part of a run name on every copied row
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


# the cap is about rows written, and `copy_statement` copies every row of the source
def _source_rows(source: str, question_ids=None) -> int:
    stmt = select(func.count()).select_from(QuestionLog).where(QuestionLog.run_name == source)
    if question_ids is not None:
        stmt = stmt.where(QuestionLog.question_id.in_(list(question_ids)))
    with Session() as session:
        return session.scalar(stmt) or 0


# `existing` is what the experiment holds: arms posted one at a time reached 26k rows
def refuse_oversized_fanout(
    source: str, arm_count: int, existing: int = 0, question_ids=None
) -> None:
    rows = _source_rows(source, question_ids)
    total = rows * (arm_count + existing)
    if total > MAX_ARM_ROWS:
        raise ValueError(
            f"{arm_count} new arms beside {existing} over {rows} rows is {total} rows to"
            f" judge, over the cap of {MAX_ARM_ROWS}: run fewer arms or a smaller source"
        )


# shape was checked and existence was not: the arm dies per log while the job says done
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


# an omitted axis is not "nothing", it is whichever version is active right now
def _effective_judge(arm: dict) -> dict:
    with Session() as session:
        served = session.scalar(
            select(Model.name)
            .join(ModelRole, ModelRole.model_id == Model.id)
            .where(ModelRole.role == Role.judging)
        )
    unnamed = [Purpose[f"judge_{axis}"] for axis in AXES if not arm.get(f"judge_{axis}")]
    active = prompt_repo.active_versions(unnamed) if unnamed else {}
    versions = {
        axis: arm.get(f"judge_{axis}") or active[f"judge_{axis}"] for axis in AXES
    }
    return {"model": arm.get("judge_model") or served, "prompts": versions}


# with no prompt versions on the rows, `all()` over nothing was True and any model matched
def _reproduces(arm: dict, judged: dict) -> bool:
    named_model = (judged.get("model") or [None])[0]
    named = {axis: seen for axis, seen in (judged.get("prompts") or {}).items() if len(seen) == 1}
    if not named_model or not named:
        return False
    effective = _effective_judge(arm)
    if effective["model"] != named_model:
        return False
    return all(effective["prompts"][axis] == seen[0] for axis, seen in named.items())


# drift between passes has run from -0.0972 to +0.2442, the size of the effects we look for
def refuse_unpaired_rejudge(source_run: str, arms: list[dict], unpaired: bool = False) -> None:
    if unpaired:
        return
    judged = _judged_by(source_run)
    named = {axis: seen for axis, seen in (judged.get("prompts") or {}).items()}
    mixed = sorted(axis for axis, seen in named.items() if len(seen) > 1)
    if mixed:
        raise ValueError(
            f"the rows of {source_run} carry more than one judge prompt version on {mixed},"
            " so no arm can reproduce their judge: judge a clean source, or pass unpaired=true"
        )
    if not named and not judged.get("model"):
        # rows written before the judge was stamped: the door can only insist a human declared one
        if not any(REPEAT in arm for arm in arms):
            raise ValueError(
                f"the rows of {source_run} carry no judge at all, so the record cannot say"
                " what a control would have to reproduce: add a `repeat` arm naming the judge"
                " you believe produced them (about 1.5 h of card per 823 rows), or pass"
                " unpaired=true and read no deltas against the source"
            )
        return
    if not any(_reproduces(arm, judged) for arm in arms):
        wanted = {axis: seen[0] for axis, seen in named.items()}
        raise ValueError(
            f"no arm reproduces the judge of {source_run} ({judged.get('model')}, {wanted}),"
            " so every delta against it would carry the drift of another pass: add that arm"
            " (about 1.5 h of card per 823 rows), or pass unpaired=true"
        )


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
            # without it a 300-character value with a newline reached psycopg as a 500
            bad = [
                v for v in values
                if not isinstance(v, str) or len(v) > MAX_MODEL_NAME
                or not MODEL_NAME_RE.fullmatch(v)
            ]
            if bad:
                raise ValueError(f"judge_model takes model names, got {bad}")
            continue
        if name == REPEAT:
            labels = [v if isinstance(v, str | int) and not isinstance(v, bool) else None
                      for v in values]
            if None in labels:
                raise ValueError(f"repeat takes labels, got {values}")
            # the label rides into `run_name` on every copied row, so it is bounded
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


# from the record: pairing by position would relabel every arm with a neighbour's name
def paired_arms(exp) -> list[tuple[dict, str]]:
    stored = (exp.procedure or {}).get("arms")
    if stored:
        return [(row["arm"], row["run"]) for row in stored]
    # experiments recorded before the mapping was stored: their grid was never extended
    return list(zip(retrieval_compare.arms(exp.axes), exp.run_names, strict=True))


def stored_arms(pairs: list[tuple[dict, str]]) -> list[dict]:
    return [{"arm": arm, "run": name} for arm, name in pairs]


# arms named one by one: the product would also name the arms already run
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
    refuse_repeated_names(arms)


# two arms under one name are one arm in every reading downstream
def refuse_repeated_names(arms: list[dict]) -> None:
    names = [retrieval_compare.arm_name(arm) for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"arms do not have distinct names: {sorted(names)}")


def _prompt_axes(arm: dict) -> dict:
    return {k: v for k, v in arm.items() if k not in ("judge_model", REPEAT)}


def arm_bench(arm: dict) -> judge.Bench:
    versions = {Purpose[name]: version for name, version in _prompt_axes(arm).items()}
    return judge.Bench(model=arm.get("judge_model"), versions=versions or None)


# the axes this arm does not move are its control, and may be judged on a sample
def control_axes(arm: dict) -> list[str]:
    named = {name.removeprefix("judge_") for name in _prompt_axes(arm)}
    return [axis for axis in AXES if axis not in named]


def arm_options(
    arm: dict, run_name: str, control_sample: int | None = None, control_seed: int = 0
) -> dict:
    out = {
        "run_name": run_name,
        "judge_model": arm.get("judge_model"),
        "judge_prompts": _prompt_axes(arm),
    }
    if control_sample:
        out["control_axes"] = control_axes(arm)
        out["control_sample"] = control_sample
        # without the seed no reader can say which questions the control was judged on
        out["control_seed"] = control_seed
    return out


# `question_ids` narrows it to the shared rows: whole-run digests of two sizes differ
def answers_digest(run_name: str, question_ids=None) -> str:
    stmt = (
        select(QuestionLog.question_id, QuestionLog.answer)
        # ordered by something stable across copies: `id` comes from an unordered INSERT SELECT
        .where(QuestionLog.run_name == run_name, QuestionLog.question_id.isnot(None))
        .order_by(QuestionLog.question_id, QuestionLog.answer)
    )
    if question_ids is not None:
        stmt = stmt.where(QuestionLog.question_id.in_(list(question_ids)))
    with Session() as session:
        rows = session.execute(stmt).all()
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
    ids = [
        qid
        for qid in sorted(set(before) & set(after))
        if not which or half_of(qid) == which
    ]
    deltas = deltas_over(
        {qid: before[qid].get(axis) for qid in ids},
        {qid: after[qid].get(axis) for qid in ids},
        ids,
    )
    if not deltas:
        return None
    # eight seeds: a bound that changes sign with the resampling is a parity, not a result
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
        "better": tally(deltas)["better"],
        "worse": tally(deltas)["worse"],
        # the interval says how big, this says whether a family of them survives together
        "p": 1.0 if all(d == 0 for d in deltas) else round(float(wilcoxon(deltas).pvalue), 6),
    }


# named rather than assumed: a round that declared a narrower family corrects over that
def _annotate_family(deltas: dict, alpha: float = 0.05) -> dict:
    tests = [
        stats
        for axes in deltas.values()
        for axis, stats in axes.items()
        if axis in AXES and stats is not None
    ]
    return annotate_holm(tests, "every pair of this report on every axis", alpha)


# the arm says what was asked for, the rows say what ran, and here those can differ
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


def _same_answers(before: str, after: str, source_run: str, loaded: dict) -> bool:
    shared = sorted(set(loaded[before]) & set(loaded[after]))
    if not shared:
        return False
    digests = {name: answers_digest(name, shared) for name in {before, after, source_run}}
    return len(set(digests.values())) == 1


# the same stable rule the grid samples with, so one size over one source reads one set
def sample_of(source: str, size: int, seed: int = 0) -> list[int]:
    with Session() as session:
        return list(
            session.scalars(
                select(QuestionLog.question_id)
                .where(QuestionLog.run_name == source, QuestionLog.question_id.isnot(None))
                .order_by(sampling.by_id_and_seed(QuestionLog.question_id, seed))
                .limit(size)
            )
        )


def _mean(scored: dict, axis: str) -> float | None:
    return mean_of((row[axis] for row in scored.values() if axis in row), 4)


# a pair is fifteen seconds over 823 rows, and the aggregation holds a transaction meanwhile
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
            # a mean over two hundred rows reads exactly like a mean over eight hundred without this
            "n_by_axis": {
                axis: sum(1 for row in loaded[name].values() if axis in row) for axis in AXES
            },
        }
        for arm, name in pairs
    }

    # the source carries verdicts already: the cheapest arm any rejudge has
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
            # over the rows the pair shares: an arm judged on fewer rows differs by size alone
            "same_answers": _same_answers(before, after, source_run, loaded),
        }

    family = _annotate_family(deltas)

    return {
        # the shape changed three times in one week; a reader compares this, not dates
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
        "multiplicity": family,
    }


def for_reading(results: dict) -> dict:
    return {
        "source_run": results.get("source_run"),
        "pairing": results.get("pairing"),
        "multiplicity": results.get("multiplicity"),
        "ranking": None,
        "arms": {
            name: {k: v for k, v in arm.items() if k in ("arm", "n", "judge", "answers_digest")}
            for name, arm in (results.get("per_arm") or {}).items()
        },
        "deltas": results.get("deltas") or {},
    }
