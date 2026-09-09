"""A blind worksheet of answer pairs, and the agreement of each judge with the human who filled it.

Both judges on this stand are language models, and so is the one writing this. The owner is the
only reader of a different nature, so his verdict is a third instrument rather than a truth: what
comes out is agreement, not correctness. The order of the list is fixed before he starts, so
stopping early is not a choice made after seeing the answers.
"""

import hashlib
import random
import re
from collections import defaultdict, namedtuple
from datetime import date
from itertools import combinations
from pathlib import Path

import config
from evals.loaders import load_logs
from evals.measurements import hand_back, store_json
from evals.pools import JOINS_BOTH_JUDGES, joins_both_judges
from evals.stats import to_unit

import db

# 1 the sheet, its key and the two readings, moved here from a script
SCHEMA = 1

GUEST = "ragas_faithfulness"
SEED = 20260908
BOTH_SPOKE = 15
REPEATS = 3
SHEETS = Path(__file__).resolve().parents[2] / "temp_files"


# a sheet the owner is filling is worth a refusal, and a library says so without exiting his shell
class Refused(ValueError):
    pass


def _guest(ql):
    return ((ql.metrics or {}).get(GUEST) or {}).get("score")


# the correlation's predicate, called rather than restated: the copy here did not need a context
def _rows():
    return sorted((ql for ql in load_logs(None) if joins_both_judges(ql)), key=lambda ql: ql.id)


# every covariate of a pair in one place: `build` wrote them and `mark` wrote them again
def _covariates(left, right) -> dict:
    langs = {name: _language(ql.answer) for name, ql in (("A", left), ("B", right))}
    leaks = {name: _leak(ql) for name, ql in (("A", left), ("B", right))}
    return {
        "A": {"log_id": left.id, "run": left.run_name, "lang": langs["A"], "leak": leaks["A"]},
        "B": {"log_id": right.id, "run": right.run_name, "lang": langs["B"], "leak": leaks["B"]},
        "cross_language": langs["A"] != langs["B"],
        # the detector reads a config mode, so the record names the one that produced these
        "detector": config.settings.retrieval.query_lang,
        # neither side found the gold file: both are ungrounded by construction
        "neither_hit_gold": not _hit_gold(left) and not _hit_gold(right),
        "template_leak": bool(leaks["A"]) or bool(leaks["B"]),
    }


# the project's own rule, not a second one: containment over what retrieval actually returned
def _hit_gold(ql) -> bool:
    from evals import retrieval_metrics

    gold = (ql.question.marked_sources if ql.question else None) or []
    _, _, names = retrieval_metrics.retrieved_sources(ql)
    return any(retrieval_metrics.is_gold(one, gold) for one in names)


LEAKS = (
    "final answer to the user", "the final answer is", "here is the answer",
    "the answer to the question is",
)


# scaffolding in the answer body is a formatting failure, not an ungrounded claim: recorded, not judged
def _leak(ql) -> str:
    head = (ql.answer or "")[:400].lower()
    if any(mark in head for mark in LEAKS):
        return "wrapper"
    asked = (ql.question_text or "")[:40].lower()
    return "echoed_the_question" if asked and asked in head else ""


# the judge charges for Russian at the same meaning, so language is a covariate of the pair
def _language(text: str) -> str:
    return db.detect_language(text or "")


# the letter on the sheet against the row it points at: four copies of this, one inverted
def _sides_of(left_id, right_id) -> dict:
    return {"A": left_id, "B": right_id}


def _sides(pair: dict) -> dict:
    return _sides_of(pair["A"]["log_id"], pair["B"]["log_id"])


def _row_picked(sides: dict, letter: str):
    return "=" if letter == "=" else sides[letter]


def _letter_picked(sides: dict, row) -> str:
    return "=" if row == "=" else next(name for name, one in sides.items() if one == row)


# read by index in five expressions before it had names
Scored = namedtuple("Scored", "left right ours guest")


def _deltas(a, b) -> tuple[float, float]:
    return to_unit(a.faithfulness) - to_unit(b.faithfulness), _guest(a) - _guest(b)


# declared before the owner sees a single answer: contested first, then whatever separates most
def _ordered(rows) -> list:
    by_question = defaultdict(list)
    for ql in rows:
        by_question[ql.question_id].append(ql)
    # the deltas are read four times per pair, and reading them once is what keeps this linear
    scored = [
        Scored(a, b, *_deltas(a, b))
        for kept in by_question.values()
        for a, b in combinations(sorted(kept, key=lambda x: x.id), 2)
    ]

    def spoke(one) -> bool:
        return abs(one.ours) > 1e-9 and abs(one.guest) > 1e-9

    def gap(one) -> float:
        return -abs(one.ours - one.guest)

    opposed = [one for one in scored if spoke(one) and one.ours * one.guest < 0]
    apart = sorted((one for one in scored if spoke(one) and one.ours * one.guest > 0), key=gap)
    silent = sorted((one for one in scored if not spoke(one)), key=gap)
    return [(one.left, one.right) for one in (*opposed, *apart, *silent)]


# flipped: on the main sheet every english answer landed on A and he chose B five times out of five
def _repeats(sitting: list) -> list:
    crossed = [
        (left, right, n) for left, right, n in sitting
        if _language(left.answer) != _language(right.answer)
    ]
    rest = [t for t in sitting[:REPEATS] if t not in crossed]
    flipped = [(right, left) for left, right, _ in (*crossed, *rest)]
    random.Random(SEED + 1).shuffle(flipped)
    return [(left, right, n) for n, (left, right) in enumerate(flipped, 1)]


def _side(pair, rng) -> tuple:
    left, right = pair
    return (right, left) if rng.random() < 0.5 else (left, right)


def _context_block(left, right) -> str:
    if list(left.contexts or []) == list(right.contexts or []):
        joined = "\n\n".join(left.contexts or [])
        return f"**Контекст, один на обе стороны**\n\n```\n{joined}\n```\n"
    return "".join(
        f"**Контекст стороны {name}**\n\n```\n" + "\n\n".join(ql.contexts or []) + "\n```\n\n"
        for name, ql in (("A", left), ("B", right))
    )


# answers travel by the pair of rows, never by the pair number: a rebuild renumbers and may flip
def _already(stamp: str) -> dict:
    # a pruned answer carries the sides it was given on, so a later rebuild cannot reinterpret it
    out = {
        frozenset((got["A"], got["B"])): _row_picked(
            _sides_of(got["A"], got["B"]), got["answer"]
        )
        for got in _load(_done_path(stamp), {}).values()
    }
    key = _load(_key_path(stamp))
    sheet = _sheet_path(stamp, "sitting")
    said = _picked(sheet) if (key and sheet.exists()) else {}
    for pair in key["pairs"]["sitting"] if key else ():
        answer = said.get(pair["n"])
        if answer:
            sides = _sides(pair)
            out[frozenset(sides.values())] = _row_picked(sides, answer)
    return out


def _sheet(entries: list, title: str, note: str, kept: dict | None = None) -> str:
    out = [f"# {title}\n", note, "\n"]
    kept = kept or {}
    for n, (left, right, _) in enumerate(entries, 1):
        out.append(f"\n## Пара {n}\n")
        out.append(f"**Вопрос**: {left.question_text}\n")
        out.append("\n" + _context_block(left, right) + "\n")
        out.append(f"**Ответ A**\n\n{left.answer}\n")
        out.append(f"\n**Ответ B**\n\n{right.answer}\n")
        was = kept.get(frozenset((left.id, right.id)))
        mark = "____" if was is None else _letter_picked(_sides_of(left.id, right.id), was)
        out.append(f"\nЛучше подкреплён контекстом (впиши A, B или `=`): **{mark}**\n")
    return "\n".join(out)


def build(stamp: str | None = None) -> dict:
    rows = _rows()
    order = _ordered(rows)
    rng = random.Random(SEED)
    sitting = [(*_side(p, rng), n) for n, p in enumerate(order[:BOTH_SPOKE], 1)]
    later = _repeats(sitting)

    stamp = _dated(stamp) if stamp else date.today().strftime("%Y%m%d")
    SHEETS.mkdir(parents=True, exist_ok=True)
    kept = _already(stamp)
    # through the one function that checks the stamp: two more places built these names by hand
    first, second = _sheet_path(stamp, "sitting"), _sheet_path(stamp, "repeats")
    # the repeats sheet never carries an answer forward, so a rebuild over a filled one loses it
    if second.exists() and any(_picked(second).values()):
        raise Refused(
            f"{second.name} already carries answers and this sheet cannot carry them forward:"
            " move it aside first, or the owner's afternoon is gone"
        )
    # an answer only survives a rebuild if its pair is still on the list or already pruned away
    safe = {frozenset((got["A"], got["B"])) for got in _load(_done_path(stamp), {}).values()}
    staying = {frozenset((left.id, right.id)) for left, right, _ in sitting}
    at_risk = set(kept) - safe - staying
    if at_risk:
        raise Refused(
            f"{len(at_risk)} answered pairs fall off the new list and would take their answers"
            " with them: run prune first, which moves them into the done file"
        )

    first.write_text(
        _sheet(
            sitting,
            f"Человеческий якорь: {len(sitting)} пар",
            "Слепой лист. Имена рук скрыты, порядок внутри пары брошен монеткой, порядок пар"
            " объявлен до того, как ты это открыл. Останавливайся где хочешь: сколько заполнено,"
            " столько и прочитаем. Пересборка листа ответы не теряет: они переносятся по паре"
            " строк, а не по номеру пары.\n\n"
            "**Что именно судишь.** Чьи утверждения лучше подкреплены **показанным контекстом**."
            " Не чей ответ лучше и не кто прав по сути. Ответ может быть верным и при этом не"
            " заземлённым: если в контексте этого нет, он взят из головы модели. Если оба стоят"
            " на воздухе, ставь `=`, это законный исход, а не отказ отвечать.",
            kept,
        ),
        encoding="utf-8",
    )
    second.write_text(
        _sheet(
            later,
            f"Человеческий якорь: {len(later)} повторов, брать в другой день",
            "Это пары из основного листа, **перевёрнутые**: то, что стояло справа, теперь слева."
            " Читается ровно одно: идёшь ты за стороной или за содержанием. На основном листе"
            " английский ответ во всех разноязычных парах лёг на A, и ты все пять раз выбрал B,"
            " поэтому позицию и язык там не разделить. Узнавание пары тут не мешает: оно работает"
            " против позиционного чтения, а не за него.",
            # never `kept`: a carried answer would be the very verdict this sheet checks
            None,
        ),
        encoding="utf-8",
    )
    for path in (first, second):
        hand_back(path)

    key = {
        "schema": SCHEMA,
        "population": JOINS_BOTH_JUDGES,
        "seed": SEED,
        "ordering": "opposed first, then both judges spoke, then by |ours - guest| descending",
        "sheets": {"sitting": str(first), "repeats": str(second)},
        "pairs": {
            sheet: [
                {
                    "n": n,
                    **_covariates(left, right),
                    "ours": round(_deltas(left, right)[0], 4),
                    "guest": round(_deltas(left, right)[1], 4),
                }
                for left, right, n in entries
            ]
            for sheet, entries in (("sitting", sitting), ("repeats", later))
        },
    }
    # the sheet he filled must be the sheet this key describes, or the pairs join to the wrong rows
    key["sheet_fingerprint"] = _fingerprint(first.read_text(encoding="utf-8"))
    where = _key_path(stamp)
    store_json(where, key)
    return {"schema": SCHEMA, "sheet": str(first), "repeats": str(second),
            "key": str(where), "pairs": len(sitting)}


# the answer lines are what he changes, so they cannot be part of what identifies the sheet
def _fingerprint(text: str) -> str:
    body = "\n".join(
        "ANSWER LINE" if "подкреплён контекстом" in line else line
        for line in text.splitlines()
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# covariates onto an existing key, without regenerating a sheet somebody is in the middle of
def mark(stamp: str) -> dict:
    from models.eval import QuestionLog
    from orm.sync_db import Session
    from sqlalchemy import select

    key = _key(stamp)
    wanted = {side["log_id"] for group in key["pairs"].values() for p in group
              for side in (p["A"], p["B"])}
    with Session() as session:
        rows = {ql.id: ql for ql in
                session.scalars(select(QuestionLog).where(QuestionLog.id.in_(wanted)))}
        touched = 0
        for group in key["pairs"].values():
            for pair in group:
                gone = [side["log_id"] for side in (pair["A"], pair["B"])
                        if side["log_id"] not in rows]
                if gone:
                    raise Refused(f"rows {gone} are gone from the base: the run was deleted")
                pair.update(_covariates(rows[pair["A"]["log_id"]], rows[pair["B"]["log_id"]]))
                touched += 1
    store_json(_key_path(stamp), key)
    return {"pairs_marked": touched}


def _dated(stamp: str) -> str:
    if not re.fullmatch(r"\d{8}", stamp):
        raise Refused(f"a stamp is a date like 20260908, not {stamp!r}")
    return stamp


def _load(path: Path, default=None):
    import json

    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _key(stamp: str) -> dict:
    got = _load(_key_path(stamp))
    if got is None:
        raise Refused(f"no key for {stamp}: build the sheet before reading it")
    return got


def _key_path(stamp: str) -> Path:
    return SHEETS / f"human_anchor_key_{_dated(stamp)}.json"


def _sheet_path(stamp: str, which: str) -> Path:
    tail = "" if which == "sitting" else f"_{which}"
    return SHEETS / f"human_anchor{tail}_{_dated(stamp)}.md"


def _done_path(stamp: str) -> Path:
    return SHEETS / f"human_anchor_done_{_dated(stamp)}.json"


# the lock guarded `read` alone, while `prune` rewrote both the sheet and the fingerprint
def _its_own(stamp: str, key: dict) -> Path:
    sheet = _sheet_path(stamp, "sitting")
    seen = _fingerprint(sheet.read_text(encoding="utf-8"))
    if key.get("sheet_fingerprint") not in (None, seen):
        raise Refused(
            "this sheet is not the one the key describes: rebuilt after it was filled, or edited"
            " beyond the answer lines. The pairs would join to the wrong rows"
        )
    return sheet


# answered pairs leave the sheet, so the next sitting is only what is left
def prune(stamp: str) -> dict:
    key = _key(stamp)
    sheet = _its_own(stamp, key)
    said = {n: v for n, v in _picked(sheet).items() if v}
    if not said:
        return {"moved": 0, "left": len(key["pairs"]["sitting"])}

    done_path = _done_path(stamp)
    done = _load(done_path, {})
    for pair in key["pairs"]["sitting"]:
        if pair["n"] in said:
            done[str(pair["n"])] = {
                "answer": said[pair["n"]],
                "A": pair["A"]["log_id"], "B": pair["B"]["log_id"],
            }
    store_json(done_path, done)

    text = sheet.read_text(encoding="utf-8")
    head, *blocks = re.split(r"(?=\n## Пара )", text)
    # numbers are never reused: the key joins by them, and a renumbered sheet would join wrong
    kept = [b for b in blocks if int(re.search(r"## Пара (\d+)", b).group(1)) not in said]
    sheet.write_text(head + "".join(kept), encoding="utf-8")
    hand_back(sheet)

    key["sheet_fingerprint"] = _fingerprint(sheet.read_text(encoding="utf-8"))
    store_json(_key_path(stamp), key)
    return {"moved": len(said), "left": len(kept), "done_file": str(done_path)}


# he fills a Russian sheet on a Russian layout, and А is not A: the answer was dropped in silence
_LOOKS_LIKE = {"А": "A", "В": "B", "С": "C"}


def _picked(sheet: Path) -> dict:
    out = {}
    n = None
    for line in sheet.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Пара "):
            n = int(line.split()[-1])
        if "подкреплён контекстом" in line and n is not None:
            said = line.rsplit("**", 2)[-2].strip().upper().strip("_")
            said = "".join(_LOOKS_LIKE.get(one, one) for one in said)
            out[n] = said if said in ("A", "B", "=") else None
    return out


# a judge that called the pair equal where the human did not is a miss, not an abstention
def _verdict(delta: float, said: str) -> str:
    # the same epsilon `_ordered` used: the deltas here were rounded to four places by `build`
    apart = abs(delta) > 1e-9
    if said == "=":
        return "missed" if apart else "agreed"
    if not apart:
        return "did_not_separate"
    return "agreed" if (delta > 0) == (said == "A") else "disagreed"


def read(stamp: str) -> dict:
    key = _key(stamp)
    _its_own(stamp, key)
    # by the pair of rows, like every other reader: joining by number scores a pair he never saw
    answered = _already(stamp)
    counted = {j: defaultdict(int) for j in ("ours", "guest")}
    buckets = ("same_language", "cross_language", "clean", "template_leak",
               "hit_gold", "neither_hit_gold")
    by_covariate = {j: {b: defaultdict(int) for b in buckets} for j in ("ours", "guest")}
    answered_here = 0
    for pair in key["pairs"]["sitting"]:
        sides = _sides(pair)
        picked = answered.get(frozenset(sides.values()))
        if picked is None:
            continue
        said = _letter_picked(sides, picked)
        answered_here += 1
        where = [
            "cross_language" if pair.get("cross_language") else "same_language",
            "template_leak" if pair.get("template_leak") else "clean",
            "neither_hit_gold" if pair.get("neither_hit_gold") else "hit_gold",
        ]
        for instrument in ("ours", "guest"):
            got = _verdict(pair[instrument], said)
            counted[instrument][got] += 1
            for bucket in where:
                by_covariate[instrument][bucket][got] += 1

    def share(counts) -> dict:
        n = sum(counts.values())
        return {
            "n": n,
            "agreed": counts["agreed"],
            "share": round(counts["agreed"] / n, 4) if n else None,
            "by_outcome": dict(counts),
        }

    def judged(instrument) -> dict:
        return {**share(counted[instrument]),
                "by_covariate": {k: share(v) for k, v in by_covariate[instrument].items()}}

    return {
        "schema": SCHEMA,
        "population": key["population"],
        # declared before the sheet was filled, not carved out of the result
        "subgroups": "declared before the sheet was filled: same against cross language (the judge"
                     " was measured charging for Russian at the same meaning, though how much"
                     " depends on the population), and clean against a scaffolding leak in the body",
        "pairs_offered": len(key["pairs"]["sitting"]),
        "pairs_answered": answered_here,
        "ours": judged("ours"),
        "guest": judged("guest"),
    }


# the same row picked twice is content, the same letter picked twice is position
def repeats(stamp: str) -> dict:
    key = _key(stamp)
    # the same source every reader uses: an answer still sitting in the sheet counts too
    before = _already(stamp)
    said = _picked(_sheet_path(stamp, "repeats"))

    tally = defaultdict(int)
    for pair in key["pairs"]["repeats"]:
        now = said.get(pair["n"])
        sides = _sides(pair)
        was = before.get(frozenset(sides.values()))
        if not now or was is None:
            continue
        picked = _row_picked(sides, now)
        if picked == was:
            tally["same_row"] += 1
        elif was == "=" or picked == "=":
            tally["one_side_called_it_equal"] += 1
        else:
            tally["same_position"] += 1
    counted = sum(tally.values())
    return {
        "schema": SCHEMA,
        "population": "the pairs of the main sheet, flipped: cross language first, then the rest",
        "n": counted,
        "share_same_row": round(tally["same_row"] / counted, 3) if counted else None,
        "by_outcome": dict(tally),
        "reads": "same_row means he follows what is written, same_position means he follows the side",
    }
