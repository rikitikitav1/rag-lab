"""A blind worksheet of answer pairs, and the agreement of each judge with the human who filled it.

Both judges on this stand are language models, and so is the one writing this. The owner is the
only reader of a different nature, so his verdict is a third instrument rather than a truth: what
comes out is agreement, not correctness. The order of the list is fixed before he starts, so
stopping early is not a choice made after seeing the answers.
"""

import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

from evals.loaders import load_logs
from evals.measurements import hand_back
from evals.pools import in_corpus_and_answered
from evals.stats import to_unit

# 1 the sheet, its key and the two readings, moved here from a script
SCHEMA = 1

GUEST = "ragas_faithfulness"
SEED = 20260908
BOTH_SPOKE = 15
REPEATS = 3
HERE = Path(__file__).resolve().parents[2] / "temp_files"


# a sheet the owner is filling is worth a refusal, and a library says so without exiting his shell
class Refused(ValueError):
    pass


def _guest(ql):
    return ((ql.metrics or {}).get(GUEST) or {}).get("score")


# one predicate, applied here and named in every file this writes; sorted, because the loader is not
def _rows():
    kept = [
        ql
        for ql in load_logs(None)
        if ql.faithfulness is not None and _guest(ql) is not None and in_corpus_and_answered(ql)
    ]
    return sorted(kept, key=lambda ql: ql.id)


# every covariate of a pair in one place: `build` wrote them and `mark` wrote them again
def _covariates(left, right) -> dict:
    return {
        "A": {"log_id": left.id, "run": left.run_name,
              "lang": _language(left.answer), "leak": _leak(left)},
        "B": {"log_id": right.id, "run": right.run_name,
              "lang": _language(right.answer), "leak": _leak(right)},
        "cross_language": _language(left.answer) != _language(right.answer),
        # neither side found the gold file: both are ungrounded by construction
        "neither_hit_gold": not _hit_gold(left) and not _hit_gold(right),
        "template_leak": bool(_leak(left)) or bool(_leak(right)),
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


# the judge loses 0.964 of a point on Russian at the same meaning, so language is a covariate
def _language(text: str) -> str:
    import db

    return db.detect_language(text or "")


# the letter on the sheet against the row it points at: three copies of this, one inverted
def _sides(pair: dict) -> dict:
    return {"A": pair["A"]["log_id"], "B": pair["B"]["log_id"]}


def _row_picked(sides: dict, letter: str):
    return "=" if letter == "=" else sides[letter]


def _letter_picked(sides: dict, row) -> str:
    return "=" if row == "=" else next(name for name, one in sides.items() if one == row)


def _deltas(a, b) -> tuple[float, float]:
    return to_unit(a.faithfulness) - to_unit(b.faithfulness), _guest(a) - _guest(b)


# declared before the owner sees a single answer: contested first, then whatever separates most
def _ordered(rows) -> list:
    by_question = defaultdict(list)
    for ql in rows:
        by_question[ql.question_id].append(ql)
    # the deltas are read four times per pair, and reading them once is what keeps this linear
    scored = [
        (a, b, *_deltas(a, b))
        for kept in by_question.values()
        for a, b in combinations(sorted(kept, key=lambda x: x.id), 2)
    ]
    spoke = [one for one in scored if abs(one[2]) > 1e-9 and abs(one[3]) > 1e-9]
    said_nothing = [one for one in scored if not (abs(one[2]) > 1e-9 and abs(one[3]) > 1e-9)]
    opposed = [one for one in spoke if one[2] * one[3] < 0]
    apart = sorted((one for one in spoke if one[2] * one[3] > 0), key=lambda one: -abs(one[2] - one[3]))
    said_nothing.sort(key=lambda one: -abs(one[2] - one[3]))
    return [(a, b) for a, b, _, _ in (*opposed, *apart, *said_nothing)]


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
    key = _load(_key_path(stamp))
    if key is None:
        return {}
    # a sheet pruned to nothing still has its answers: they moved to the done file
    sheet = _sheet_path(stamp, "sitting")
    said = _picked(sheet) if sheet.exists() else {}
    said.update(_answered_earlier(stamp))
    out = {}
    for pair in key["pairs"]["sitting"]:
        answer = said.get(pair["n"])
        if not answer:
            continue
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
        sides = {"A": left.id, "B": right.id}
        mark = "____" if was is None else _letter_picked(sides, was)
        out.append(f"\nЛучше подкреплён контекстом (впиши A, B или `=`): **{mark}**\n")
    return "\n".join(out)


def build(on: date | None = None) -> dict:
    rows = _rows()
    order = _ordered(rows)
    rng = random.Random(SEED)
    sitting = [(*_side(p, rng), n) for n, p in enumerate(order[:BOTH_SPOKE], 1)]
    later = _repeats(sitting)

    stamp = (on or date.today()).strftime("%Y%m%d")
    HERE.mkdir(parents=True, exist_ok=True)
    kept = _already(stamp)
    first = HERE / f"human_anchor_{stamp}.md"
    second = HERE / f"human_anchor_repeats_{stamp}.md"

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
        "population": "corpus pool, answered, both our faithfulness and the guest's on the row",
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
    _store(where, key)
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
                pair.update(_covariates(rows[pair["A"]["log_id"]], rows[pair["B"]["log_id"]]))
                touched += 1
    _store(_key_path(stamp), key)
    return {"pairs_marked": touched}


def _dated(stamp: str) -> str:
    if not re.fullmatch(r"\d{8}", stamp):
        raise Refused(f"a stamp is a date like 20260908, not {stamp!r}")
    return stamp


def _load(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _store(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    hand_back(path)


def _key(stamp: str) -> dict:
    got = _load(_key_path(stamp))
    if got is None:
        raise Refused(f"no key for {stamp}: build the sheet before reading it")
    return got


def _key_path(stamp: str) -> Path:
    return HERE / f"human_anchor_key_{_dated(stamp)}.json"


def _sheet_path(stamp: str, which: str) -> Path:
    tail = "" if which == "sitting" else f"_{which}"
    return HERE / f"human_anchor{tail}_{_dated(stamp)}.md"


def _done_path(stamp: str) -> Path:
    return HERE / f"human_anchor_done_{_dated(stamp)}.json"


# the lock guarded `read` alone, while `prune` rewrote both the sheet and the fingerprint
def _its_own(stamp: str, key: dict, which: str) -> Path:
    sheet = _sheet_path(stamp, which)
    seen = _fingerprint(sheet.read_text(encoding="utf-8"))
    if which == "sitting" and key.get("sheet_fingerprint") not in (None, seen):
        raise Refused(
            "this sheet is not the one the key describes: rebuilt after it was filled, or edited"
            " beyond the answer lines. The pairs would join to the wrong rows"
        )
    return sheet


def _answered_earlier(stamp: str) -> dict:
    return {int(n): got["answer"] for n, got in (_load(_done_path(stamp), {})).items()}


# answered pairs leave the sheet, so the next sitting is only what is left
def prune(stamp: str) -> dict:
    key = _key(stamp)
    sheet = _its_own(stamp, key, "sitting")
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
    _store(done_path, done)

    text = sheet.read_text(encoding="utf-8")
    head, *blocks = re.split(r"(?=\n## Пара )", text)
    # numbers are never reused: the key joins by them, and a renumbered sheet would join wrong
    kept = [b for b in blocks if int(re.search(r"## Пара (\d+)", b).group(1)) not in said]
    sheet.write_text(head + "".join(kept), encoding="utf-8")
    hand_back(sheet)

    key["sheet_fingerprint"] = _fingerprint(sheet.read_text(encoding="utf-8"))
    _store(_key_path(stamp), key)
    return {"moved": len(said), "left": len(kept), "done_file": str(done_path)}


def _picked(sheet: Path) -> dict:
    out = {}
    n = None
    for line in sheet.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Пара "):
            n = int(line.split()[-1])
        if "подкреплён контекстом" in line and n is not None:
            said = line.rsplit("**", 2)[-2].strip().upper().strip("_")
            out[n] = said if said in ("A", "B", "=") else None
    return out


# a judge that called the pair equal where the human did not is a miss, not an abstention
def _verdict(delta: float, said: str) -> str:
    if said == "=":
        return "agreed" if delta == 0 else "missed"
    if delta == 0:
        return "did_not_separate"
    return "agreed" if (delta > 0) == (said == "A") else "disagreed"


def read(stamp: str) -> dict:
    key = _key(stamp)
    sheet = _its_own(stamp, key, "sitting")
    answered = _answered_earlier(stamp)
    answered.update({n: v for n, v in _picked(sheet).items() if v})
    tally = {j: defaultdict(int) for j in ("ours", "guest")}
    buckets = ("same_language", "cross_language", "clean", "template_leak",
               "hit_gold", "neither_hit_gold")
    split = {j: {b: defaultdict(int) for b in buckets} for j in ("ours", "guest")}
    for pair in key["pairs"]["sitting"]:
        said = answered.get(pair["n"])
        if not said:
            continue
        where = [
            "cross_language" if pair.get("cross_language") else "same_language",
            "template_leak" if pair.get("template_leak") else "clean",
            "neither_hit_gold" if pair.get("neither_hit_gold") else "hit_gold",
        ]
        for judge in ("ours", "guest"):
            got = _verdict(pair[judge], said)
            tally[judge][got] += 1
            for bucket in where:
                split[judge][bucket][got] += 1

    def share(counts) -> dict:
        counted = sum(counts.values())
        return {
            "n": counted,
            "agreed": counts["agreed"],
            "share": round(counts["agreed"] / counted, 4) if counted else None,
            "by_outcome": dict(counts),
        }

    def judged(judge) -> dict:
        return {**share(tally[judge]),
                "by_covariate": {k: share(v) for k, v in split[judge].items()}}

    return {
        "schema": SCHEMA,
        "population": key["population"],
        # declared before the sheet was filled, not carved out of the result
        "subgroups": "declared before the sheet was filled: same against cross language (the judge"
                     " loses 0.964 of a point on Russian at the same meaning), and clean against a"
                     " scaffolding leak in the answer body",
        "pairs_offered": len(key["pairs"]["sitting"]),
        "pairs_answered": len(answered),
        "ours": judged("ours"),
        "guest": judged("guest"),
    }


# the same row picked twice is content, the same letter picked twice is position
def repeats(stamp: str) -> dict:
    key = _key(stamp)
    # the same source `_already` uses: an answer still sitting in the sheet counts too
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
