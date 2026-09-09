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
import sys
from collections import defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

from evals.loaders import load_logs
from evals.measurements import hand_back, record
from evals.pools import in_corpus_and_answered

GUEST = "ragas_faithfulness"
SEED = 20260908
BOTH_SPOKE = 15
REPEATS = 3
HERE = Path(__file__).resolve().parents[1] / "temp_files"


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


# the judge loses 0.964 of a point on Russian at the same meaning, so language is a covariate here
def _hit_gold(ql) -> bool:
    gold = set((ql.question.marked_sources if ql.question else None) or [])
    return bool(gold & {x.get("source") for x in (ql.sources or [])})


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


def _language(text: str) -> str:
    import db

    return db.detect_language(text or "")


def _deltas(a, b) -> tuple[float, float]:
    return int(a.faithfulness) / 10 - int(b.faithfulness) / 10, _guest(a) - _guest(b)


# declared before the owner sees a single answer: contested first, then whatever separates most
def _ordered(rows) -> list:
    by_question = defaultdict(list)
    for ql in rows:
        by_question[ql.question_id].append(ql)
    pairs = [
        (a, b)
        for kept in by_question.values()
        for a, b in combinations(sorted(kept, key=lambda x: x.id), 2)
    ]
    spoke = [p for p in pairs if all(abs(d) > 1e-9 for d in _deltas(*p))]
    opposed = [p for p in spoke if _deltas(*p)[0] * _deltas(*p)[1] < 0]
    rest = [p for p in pairs if p not in spoke]
    agreed = sorted(
        (p for p in spoke if p not in opposed),
        key=lambda p: -abs(_deltas(*p)[0] - _deltas(*p)[1]),
    )
    rest.sort(key=lambda p: -abs(_deltas(*p)[0] - _deltas(*p)[1]))
    return [*opposed, *agreed, *rest]


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
    key_path = HERE / f"human_anchor_key_{stamp}.json"
    if not key_path.exists():
        return {}
    key = json.loads(key_path.read_text(encoding="utf-8"))
    sheet = Path(key["sheets"]["sitting"])
    if not sheet.exists():
        return {}
    said = _picked(sheet)
    # pruned answers live in the done file, and a rebuild that forgot them would ask them again
    done_path = _done_path(stamp)
    if done_path.exists():
        earlier = json.loads(done_path.read_text(encoding="utf-8"))
        said.update({int(n): got["answer"] for n, got in earlier.items()})
    out = {}
    for pair in key["pairs"]["sitting"]:
        answer = said.get(pair["n"])
        if not answer:
            continue
        sides = {"A": pair["A"]["log_id"], "B": pair["B"]["log_id"]}
        out[frozenset(sides.values())] = "=" if answer == "=" else sides[answer]
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
        mark = "____" if was is None else (
            "=" if was == "=" else ("A" if was == left.id else "B")
        )
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
            kept,
        ),
        encoding="utf-8",
    )
    for path in (first, second):
        hand_back(path)

    key = {
        "population": "corpus pool, answered, both our faithfulness and the guest's on the row",
        "seed": SEED,
        "ordering": "opposed first, then both judges spoke, then by |ours - guest| descending",
        "sheets": {"sitting": str(first), "repeats": str(second)},
        "pairs": {
            sheet: [
                {
                    "n": n,
                    "A": {"log_id": left.id, "run": left.run_name,
                          "lang": _language(left.answer), "leak": _leak(left)},
                    "B": {"log_id": right.id, "run": right.run_name,
                          "lang": _language(right.answer), "leak": _leak(right)},
                    "ours": round(_deltas(left, right)[0], 4),
                    "guest": round(_deltas(left, right)[1], 4),
                    "cross_language": _language(left.answer) != _language(right.answer),
                    # neither side found the gold file: both are ungrounded by construction
                    "neither_hit_gold": not _hit_gold(left) and not _hit_gold(right),
                    "template_leak": bool(_leak(left)) or bool(_leak(right)),
                }
                for left, right, n in entries
            ]
            for sheet, entries in (("sitting", sitting), ("repeats", later))
        },
    }
    # the sheet he filled must be the sheet this key describes, or the pairs join to the wrong rows
    key["sheet_md5"] = _fingerprint(first.read_text(encoding="utf-8"))
    where = HERE / f"human_anchor_key_{stamp}.json"
    where.write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    hand_back(where)
    return {"sheet": str(first), "repeats": str(second), "key": str(where), "pairs": len(sitting)}


# the answer lines are what he changes, so they cannot be part of what identifies the sheet
def _fingerprint(text: str) -> str:
    body = "\n".join(
        "ANSWER LINE" if "подкреплён контекстом" in line else line
        for line in text.splitlines()
    )
    return hashlib.md5(body.encode("utf-8")).hexdigest()


# covariates onto an existing key, without regenerating a sheet somebody is in the middle of
def mark(stamp: str) -> dict:
    from models.eval import QuestionLog
    from orm.sync_db import Session
    from sqlalchemy import select

    key_path = HERE / f"human_anchor_key_{stamp}.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    wanted = {side["log_id"] for group in key["pairs"].values() for p in group
              for side in (p["A"], p["B"])}
    with Session() as session:
        rows = {ql.id: ql for ql in
                session.scalars(select(QuestionLog).where(QuestionLog.id.in_(wanted)))}
        touched = 0
        for group in key["pairs"].values():
            for pair in group:
                left, right = rows[pair["A"]["log_id"]], rows[pair["B"]["log_id"]]
                for name, ql in (("A", left), ("B", right)):
                    pair[name]["lang"] = _language(ql.answer)
                    pair[name]["leak"] = _leak(ql)
                pair["cross_language"] = _language(left.answer) != _language(right.answer)
                pair["neither_hit_gold"] = not _hit_gold(left) and not _hit_gold(right)
                pair["template_leak"] = bool(_leak(left)) or bool(_leak(right))
                touched += 1
    key_path.write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    hand_back(key_path)
    return {"pairs_marked": touched}


def _done_path(stamp: str) -> Path:
    return HERE / f"human_anchor_done_{stamp}.json"


# answered pairs leave the sheet so the next sitting is only what is left, and land here instead
def prune(stamp: str) -> dict:
    key_path = HERE / f"human_anchor_key_{stamp}.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    sheet = Path(key["sheets"]["sitting"])
    said = {n: v for n, v in _picked(sheet).items() if v}
    if not said:
        return {"moved": 0, "left": len(key["pairs"]["sitting"])}

    done_path = _done_path(stamp)
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
    for pair in key["pairs"]["sitting"]:
        if pair["n"] in said:
            done[str(pair["n"])] = {
                "answer": said[pair["n"]],
                "A": pair["A"]["log_id"], "B": pair["B"]["log_id"],
            }
    done_path.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")
    hand_back(done_path)

    text = sheet.read_text(encoding="utf-8")
    head, *blocks = re.split(r"(?=\n## Пара )", text)
    # numbers are never reused: the key joins by them, and a renumbered sheet would join wrong
    kept = [b for b in blocks if int(re.search(r"## Пара (\d+)", b).group(1)) not in said]
    sheet.write_text(head + "".join(kept), encoding="utf-8")
    hand_back(sheet)

    key["sheet_md5"] = _fingerprint(sheet.read_text(encoding="utf-8"))
    key_path.write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    hand_back(key_path)
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
    key = json.loads((HERE / f"human_anchor_key_{stamp}.json").read_text(encoding="utf-8"))
    sheet = Path(key["sheets"]["sitting"])
    seen = _fingerprint(sheet.read_text(encoding="utf-8"))
    if key.get("sheet_md5") and seen != key["sheet_md5"]:
        raise SystemExit(
            "this sheet is not the one the key describes: rebuilt after it was filled, or edited"
            " beyond the answer lines. The pairs would join to the wrong rows"
        )
    done_path = _done_path(stamp)
    earlier = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
    answered = {int(n): got["answer"] for n, got in earlier.items()}
    answered.update({n: v for n, v in _picked(sheet).items() if v})
    tally = {j: defaultdict(int) for j in ("ours", "guest")}
    buckets = ("same_language", "cross_language", "clean", "template_leak")
    split = {j: {b: defaultdict(int) for b in buckets} for j in ("ours", "guest")}
    for pair in key["pairs"]["sitting"]:
        said = answered.get(pair["n"])
        if not said:
            continue
        where = [
            "cross_language" if pair.get("cross_language") else "same_language",
            "template_leak" if pair.get("template_leak") else "clean",
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
    key = json.loads((HERE / f"human_anchor_key_{stamp}.json").read_text(encoding="utf-8"))
    before = {}
    done_path = _done_path(stamp)
    if done_path.exists():
        for got in json.loads(done_path.read_text(encoding="utf-8")).values():
            sides = {"A": got["A"], "B": got["B"]}
            before[frozenset(sides.values())] = "=" if got["answer"] == "=" else sides[got["answer"]]
    said = _picked(Path(key["sheets"]["repeats"]))

    tally = defaultdict(int)
    for pair in key["pairs"]["repeats"]:
        now = said.get(pair["n"])
        was = before.get(frozenset((pair["A"]["log_id"], pair["B"]["log_id"])))
        if not now or was is None:
            continue
        picked = "=" if now == "=" else pair[now]["log_id"]
        if picked == was:
            tally["same_row"] += 1
        elif was == "=" or picked == "=":
            tally["one_side_called_it_equal"] += 1
        else:
            tally["same_position"] += 1
    counted = sum(tally.values())
    return {
        "population": "the pairs of the main sheet, flipped: cross language first, then the rest",
        "n": counted,
        "share_same_row": round(tally["same_row"] / counted, 3) if counted else None,
        "by_outcome": dict(tally),
        "reads": "same_row means he follows what is written, same_position means he follows the side",
    }


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "build"
    if action == "build":
        print(json.dumps(build(), indent=2, ensure_ascii=False))
    elif action == "repeats":
        got = repeats(sys.argv[2] if len(sys.argv) > 2 else date.today().strftime("%Y%m%d"))
        print(json.dumps(got, indent=2, ensure_ascii=False))
        if "--record" in sys.argv:
            print("записано:", record("human_anchor_repeats", "arc4", got))
    elif action == "mark":
        print(json.dumps(mark(sys.argv[2] if len(sys.argv) > 2
                              else date.today().strftime("%Y%m%d")), ensure_ascii=False))
    elif action == "prune":
        stamp = sys.argv[2] if len(sys.argv) > 2 else date.today().strftime("%Y%m%d")
        print(json.dumps(prune(stamp), indent=2, ensure_ascii=False))
    else:
        got = read(sys.argv[2] if len(sys.argv) > 2 else date.today().strftime("%Y%m%d"))
        print(json.dumps(got, indent=2, ensure_ascii=False))
        # a dry read must not leave a number behind: `record` is asked for, never a side effect
        if "--record" in sys.argv:
            print("записано:", record("human_anchor", "arc4", got))
        else:
            print("не записано; добавь --record, когда лист заполнен по-настоящему")
