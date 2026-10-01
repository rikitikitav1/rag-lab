from collections import Counter

import config
import engines
import llm
import logging_setup
import prompt_repo
from evals import measurements, pair_judge, question_acceptance, section_questions
from models.eval import ACCEPTED, CANDIDATE, REFUSED, Question
from models.registry import Purpose, Role
from orm.sync_db import Session
from passes import Pass, Seat
from sqlalchemy import or_, select, update
from use_cases import section_export

from .base import register, reported, require_role_ready, stamp

log = logging_setup.get_logger(__name__)


# a set's candidate pairs of one source read from their section by the grader's model, a pair settled whole
@register("accept_questions")
def accept_questions(options: dict) -> dict | None:
    require_role_ready(Role.accepting)
    source, set_name = options["source"], options["set_name"]
    texts = {section_questions.section_key(r): r["text"] for r in section_export.of_source(source)["sections"]}
    pairs = _pairs_to_read(options, set_name, {key[0] for key in texts}, again=bool(options.get("again")))
    template, version = prompt_repo.active(Purpose.answer_from_section)
    picked = llm.resolve(Role.accepting)
    rows, outcomes = [], Counter()
    walk = Pass(options.get("_job_id"), (Seat(Role.accepting, spill_from=None),))

    def read_pairs() -> None:
        for pair_id, members in pairs.items():
            if not walk.before_row():
                break
            text = texts.get(section_questions.gold_key(members[0].gold))
            # a section the export refuses now: settled once, or every rerun would read and skip it again
            if text is None:
                outcomes["refused"] += 1
                if options.get("settle", True):
                    gone = {"answerable": None, "why": question_acceptance.SECTION_GONE}
                    _settle(members, [gone] * len(members), "refused", question_acceptance.SECTION_GONE)
                continue
            verdicts, read_here = [], []
            for question in members:
                reply, said, read = _read(question, text, template)
                verdicts.append(said)
                read_here.append({"question_id": question.id, "pair_id": pair_id, "language": question.language,
                                  "reply": reply.text if reply else None, "windows_read": read, **said})
            outcome, why = question_acceptance.pair_outcome(verdicts)
            outcomes[outcome] += 1
            rows.extend({**row, "outcome": outcome} for row in read_here)
            if options.get("settle", True):
                _settle(members, verdicts, outcome, why)

    def finish(stopped: Exception | None) -> dict:
        summary = {
            "source": source,
            "set_name": set_name,
            **stamp(Role.accepting, picked, template, version),
            "evidence_held": question_acceptance.EVIDENCE_HELD,
            "window_words": question_acceptance.WINDOW_WORDS,
            "settled": options.get("settle", True),
            "pairs_read": sum(outcomes[k] for k in ("accepted", "refused", "undecided")),
            "pairs": dict(outcomes),
            "by_language": _by_language(rows),
            "why": dict(Counter(r["why"] for r in rows if r["why"])),
        }
        summary["set"] = _set_counts(set_name, {key[0] for key in texts})
        # this source's accepted pairs; the whole set's standing against the floor is `question_sets`'s to say
        summary["source_under_the_floor"] = (
            summary["set"]["accepted_pairs"] < config.settings.evals.question_set.min_pairs
        )
        summary["stopped_by"] = repr(stopped) if stopped else None
        report = measurements.record(
            "question_acceptance", f"{set_name}_{source}", {**summary, "rows": rows}, bulk=("rows",)
        )
        log.info("questions.accepted", source=source, **summary["pairs"])
        return {**summary, "report": report}

    return reported(read_pairs, finish)


# the pairs the sieve left undecided, each half judged on its evidence in its passage, a pair settled whole
@register("judge_questions")
def judge_questions(options: dict) -> dict | None:
    model = options.get("model")
    picked = llm.resolve_for(Role.judging, model)
    # a cloud judge needs its row ready, never the card
    if model is None or not engines.is_cloud(picked.engine.kind):
        require_role_ready(Role.judging)
    source, set_name = options["source"], options["set_name"]
    exported = section_export.of_source(source)["sections"]
    texts = {section_questions.section_key(r): r["text"] for r in exported}
    blocks = {section_questions.section_key(r): r.get("blocks") or [] for r in exported}
    # the judge reads what the sieve has read: run first, it would settle pairs the reader never saw
    pairs = _pairs_to_read(options, set_name, {key[0] for key in texts}, again=True, sieved=True)
    template, version = prompt_repo.active(Purpose.judge_pair)
    rows, outcomes = [], Counter()
    walk = Pass(options.get("_job_id"), (Seat(Role.judging, model, spill_from=None),))

    def judge_pairs() -> None:
        for pair_id, members in pairs.items():
            if not walk.before_row():
                break
            key = section_questions.gold_key(members[0].gold)
            # a gone section is the sieve's to settle; the judge finding nothing there would hand the pair back
            if key not in texts:
                outcomes["section_gone"] += 1
                continue
            judged = [_judge_half(question, texts[key], blocks[key], template, model) for question in members]
            outcome, why = pair_judge.pair_outcome([half["said"] for half in judged])
            for reason in (pair_judge.REFUSED_CALL, pair_judge.NOT_FOUND):
                if any(half["why"] == reason for half in judged):
                    outcome, why = "undecided", reason
            outcomes[outcome] += 1
            rows.extend({"question_id": question.id, "pair_id": pair_id, "language": question.language,
                         "reply": half["reply"], "found": half["why"] != pair_judge.NOT_FOUND, "said": half["said"],
                         "outcome": outcome} for question, half in zip(members, judged, strict=True))
            if options.get("settle", True):
                _settle_judged(members, outcome, why)

    def finish(stopped: Exception | None) -> dict:
        summary = {
            "source": source,
            "set_name": set_name,
            **stamp(Role.judging, picked, template, version),
            "around_words": pair_judge.AROUND_WORDS,
            "second_judge": model is not None,
            "settled": options.get("settle", True),
            "pairs": dict(outcomes),
            "said_by_language": {
                lang: dict(Counter(str(r["said"]) for r in rows if r["language"] == lang))
                for lang in sorted({r["language"] for r in rows})
            },
        }
        summary["set"] = _set_counts(set_name, {key[0] for key in texts})
        summary["stopped_by"] = repr(stopped) if stopped else None
        report = measurements.record(
            "question_judgement", f"{set_name}_{source}", {**summary, "rows": rows}, bulk=("rows",)
        )
        log.info("questions.judged", source=source, **summary["pairs"])
        return {**summary, "report": report}

    return reported(judge_pairs, finish)


# one half read by the judge in its passage; a request the engine refuses is this half undecided, not the pass
def _judge_half(question, text: str, blocks: list[str], template: str, model: str | None) -> dict:
    passage = _passage(question, text, blocks)
    if passage is None:
        return {"said": None, "reply": None, "why": pair_judge.NOT_FOUND}
    try:
        reply = llm.ask(
            system=template,
            user=pair_judge.user_turn(question.original_text, question.evidence, passage),
            role=Role.judging,
            model=model,
        )
    except llm.RequestRefused:
        return {"said": None, "reply": None, "why": pair_judge.REFUSED_CALL}
    return {"said": pair_judge.verdict(reply.text), "reply": reply.text, "why": None}


# the place the generator read when the block is still that block; the section searched otherwise
def _passage(question, text: str, blocks: list[str]) -> str | None:
    at = getattr(question, "evidence_at", None) or {}
    block = at.get("block")
    if at.get("char") is not None and block is not None and block < len(blocks):
        if section_questions.block_sha(blocks[block]) == at.get("block_sha"):
            return pair_judge.around(blocks[block], question.evidence or "", at["char"])
    return pair_judge.around(text, question.evidence or "")


# the pairs a pass reads; a pass that only reports reads the whole set, settled pairs too, to sit beside another
def _pairs_to_read(options: dict, set_name: str, files: set, *, again: bool, sieved: bool = False) -> dict:
    every = bool(options.get("every")) or not options.get("settle", True)
    pairs = _candidate_pairs(set_name, files, again=again, every=every, sieved=sieved)
    if cap := options.get("max_pairs"):
        pairs = dict(list(pairs.items())[:cap])
    return pairs


# a pair's outcome as its questions' status: one left undecided waits as a candidate
def _status_of(outcome: str):
    return {"accepted": ACCEPTED, "refused": REFUSED}.get(outcome, CANDIDATE)


# the judge's word moves the status and the reason; the reader's own word on each row stays as it read
def _settle_judged(members: list, outcome: str, why: str | None) -> None:
    status = _status_of(outcome)
    with Session() as session:
        session.execute(
            update(Question).where(Question.id.in_([q.id for q in members])).values(status=status, acceptance_why=why)
        )
        session.commit()


# the windows of the section in order until the reader answers from one; the last word stands when none does
def _read(question, text: str, template: str) -> tuple:
    reply, said, read = None, None, 0
    for part in question_acceptance.windows(text):
        read += 1
        try:
            reply = llm.ask(
                system=template,
                user=question_acceptance.user_turn(question.original_text, question.gold["section"], part),
                role=Role.accepting,
            )
        # a window past the model's own is the judge's to read, not a reason to stop the pass
        except llm.InputOverWindow:
            reply = None
            said = {"answerable": None, "why": question_acceptance.TOO_LONG, "held": None, "f1": None}
            continue
        # a request the engine refuses is this window unread; the next one may pass
        except llm.RequestRefused:
            reply = None
            said = {"answerable": None, "why": question_acceptance.REFUSED_CALL, "held": None, "f1": None}
            continue
        said = question_acceptance.verdict(reply.text, reply.finish_reason, question.evidence)
        if said["answerable"]:
            break
    return reply, said, read


# a set's unread candidates whose gold lies in the source's files, by pair in writing order
def _candidate_pairs(
    set_name: str, files: set, again: bool = False, every: bool = False, sieved: bool = False
) -> dict[str, list]:
    wanted = [Question.set_name == set_name, Question.pair_id.is_not(None)]
    if not every:
        wanted.append(Question.status == CANDIDATE)
    if not (again or every):
        wanted.append(Question.acceptance_why.is_(None))
    # a candidate the sieve has read always carries its reason; a settled pair was read to be settled
    if sieved:
        wanted.append(or_(Question.status != CANDIDATE, Question.acceptance_why.is_not(None)))
    with Session() as session:
        found = session.scalars(select(Question).where(*wanted).order_by(Question.id)).all()
        session.expunge_all()
    pairs: dict[str, list] = {}
    for question in found:
        if question.gold and question.gold["file"] in files:
            pairs.setdefault(question.pair_id, []).append(question)
    return pairs


def _settle(members: list, verdicts: list[dict], outcome: str, why: str | None) -> None:
    status = _status_of(outcome)
    with Session() as session:
        for question, said in zip(members, verdicts, strict=True):
            session.execute(
                update(Question)
                .where(Question.id == question.id)
                .values(answerable_by_reader=said["answerable"], status=status, acceptance_why=why)
            )
        session.commit()


def _by_language(rows: list[dict]) -> dict:
    out: dict = {}
    for row in rows:
        counts = out.setdefault(row["language"], Counter())
        counts["asked"] += 1
        counts["answered"] += row["answerable"] is True
        counts["no_answer"] += row["answerable"] is False
        counts["evidence_held"] += row["answerable"] is True and row["why"] is None
        counts["undecided_halves"] += row["outcome"] == "undecided"
    return {language: dict(counts) for language, counts in out.items()}


# the set of this source as it stands after the pass, pairs by status and questions refused by reason
def _set_counts(set_name: str, files: set) -> dict:
    with Session() as session:
        found = session.execute(
            select(Question.pair_id, Question.status, Question.acceptance_why, Question.gold).where(
                Question.set_name == set_name, Question.pair_id.is_not(None)
            )
        ).all()
    mine = [row for row in found if row.gold and row.gold["file"] in files]
    by_status = Counter(str(row.status) for row in {(r.pair_id, r.status): r for r in mine}.values())
    return {
        "pairs": dict(by_status),
        "accepted_pairs": by_status.get(str(ACCEPTED), 0),
        "why_by_status": {
            str(status): dict(Counter(r.acceptance_why for r in mine if r.status == status and r.acceptance_why))
            for status in (REFUSED, CANDIDATE)
        },
    }
