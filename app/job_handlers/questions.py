import json
from collections import Counter

import config
import job_queue
import llm
import logging_setup
import prompt_repo
from evals import measurements, question_sets, section_questions
from models.eval import Question
from models.registry import Purpose, Role
from orm.sync_db import Session
from sqlalchemy import select
from use_cases import gold_reanchor, section_export

from .base import register, reported, require_role_ready, stamp

log = logging_setup.get_logger(__name__)


# a source's question pairs from its sections, each section one call and written as it lands; the set's report beside
@register("generate_questions")
def generate_questions(options: dict) -> dict | None:
    # a cloud seat needs its row ready, never the card
    require_role_ready(Role.questioning, take_card=False)
    source, set_name = options["source"], options["set_name"]
    exported = section_export.of_source(source)
    wanted = section_questions.section_pairs(exported["sections"], exported["quota"])
    if cap := options.get("max_pairs"):
        wanted = _capped(wanted, cap)
    # a smoke asks the sections the cap would pick first, in that order, until this many pairs survive the checks
    enough = options.get("kept_at_least")
    if enough:
        wanted = dict(sorted(wanted.items(), key=lambda kv: -kv[1]))
    template, version = prompt_repo.active(Purpose.questions_from_section)
    picked = llm.resolve(Role.questioning)
    rows = {section_questions.section_key(r): r for r in exported["sections"]}
    anchored_in = section_questions.anchors_for(exported["sections"])
    languages = section_questions.set_languages(options.get("languages"))
    # a retried job, or a second pass into one set, asks a section only for the pairs the set does not hold yet
    held = _held_by_section(set_name, languages[0])
    sections, tally = [], Counter()

    def finish(stopped: Exception | None) -> dict:
        # a stopped pass may have written pairs its report does not count, and they need their vectors all the same
        if tally["written"] or stopped:
            # the compare door reads embedded questions only; a run embeds its own at ask time
            job_queue.enqueue("embed_questions", {})
        pairs_kept = sum(s["kept"] for s in sections)
        summary = {
            "source": source,
            "set_name": set_name,
            **stamp(Role.questioning, picked, template, version),
            "languages": list(languages),
            "sections_exported": len(exported["sections"]),
            "sections_refused": exported["refused"],
            "stubs": sum(1 for r in exported["sections"] if r["words"] < section_questions.MIN_WORDS),
            "pairs_asked": sum(s["asked"] for s in sections),
            "pairs_kept": pairs_kept,
            "pairs_already_held": tally["already"],
            "pairs_dropped_as_repeats": tally["dropped"],
            "refused_by_reason": dict(Counter(why for s in sections for why in s["refused"])),
            # pairs, not replies: the kept, the repeats and these add up to the pairs asked
            "pairs_lost_by_reason": dict(sum((Counter(s["lost"]) for s in sections), Counter())),
            "questions_written": tally["written"],
            # this pass's own pairs; the set's standing against the floor is `question_sets`'s to say
            "kept_under_the_floor": pairs_kept < config.settings.evals.question_set.min_pairs,
            "stopped_by": repr(stopped) if stopped else None,
        }
        report = measurements.record(
            "question_set", f"{set_name}_{source}", {**summary, "sections": sections}, bulk=("sections",)
        )
        log.info("questions.written", **{k: summary[k] for k in ("source", "pairs_asked", "pairs_kept")})
        return {**summary, "report": report}

    def ask_sections() -> None:
        for key, pairs in wanted.items():
            if options.get("_job_id") is not None and job_queue.is_cancelled(options["_job_id"]):
                break
            if enough and sum(s["kept"] for s in sections) + tally["already"] >= enough:
                break
            prior = held.get(key, [])[:pairs]
            tally["already"] += len(prior)
            if pairs == len(prior):
                continue
            sections.append(_generate_section(key, rows[key], pairs - len(prior), prior, set_name, source, template,
                                              anchored_in, languages))
            tally["written"] += sections[-1].pop("written")
            tally["dropped"] += sections[-1].pop("dropped")

    return reported(ask_sections, finish)


# one section asked block by block, its pairs written as they land; the set's own pairs of it are not asked again
def _generate_section(key, row, pairs, prior, set_name, source, template, anchored_in, languages) -> dict:
    kept_here, refused, calls, written, dropped = list(prior), [], [], 0, 0
    parts = row.get("blocks") or [row["text"]]
    for block, (text, count) in enumerate(zip(parts, section_questions.pairs_by_block(pairs, parts), strict=True)):
        part = {**row, "text": text}
        block_sha = section_questions.block_sha(text)
        for ask in section_questions.asks(count):
            try:
                reply = llm.ask(
                    system=section_questions.prompt(template, source, ask, languages),
                    user=section_questions.user_turn(part, [pair[languages[0]] for pair in kept_here]),
                    role=Role.questioning,
                )
            # one block the model refuses is lost alone, and the run goes on
            except llm.RequestRefused:
                refused.append({"why": section_questions.REFUSED_CALL, "count": ask})
                calls.append({"asked": ask, "block": block, "block_sha": block_sha, "kept": 0,
                              "finish_reason": None, "tokens": None, "reply": None, "reasoning_chars": None})
                continue
            kept, lost = section_questions.parse(
                reply.text, part, ask, [pair["evidence"] for pair in kept_here], languages
            )
            # a cut reply ends in broken json: the budget, not the model, is what failed there
            if reply.finish_reason == "length" and not kept:
                lost = [{"why": "the reply was cut at max_tokens", "count": ask}]
            section_rows = [q for pair in kept for q in section_questions.question_rows(
                pair, row, set_name, anchored_in(row), section_questions.placed_in(block, text), languages)]
            done, gone = question_sets.write_pairs(section_rows)
            written, dropped = written + done, dropped + gone
            kept_here += kept
            refused += lost
            calls.append({
                "asked": ask, "block": block, "block_sha": block_sha, "kept": len(kept) - gone,
                "finish_reason": reply.finish_reason,
                "tokens": [reply.prompt_tokens, reply.completion_tokens],
                # the reply as it came, so a refusal or a long answer can be read after the fact
                "reply": reply.text,
                "reasoning_chars": reply.parsed.reasoning_chars if reply.parsed else None,
            })
    return {
        "file": key[0], "section": key[1], "version": key[2], "asked": pairs, "held": len(prior),
        "blocks": len(parts), "kept": sum(c["kept"] for c in calls),
        "refused": [r["why"] for r in refused], "lost": _lost(refused), "calls": calls,
        "written": written, "dropped": dropped,
    }


# a generation's stored replies read again by today's checks, the pairs they hold now written; nothing is asked anew
@register("reparse_questions")
def reparse_questions(options: dict) -> dict | None:
    source, set_name = options["source"], options["set_name"]
    path = measurements.FOLDER / options["report"]
    stored = json.loads(path.read_text())
    if (stored.get("source"), stored.get("set_name")) != (source, set_name):
        raise ValueError(f"{path.name} is the report of {stored.get('set_name')} on {stored.get('source')}")
    exported = section_export.of_source(source)["sections"]
    rows = {section_questions.section_key(r): r for r in exported}
    anchored_in = section_questions.anchors_for(exported)
    # the languages the stored replies were asked in; a report older than the option asked the two
    languages = tuple(stored.get("languages") or ("en", "ru"))
    taken_by_section = _held_by_section(set_name, languages[0])
    # a report older than the version field names a section by file and heading, sound while one stream has it
    by_place = Counter((file, section) for file, section, _ in rows)
    unversioned = {(file, section): r for (file, section, _), r in rows.items() if by_place[(file, section)] == 1}
    kept_now, written, dropped, gone, moved = 0, 0, 0, 0, 0
    for sec in measurements.rows_of(path, "sections"):
        if "version" in sec:
            row = rows.get((sec["file"], sec["section"], sec["version"]))
        else:
            row = unversioned.get((sec["file"], sec["section"]))
        if row is None:
            gone += 1
            continue
        # a reply is judged against the text it was written from; a block the intake has moved since is not that text
        now = [section_questions.block_sha(b) for b in row.get("blocks") or [row["text"]]]
        if any(c.get("block_sha") and (c.get("block", 0) >= len(now) or now[c["block"]] != c["block_sha"])
               for c in sec["calls"]):
            moved += 1
            continue
        taken = [pair["evidence"] for pair in taken_by_section.get(section_questions.section_key(row), [])]
        parts = row.get("blocks") or [row["text"]]
        for call in sec["calls"]:
            # a reply read by a block is checked against that block, as the generation checked it
            block = call.get("block", 0) if call.get("block_sha") else None
            read = {**row, "text": parts[block]} if block is not None else row
            kept, _ = section_questions.parse(call["reply"] or "", read, call["asked"], taken, languages)
            taken += [pair["evidence"] for pair in kept]
            placed = section_questions.placed_in(block, parts[block]) if block is not None else None
            done, repeat = question_sets.write_pairs([q for pair in kept for q in section_questions.question_rows(
                pair, row, set_name, anchored_in(row), placed, languages)])
            kept_now, written, dropped = kept_now + len(kept), written + done, dropped + repeat
    if written:
        job_queue.enqueue("embed_questions", {})
    summary = {"source": source, "set_name": set_name, "report": path.name, "pairs_kept_now": kept_now,
               "questions_written": written, "pairs_dropped_as_repeats": dropped, "sections_gone": gone,
               "sections_moved": moved}
    log.info("questions.reparsed", **summary)
    return {**summary, "record": measurements.record("question_reparse", f"{set_name}_{source}", summary)}


# a set's rows given the identifiers their gold sections hold, by the rule generation writes them with
@register("anchor_questions")
def anchor_questions(options: dict) -> dict | None:
    source, set_name = options["source"], options["set_name"]
    exported = section_export.of_source(source)["sections"]
    rows = {section_questions.section_key(r): r for r in exported}
    anchored_in = section_questions.anchors_for(exported)
    counted = Counter()
    with Session() as session:
        for question in session.scalars(select(Question).where(Question.set_name == set_name)):
            row = rows.get(section_questions.gold_key(question.gold))
            if row is None:
                counted["section_gone"] += 1
                continue
            question.anchors = anchored_in(row)(question.original_text)
            counted["anchored" if question.anchors else "none"] += 1
        session.commit()
    summary = {"source": source, "set_name": set_name, **counted}
    log.info("questions.anchored", **summary)
    return summary


# a set's golds moved to the section of the same file and leaf after a cleanup renamed the page's root
@register("reanchor_questions")
def reanchor_questions(options: dict) -> dict | None:
    variant = options.get("variant") or config.settings.corpus.variant
    summary = gold_reanchor.reanchor(options["set_name"], variant, dry=options.get("dry", False))
    log.info("questions.reanchored", **{k: v for k, v in summary.items() if k != "sections"})
    return {**summary, "record": measurements.record("question_reanchor", options["set_name"], summary)}


# the pairs a set already holds, by section, as the generator keeps them: a pair on their evidence is the same fact
def _held_by_section(set_name: str, language: str) -> dict[tuple, list[dict]]:
    with Session() as session:
        found = session.execute(
            select(Question.gold, Question.original_text, Question.evidence)
            .where(Question.set_name == set_name, Question.language == language, Question.pair_id.is_not(None))
            .order_by(Question.id)
        ).all()
    out: dict[tuple, list[dict]] = {}
    for gold, text, evidence in found:
        if gold and evidence:
            out.setdefault(section_questions.gold_key(gold), []).append({language: text, "evidence": evidence})
    return out


# the pairs each refusal cost; a reply that ran past its ask lost nothing
def _lost(refused: list[dict]) -> dict:
    out: Counter = Counter()
    for r in refused:
        if r["why"] != section_questions.PAST_THE_ASK:
            out[r["why"]] += r.get("count", 1)
    return dict(out)


# a probe's cap taken from the sections with the most pairs first, one at a time, so it spreads as the quota does
def _capped(wanted: dict, cap: int) -> dict:
    left, out = dict(wanted), {}
    for _ in range(min(cap, sum(wanted.values()))):
        key = max(left, key=lambda k: left[k])
        out[key] = out.get(key, 0) + 1
        left[key] -= 1
    return out
