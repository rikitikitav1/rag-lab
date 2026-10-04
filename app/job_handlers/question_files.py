import json
from pathlib import Path

import config
import job_queue
from evals import question_sets, section_questions
from models.eval import CANDIDATE, Question, text_hash
from orm.sync_db import Session
from sqlalchemy import select
from use_cases import section_export

from .base import Final, register

# a generated set beside its sources: generated once, poured back on a later intake without asking anyone again
SAVED = Path(config.beside_config("sources")) / "questions"
# the evidence's place goes along: the judge opens it while its block still matches, and searches the section if not
_KEPT_FIELDS = ("original_text", "language", "gold", "reference_answer", "evidence", "pair_id", "kind", "evidence_at")


@register("save_questions")
def save_questions(options: dict) -> dict | None:
    set_name = options["set_name"]
    with Session() as session:
        rows = session.scalars(
            select(Question).where(Question.set_name == set_name, Question.pair_id.is_not(None)).order_by(Question.id)
        ).all()
        lines = [json.dumps({k: getattr(q, k) for k in _KEPT_FIELDS}, ensure_ascii=False) for q in rows]
    if not lines:
        raise Final(f"set {set_name} holds no generated pairs to save")
    SAVED.mkdir(parents=True, exist_ok=True)
    path = SAVED / f"{set_name}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"set_name": set_name, "questions": len(lines), "file": str(path)}


# a saved set written back as candidates, acceptance and the judge to read them again; a pair of a gone section counted
@register("load_questions")
def load_questions(options: dict) -> dict | None:
    source, set_name = options["source"], options["set_name"]
    path = SAVED / f"{set_name}.jsonl"
    if not path.is_file():
        raise Final(f"no saved set at {path}")
    exported = section_export.of_source(source)["sections"]
    rows = {section_questions.section_key(r): r for r in exported}
    anchored_in = section_questions.anchors_for(exported)
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    here, gone = [], set()
    for q in saved:
        key = section_questions.gold_key(q.get("gold"))
        if key not in rows:
            gone.add(q["pair_id"])
            continue
        here.append({**q, "text_hash": text_hash(q["original_text"]), "set_name": set_name, "status": CANDIDATE,
                     "anchors": anchored_in(rows[key])(q["original_text"])})
    written, repeats = question_sets.write_pairs(here)
    if written:
        job_queue.enqueue("embed_questions", {})
    return {"source": source, "set_name": set_name, "questions_written": written, "pairs_dropped_as_repeats": repeats,
            "pairs_of_gone_sections": len(gone)}


@register("load_variants")
def load_variants(options: dict) -> dict:
    from use_cases import question_variants

    done = question_variants.load(options["set_name"], options["rows"])
    if done["questions_written"]:
        job_queue.enqueue("embed_questions", {})
    return done
