import config
import corpus_search
import logging_setup
import search_depth
from errors import Final
from models.corpus import DataSource
from orm.sync_db import Session, engine
from sqlalchemy import select
from use_cases import retrieval_compare

log = logging_setup.get_logger(__name__)
HIT_DEPTH = 5


def _hit_share(rows: list[dict]) -> float:
    return round(sum(1 for r in rows if r["file_rank"] and r["file_rank"] <= HIT_DEPTH) / len(rows), 4)


# found in itself is the cut and the questions, found in the corpus is the neighbours: two numbers, two next steps
def run(name: str, set_name: str) -> dict:
    gate = config.settings.intake.quality.source_gate
    variant = config.settings.corpus.variant
    ef = search_depth.resolve(variant)
    with engine.connect() as conn:
        opened = retrieval_compare.measure(corpus_search, conn, set_name, variant, 10**6, exact=False, ef=ef,
                                           source=name)
        clamped = retrieval_compare.measure(corpus_search, conn, set_name, variant, 10**6, exact=False, ef=ef,
                                            source=name, clamped=True)
    n = len(opened)
    said = {"set_name": set_name, "n": n, "clamped_min": gate.clamped_min, "open_min": gate.open_min,
            "min_questions": gate.min_questions}
    if n < gate.min_questions:
        said |= {"verdict": "too_few", "next": f"{n} questions under the floor of {gate.min_questions}; "
                                                "the gate reads nothing, add pairs"}
    else:
        said |= {"clamped": _hit_share(clamped), "open": _hit_share(opened)}
        if said["clamped"] < gate.clamped_min:
            said |= {"verdict": "clamped_low", "next": "the agent mends the cut by a knob or regenerates "
                                                         "the questions once, into a new set"}
        elif said["open"] < gate.open_min:
            said |= {"verdict": "open_low", "next": "waits for the owner: found in itself, not in the corpus"}
        else:
            said |= {"verdict": "passed", "next": "keep it in search"}
    with Session() as session:
        row = session.scalar(select(DataSource).where(DataSource.name == name))
        if row is None:
            raise Final(f"no source named {name}")
        row.raw = {**(row.raw or {}), "gate": said}
        session.commit()
    log.info("source_gate.read", source=name, **{k: said.get(k) for k in ("verdict", "n", "clamped", "open")})
    return said
