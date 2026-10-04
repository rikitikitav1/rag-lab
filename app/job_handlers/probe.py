from collections import Counter

import job_queue
from models.corpus import DataSource
from orm.sync_db import Session
from paths import FETCHED, ROOT
from sqlalchemy import select
from use_cases import converting, intake_fetch, layer_check, reading, route, source_intake
from use_cases.converting import load_settings

from .base import Final, register
from .card import converter_hold


def _file(source: DataSource, wanted: str | None):
    root, gathered, *_ = intake_fetch.gather(source, FETCHED / source.name, ROOT)
    named, _ = intake_fetch.named_files(root, gathered, FETCHED / source.name)
    pdfs = {rel: file for file, rel in named.items() if file.suffix.lower() == ".pdf"}
    if wanted:
        if wanted not in pdfs:
            raise Final(f"{source.name} has no PDF named {wanted}; it has {sorted(pdfs)[:5]}")
        return pdfs[wanted]
    if len(pdfs) != 1:
        raise Final(f"{source.name} has {len(pdfs)} PDFs; name one as `file`: {sorted(pdfs)[:5]}")
    return next(iter(pdfs.values()))


def _read(file, pages: tuple[int, int], block: dict, language: str, layers: list[str]) -> Counter:
    rule, names = source_intake.intake_rule(block)
    loaded = {name: load_settings(name) for name in names.values()}
    parts, routes = [], []
    for run, piece, name in reading.plan(file, names, loaded, rule, pages):
        layer = reading.layer_of(layers, piece)
        done, _, _ = reading.read_piece(file, run.engine, piece, name, loaded, language, layer, rule)
        parts.append(done["markdown"])
        routes.append(run.why)
    whole, _ = reading.whole_file(parts, routes, rule, layers[pages[0] - 1 : pages[1]])
    found: Counter = Counter()
    for _, page in layer_check.check_document(whole, layers[pages[0] - 1 : pages[1]], pages[0]):
        found.update(page)
    return found


# pages of a source's file read with its own knobs and with the probe's over them, each scored against the text layer
@register("probe_intake")
def probe_intake(options: dict) -> dict | None:
    with Session() as session:
        source = session.scalar(select(DataSource).where(DataSource.name == options["source"]))
        if source is None:
            raise Final(f"no source named {options['source']}")
        session.expunge(source)
    file = _file(source, options.get("file"))
    pages = tuple(options["pages"])
    layers = route.layer_texts(file)
    language = intake_fetch.source_language(source, [file], {file: layers})
    own = source_intake.intake_block(source.declaration)
    counts = {}
    for arm, block in (("before", own), ("after", {**own, **options["knobs"]})):
        if options.get("_job_id") is not None and job_queue.is_cancelled(options["_job_id"]):
            return None
        with converting.card_hold(converter_hold()):
            counts[arm] = _read(file, pages, block, language, layers)
    return {
        "file": str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file),
        "pages": list(pages),
        "knobs": options["knobs"],
        "defects": {arm: layer_check.defects(found) for arm, found in counts.items()},
        **{arm: dict(found) for arm, found in counts.items()},
    }
