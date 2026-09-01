import re
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime

import config
import ingest
import logging_setup
import sources.base
from ingest import BOILERPLATE_MIN_FILES, MAX_CHUNK_SIZE
from models.corpus import DataChunk, DataSource, Verdict
from orm.sync_db import Session
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

log = logging_setup.get_logger(__name__)

# named by the model: two declarations of one closed set let the column take any string
VERDICTS = tuple(v.value for v in Verdict)
MODES = ("dry", "indexed")


# one shape for a chunk just cut and one read back: indexed rows have no body or root
@dataclass
class Sample:
    file: str
    content: str
    chunk_index: int
    body: str | None = None
    section: str | None = None
    root: str | None = None
    # only a dry run knows it; indexed rows abstain rather than guess it back from a length
    cut_by: str | None = None


@dataclass
class Metrics:
    chunks: int
    files: int
    section_coverage: float | None
    prefix_dominates: float | None
    dup_in_file: float | None
    dup_in_source: float | None
    tiny: float | None
    # None, not zero: zero would read as clean on all 172 interview repositories
    boilerplate: float | None
    orphans: float | None
    size_cut: float | None
    soup: float | None
    code_only: float | None
    score: int | None = None
    # a ceiling that abstains below a count needs the count, and the shares differ
    denominators: dict[str, int] | None = None


@dataclass
class Report:
    metrics: Metrics
    verdict: str
    breaches: list[str] = field(default_factory=list)


TINY_SHARE_OF_CEILING = 0.1
SOUP_ALNUM_RATIO = 0.55
PROSE_WORD_LETTERS = 4

# every other metric is a defect: more is worse
HIGHER_IS_BETTER = frozenset({"section_coverage"})

FENCE = re.compile(r"```.*?```", re.DOTALL)
PROSE_WORD = re.compile(rf"[^\W\d_]{{{PROSE_WORD_LETTERS},}}", re.UNICODE)
NOT_ALNUM_OR_SPACE = re.compile(r"[^\w\s]", re.UNICODE)
OPENS_WITH_HEADING = re.compile(r"^\s*#")


# zero would pass a gate on a number nobody took, and take its full weight in the score
def _share(hits: int, total: int) -> float | None:
    return hits / total if total else None


def _prefix_of(sample: Sample) -> str | None:
    if sample.body is None or not sample.content.endswith(sample.body):
        return None
    return sample.content[: len(sample.content) - len(sample.body)]


# the gate is about structure under the root, the two-level path both cutters write
def _under_a_heading(sample: Sample) -> bool:
    return bool(sample.section) and " > " in sample.section


def _repeats(groups: dict[object, int]) -> int:
    return sum(n - 1 for n in groups.values() if n > 1)


def _count(values) -> dict:
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


def _is_soup(text: str) -> bool:
    if not text:
        return True
    kept = len(NOT_ALNUM_OR_SPACE.sub("", text))
    return kept / len(text) < SOUP_ALNUM_RATIO


def _is_code_only(text: str) -> bool:
    return not PROSE_WORD.search(FENCE.sub(" ", text))


def _boilerplate_hits(samples: list[Sample], measurable_files: int) -> int:
    wide = ingest.wide_bodies(((s.body, s.file) for s in samples), measurable_files)
    return sum(1 for s in samples if s.body in wide)


def measure(
    samples: list[Sample], ceiling: int = MAX_CHUNK_SIZE, records_sections: bool = True
) -> Metrics:
    total = len(samples)
    files = len({s.file for s in samples})
    tiny_below = ceiling * TINY_SHARE_OF_CEILING

    bodied = [(s, p) for s in samples if (p := _prefix_of(s)) is not None]
    # the prefix repeats on every sibling: on content the chunks look full and prose-like
    measurable = [s for s in samples if s.body is not None]
    decided = [s for s in samples if s.cut_by is not None]
    text = {s.file: [] for s in measurable}
    for s in measurable:
        text[s.file].append(s.body)
    bodies = [s.body for s in measurable]
    n = len(measurable)
    # named once: a floor and a denominator counting different things inverted this metric
    measurable_files = len({s.file for s in measurable})

    return Metrics(
        chunks=total,
        files=files,
        section_coverage=(
            _share(sum(1 for s in samples if _under_a_heading(s)), total)
            if records_sections
            else None
        ),
        prefix_dominates=_share(
            sum(1 for s, p in bodied if len(p) > len(s.body)), len(bodied)
        ),
        dup_in_file=_share(sum(_repeats(_count(t)) for t in text.values()), n),
        dup_in_source=_share(_repeats(_count(bodies)), n),
        tiny=_share(sum(1 for b in bodies if len(b) < tiny_below), n),
        boilerplate=(
            None
            if measurable_files < BOILERPLATE_MIN_FILES or not measurable
            else _share(_boilerplate_hits(measurable, measurable_files), n)
        ),
        orphans=_share(
            sum(1 for s in samples if not OPENS_WITH_HEADING.match(s.content)), total
        ),
        size_cut=_share(
            sum(1 for s in decided if s.cut_by == "size"), len(decided)
        ),
        soup=_share(sum(1 for b in bodies if _is_soup(b)), n),
        code_only=_share(sum(1 for b in bodies if _is_code_only(b)), n),
        denominators={
            "section_coverage": total,
            "orphans": total,
            "prefix_dominates": len(bodied),
            "dup_in_file": n,
            "dup_in_source": n,
            "tiny": n,
            "boilerplate": n,
            "soup": n,
            "code_only": n,
            "size_cut": len(decided),
        },
    )


# a share over a small denominator measures the size of the source, not its cut
MIN_BREACHING_CHUNKS = 5


def _too_few_to_judge(metrics: Metrics, name: str, value: float) -> bool:
    # the share's own denominator: `size_cut` over three decided chunks read as a hundred
    denominator = (metrics.denominators or {}).get(name, metrics.chunks)
    return round(value * denominator) < MIN_BREACHING_CHUNKS


# `getattr(..., None)` gave an unknown name the answer an abstaining metric gives
def _measured(metrics: "Metrics", name: str):
    if name not in {f.name for f in fields(Metrics)}:
        raise ValueError(f"nothing measures {name!r}: gates and weights name what Metrics carries")
    return getattr(metrics, name)


def gate_breaches(metrics: Metrics, gates) -> list[str]:
    declared = gates if isinstance(gates, dict) else gates.model_dump(exclude_none=True)
    breached = []
    for name, bounds in declared.items():
        value = _measured(metrics, name)
        if value is None or not bounds:
            continue
        low, high = bounds.get("min"), bounds.get("max")
        if low is not None and value < low:
            breached.append(f"{name}.min")
        # only the ceiling: a floor is breached by chunks that are missing, which says nothing
        if high is not None and value > high and not _too_few_to_judge(metrics, name, value):
            breached.append(f"{name}.max")
    return breached


def judged_by(metrics: Metrics, gates) -> list[str]:
    declared = gates if isinstance(gates, dict) else gates.model_dump(exclude_none=True)
    return [
        name
        for name, bounds in declared.items()
        if bounds and _measured(metrics, name) is not None
    ]


def verdict(
    hard: list[str], soft: list[str] | None = None, judged: bool = True
) -> str | None:
    # no verdict where no hard gate could be evaluated: "ok" would be the opposite of true
    if not judged:
        return None
    if hard:
        return "broken"
    return "dirty" if soft else "ok"


# a metric that was not measured leaves the score, weight and all
def scored_weights(metrics: Metrics, weights) -> dict[str, float]:
    declared = weights if isinstance(weights, dict) else weights.model_dump()
    return {
        name: weight
        for name, weight in declared.items()
        if weight and _measured(metrics, name) is not None
    }


def score(metrics: Metrics, weights) -> int | None:
    # no declared weights means no score: the verdict comes from the gates
    scored = scored_weights(metrics, weights)
    total = sum(scored.values())
    if not total:
        return None
    earned = sum(
        weight * (getattr(metrics, name) if name in HIGHER_IS_BETTER else 1 - getattr(metrics, name))
        for name, weight in scored.items()
    )
    return round(100 * earned / total)


def collect_dry(source_name: str, *, variant: str) -> list[Sample]:
    import sources.factory

    source = sources.factory.one(source_name)
    policy = config.settings.corpus.policy(variant)
    # the same door the indexer and the digest walk
    return [
        Sample(
            file=doc.source,
            content=doc.content,
            chunk_index=doc.chunk_index,
            body=doc.body,
            section=doc.section,
            root=doc.root,
            cut_by=doc.cut_by,
        )
        for doc in source.documents(policy)
    ]


def collect_indexed(source_name: str, *, variant: str) -> list[Sample]:
    with Session() as session:
        rows = session.scalars(
            select(DataChunk)
            .join(DataSource, DataSource.id == DataChunk.source_id)
            .where(DataSource.name == source_name, DataChunk.variant == variant)
        ).all()
        return [
            Sample(
                file=r.source,
                content=r.content,
                chunk_index=r.chunk_index,
                body=None if r.prefix_len is None else r.content[r.prefix_len :],
                section=r.section,
                # the stored path and prefix descend from one string, so comparing them answers nothing
                root=None,
            )
            for r in rows
        ]


def analyze(source_name: str, *, variant: str, mode: str) -> dict:
    from use_cases.index import check_variant

    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    check_variant(variant)

    cfg = config.settings.ingest_quality
    # refused before the rows are loaded: an undeclared variant is not a shape question
    policy = config.settings.corpus.policy(variant)
    samples = (
        collect_dry(source_name, variant=variant)
        if mode == "dry"
        else collect_indexed(source_name, variant=variant)
    )
    # the legacy cut records a section only where the file opens H1 then H2
    metrics = measure(
        samples,
        ceiling=policy.get("max_chunk_size") or MAX_CHUNK_SIZE,
        records_sections=sources.base.hygienic(policy),
    )
    metrics.score = score(metrics, cfg.weights)
    hard = gate_breaches(metrics, cfg.hard_gates)
    soft = gate_breaches(metrics, cfg.soft_gates)
    judged = judged_by(metrics, cfg.hard_gates)
    entry = {
        "at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "verdict": verdict(hard, soft, judged=bool(judged)),
        "judged_by": judged,
        # a variant that abstains on a metric is scored on another basis, and the two differ
        "score_basis": list(scored_weights(metrics, cfg.weights)),
        "breaches": hard + soft,
        "hard_breaches": hard,
        "soft_breaches": soft,
        "hard_gates": cfg.hard_gates.model_dump(exclude_none=True),
        "soft_gates": cfg.soft_gates.model_dump(exclude_none=True),
        "score_formula": cfg.score_formula,
        # so a dry run is never compared against an indexed one that saw different fields
        "body_available": any(s.body is not None for s in samples),
        "root_available": any(s.root is not None for s in samples),
        "policy": policy,
        # only a dry run cut anything, so only a dry run may name the parser that did it
        "parser": ingest.parser_version() if mode == "dry" else None,
        **asdict(metrics),
    }
    _persist(source_name, variant=variant, entry=entry, mode=mode)
    log.info(
        "ingest_quality.analyzed",
        source=source_name,
        variant=variant,
        mode=mode,
        verdict=entry["verdict"],
        breaches=entry["breaches"],
    )
    return entry


def _persist(source_name: str, *, variant: str, entry: dict, mode: str) -> None:
    keep = config.settings.ingest_quality.history_per_variant
    with Session() as session:
        source = session.scalar(
            select(DataSource).where(DataSource.name == source_name)
        )
        if source is None:
            raise LookupError(f"no such source: {source_name}")
        reports = dict(source.ingest_reports or {})
        # -0 is the identity slice, so "keep nothing" would have kept everything
        history = [*reports.get(variant, []), entry]
        reports[variant] = history[-keep:] if keep else []
        source.ingest_reports = reports
        flag_modified(source, "ingest_reports")
        # only the served variant writes it: a dry run says what the cut would be
        served = variant == config.settings.corpus.variant
        if mode == "indexed" and served and entry["verdict"] is not None:
            source.ingest_quality = entry["verdict"]
            source.ingest_variant = variant
            source.ingest_checked_at = datetime.now(UTC)
        session.commit()
