from errors import Refusal
from models import Job, JobStatus
from models.experiment import Experiment, ExperimentKind
from sqlalchemy import select
from use_cases import experiment, rejudge, retrieval_compare

# each kind writes its own shape, and only its writer knows which key holds what
READING = {
    ExperimentKind.generation: experiment.for_reading,
    ExperimentKind.retrieval: retrieval_compare.for_reading,
    ExperimentKind.rejudge: rejudge.for_reading,
}


# one experiment's report, the same at the MCP door and the REST one; halves and seeds stay with the aggregation
def report_of(session, id: int, pair: str | None = None) -> dict:
    exp = session.get(Experiment, id)
    if exp is None:
        raise Refusal("missing", f"no experiment {id}")
    read = READING[ExperimentKind(exp.kind)](exp.results or {})
    out = {
        "id": exp.id,
        "name": exp.name,
        "kind": exp.kind,
        "status": exp.status,
        "conclusion": exp.conclusion,
        **{k: v for k, v in read.items() if k != "deltas"},
    }
    # the rule the stored numbers were read with, not today's: an older report names none
    if exp.kind != ExperimentKind.retrieval:
        out["outcome_rule"] = (exp.results or {}).get("outcome_rule")
        out["code"] = (exp.results or {}).get("code") or {}
    guests = (
        session.execute(
            select(Job.status).where(
                Job.type == "judge_guest_axes", Job.options["run_name"].astext.in_(exp.run_names or [])
            )
        )
        .scalars()
        .all()
    )
    if guests:
        out["guest_passes"] = {
            "done": sum(1 for status in guests if status == JobStatus.done),
            "of": len(guests),
            "reads": "guest numbers are read per question_id, not compared here; "
            "run_metrics debts.guests says what a copy still owes",
        }
    deltas = read["deltas"]
    if pair is not None:
        if pair not in deltas:
            raise Refusal("missing", f"no pair {pair!r}; this report has {sorted(deltas)}")
        deltas = {pair: deltas[pair]}
    # halves are the aggregation's own check and read as four more numbers per axis here
    out["deltas"] = {name: {k: v for k, v in body.items() if k != "halves"} for name, body in deltas.items()}
    return out
