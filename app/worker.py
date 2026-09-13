import os
import threading
import time
from datetime import timedelta

import job_handlers
import job_queue
import job_specs
import llm
import logging_setup
from engines import balances
from redaction import redact

log = logging_setup.get_logger(__name__)

POLL_INTERVAL = 3
MAX_ATTEMPTS = 3
# a deferral is not a failure, so it has a ceiling of its own, counted in seconds waited
MAX_DEFERRED_SECONDS = 3600

Deferred = job_handlers.Deferred
Final = job_handlers.Final
HANDLERS = job_handlers.HANDLERS
QUEUES = [q.strip() for q in os.getenv("WORKER_QUEUES", "default,io").split(",") if q.strip()]


def reclaim(queues: list[str]) -> None:
    # single worker by design: anything left running is from a process that died
    stale = job_queue.requeue_stale(queues)
    if stale:
        log.warning("worker.requeued_stale", ids=stale)


def run_once(queues: list[str]) -> bool:
    claimed = job_queue.claim_next(queues)
    if claimed is None:
        return False

    log.info("worker.claimed", id=claimed.id, type=claimed.type)
    handler = HANDLERS.get(claimed.type)
    if handler is None:
        # it called nothing, and an empty count says so instead of reading as a job from before the count
        job_queue.add_tokens(claimed.id, {})
        job_queue.fail(claimed.id, {"error": f"no handler for type {claimed.type}"})
        return True

    # a row written straight into the table never passed the door, so the check runs here too
    try:
        job_specs.check(claimed.type, claimed.options, from_the_worker=True)
    except Exception as bad:
        job_queue.add_tokens(claimed.id, {})
        job_queue.fail(claimed.id, {"error": redact(str(bad))})
        return True

    start = time.perf_counter()
    tally = llm.Tally()
    clouds = _clouds_of(claimed)
    before = _balances(clouds)
    try:
        with llm.accounting(tally), llm.cache_keyed(f"job-{claimed.id}"):
            handler(claimed.options | {"_job_id": claimed.id})
        # written before the status: a reader of a finished job found it done and its count still empty
        _record_spend(claimed.id, tally, clouds, before)
        elapsed = round(time.perf_counter() - start, 3)
        job_queue.complete(claimed.id, elapsed=elapsed)
        log.info("worker.done", id=claimed.id, type=claimed.type, elapsed=elapsed)
    except Deferred as d:
        _record_spend(claimed.id, tally, clouds, before)
        # without a ceiling of its own a job waiting for what never arrives holds its lane
        waited = claimed.options.get("deferred_seconds", 0) + d.delay_seconds
        if waited > MAX_DEFERRED_SECONDS:
            job_queue.fail(
                claimed.id,
                {"error": f"waited {waited}s for what it needs and gave up: {d}",
                 "deferred_seconds": waited},
            )
            log.error(
                "worker.deferred_out",
                id=claimed.id,
                type=claimed.type,
                deferred_seconds=waited,
            )
            _fail_the_experiment_waiting_on(claimed)
            return True
        job_queue.reschedule(
            claimed.id,
            {**claimed.options, "deferred_seconds": waited},
            timedelta(seconds=d.delay_seconds),
        )
        log.info(
            "worker.deferred",
            id=claimed.id,
            type=claimed.type,
            retry_in=d.delay_seconds,
            deferred_seconds=waited,
        )
    except Exception as e:
        # the quota is spent by calls, and a failed attempt spent it as well
        _record_spend(claimed.id, tally, clouds, before)
        elapsed = round(time.perf_counter() - start, 3)
        attempts = claimed.options.get("attempts", 0) + 1
        if attempts < MAX_ATTEMPTS and not isinstance(e, Final):
            job_queue.reschedule(
                claimed.id,
                {**claimed.options, "attempts": attempts},
                timedelta(seconds=60 * attempts),
                elapsed=elapsed,
            )
            log.error("worker.retry", id=claimed.id, attempts=attempts, error=str(e))
        else:
            job_queue.fail(
                claimed.id, {"error": redact(str(e)), "attempts": attempts}, elapsed=elapsed
            )
            log.error("worker.failed", id=claimed.id, error=str(e))
            _fail_the_experiment_waiting_on(claimed)
    return True


# every cloud with a reader for any job that calls a model: one or two requests, and no map of roles to drift
def _clouds_of(claimed) -> list:
    options = claimed.options or {}
    if not (job_specs.LOADS.get(claimed.type) or any(options.get(k) for k in job_specs.MODEL_OVERRIDES.values())):
        return []
    try:
        return balances.readable()
    except Exception as e:
        log.error("worker.balances_unlisted", id=claimed.id, error=str(e))
        return []


# a reader that fails says why in its answer, so a broker's silence never takes the job down
def _balances(clouds: list) -> dict:
    return {spec.name: balances.read(spec, reader) for spec, reader in clouds}


# what the calls took, per role and per broker, on every way out and before the status
def _record_spend(job_id: int, tally, clouds: list, before: dict) -> None:
    _record_tokens(job_id, tally)
    _record_balances(job_id, clouds, before)


def _record_balances(job_id: int, clouds: list, before: dict) -> None:
    if not clouds:
        return
    try:
        job_queue.add_balances(job_id, before, _balances(clouds))
    except Exception as e:
        log.error("worker.balances_not_recorded", id=job_id, error=str(e))


def _record_tokens(job_id: int, tally) -> None:
    try:
        job_queue.add_tokens(job_id, tally.record())
    except Exception as e:
        log.error("worker.tokens_not_recorded", id=job_id, error=str(e))


# judging counts as the experiment's work too: aggregation is reachable only through it
def _fail_the_experiment_waiting_on(claimed) -> None:
    run_name = (claimed.options or {}).get("run_name")
    if claimed.type not in job_queue.EXPERIMENT_JOBS or not run_name:
        return
    try:
        from use_cases import experiment

        experiment.mark_failed_for_run(run_name)
    except Exception as e:
        log.error("worker.experiment_not_failed", run_name=run_name, error=str(e))


def _loop(queues: list[str]) -> None:
    while True:
        try:
            busy = run_once(queues)
        except Exception as e:
            log.error("worker.loop_error", queues=queues, error=str(e))
            busy = False
        if not busy:
            time.sleep(POLL_INTERVAL)


def main() -> None:
    logging_setup.configure(os.getenv("LOG_LEVEL", "INFO"))
    if not QUEUES:
        raise SystemExit("WORKER_QUEUES is empty")
    log.info("worker.start", queues=QUEUES, handlers=list(HANDLERS))
    llm.take_the_card_before_calls(job_handlers.card.take_for_call)
    reclaim(QUEUES)
    for lane in QUEUES[1:]:
        threading.Thread(target=_loop, args=([lane],), daemon=True).start()
    _loop([QUEUES[0]])


if __name__ == "__main__":
    main()
