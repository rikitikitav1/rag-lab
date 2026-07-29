import os
import threading
import time
from datetime import timedelta

import job_handlers
import job_queue
import logging_setup

log = logging_setup.get_logger(__name__)

POLL_INTERVAL = 3
MAX_ATTEMPTS = 3

Deferred = job_handlers.Deferred
HANDLERS = job_handlers.HANDLERS
QUEUES = [q.strip() for q in os.getenv("WORKER_QUEUES", "default,io").split(",") if q.strip()]


def run_once(queues: list[str]) -> bool:
    claimed = job_queue.claim_next(queues)
    if claimed is None:
        return False

    log.info("worker.claimed", id=claimed.id, type=claimed.type)
    handler = HANDLERS.get(claimed.type)
    if handler is None:
        job_queue.fail(claimed.id, {"error": f"no handler for type {claimed.type}"})
        return True

    start = time.perf_counter()
    try:
        handler(claimed.options | {"_job_id": claimed.id})
        elapsed = round(time.perf_counter() - start, 3)
        job_queue.complete(claimed.id, elapsed=elapsed)
        log.info("worker.done", id=claimed.id, type=claimed.type, elapsed=elapsed)
    except Deferred as d:
        job_queue.reschedule(
            claimed.id, claimed.options, timedelta(seconds=d.delay_seconds)
        )
        log.info(
            "worker.deferred",
            id=claimed.id,
            type=claimed.type,
            retry_in=d.delay_seconds,
        )
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 3)
        attempts = claimed.options.get("attempts", 0) + 1
        if attempts < MAX_ATTEMPTS:
            job_queue.reschedule(
                claimed.id,
                {**claimed.options, "attempts": attempts},
                timedelta(seconds=60 * attempts),
                elapsed=elapsed,
            )
            log.error("worker.retry", id=claimed.id, attempts=attempts, error=str(e))
        else:
            job_queue.fail(
                claimed.id, {"error": str(e), "attempts": attempts}, elapsed=elapsed
            )
            log.error("worker.failed", id=claimed.id, error=str(e))
    return True


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
    for lane in QUEUES[1:]:
        threading.Thread(target=_loop, args=([lane],), daemon=True).start()
    _loop([QUEUES[0]])


if __name__ == "__main__":
    main()
