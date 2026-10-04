"""Block until every named job is finished, woken by the queue's notice instead of polling."""

import argparse
import json
import sys
import time

import job_queue
from models import Job, JobStatus
from orm.sync_db import Session, engine
from sqlalchemy import select


def known(ids: set[int]) -> set[int]:
    with Session() as session:
        return set(session.scalars(select(Job.id).where(Job.id.in_(ids))))


def finished_now(ids: set[int]) -> dict[int, str]:
    with Session() as session:
        rows = session.execute(select(Job.id, Job.status).where(Job.id.in_(ids), Job.status.in_(job_queue.FINISHED)))
        return {id: status.value for id, status in rows}


# a notice wakes it at once; a quiet spell re-reads the rows, so a worker that does not announce is still waited out
def wait(ids: set[int], timeout: float | None, recheck: float = 30) -> dict[int, str]:
    raw = engine.raw_connection()
    try:
        listener = raw.driver_connection
        listener.autocommit = True
        # listening before the first read: a job finishing between the two is still heard
        listener.execute(f"LISTEN {job_queue.FINISHED_CHANNEL}")
        seen: dict[int, str] = {}
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            for id, status in finished_now(ids - set(seen)).items():
                seen[id] = status
                print(json.dumps({"id": id, "status": status}), flush=True)
            if len(seen) == len(ids) or (deadline is not None and time.monotonic() >= deadline):
                return seen
            tick = recheck if deadline is None else max(0.1, min(recheck, deadline - time.monotonic()))
            for notice in listener.notifies(timeout=tick, stop_after=1):
                event = json.loads(notice.payload)
                if event["id"] in ids and event["id"] not in seen:
                    seen[event["id"]] = event["status"]
                    print(json.dumps(event), flush=True)
    finally:
        raw.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", type=int, nargs="*")
    parser.add_argument("--line", action="store_true", help="also wait for every job active when the wait starts")
    parser.add_argument("--timeout", type=float, default=None, help="seconds before giving up")
    args = parser.parse_args()
    # a paused job finishes only when someone resumes it, so the line waits for what can still run
    line = job_queue.live_ids(statuses=(JobStatus.new, JobStatus.running)) if args.line else []
    ids = set(args.ids) | set(line)
    if not ids:
        parser.error("name job ids or pass --line")
    unknown = sorted(ids - known(ids))
    if unknown:
        print(json.dumps({"no_such_job": unknown}), flush=True)
        return 2
    seen = wait(ids, args.timeout)
    missing = sorted(ids - set(seen))
    if missing:
        print(json.dumps({"timed_out": missing}), flush=True)
        return 2
    return 0 if all(s == "done" for s in seen.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
