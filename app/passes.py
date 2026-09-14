import json
import threading
from dataclasses import dataclass

import job_queue
import llm
import logging_setup
from engines import card
from errors import Final, StandFault

log = logging_setup.get_logger(__name__)


# a role reseated mid-pass answers its next row with another model under the pass's stamp
class SeatChanged(StandFault, Final):
    pass


# a pass that owed rows and did none is a broken instrument, as a run that answered nothing is
class NothingDone(StandFault, Final):
    pass


@dataclass(frozen=True)
class Seat:
    role: str
    model: str | None = None
    # the row a spill is read from: 0 when the pass took the card for it, 1 when its first call loads it
    spill_from: int | None = 0


# a failed reading is unknown, not a move: a question about the seat must not take the row down
def _resolve(seat: Seat):
    try:
        return llm.resolve_for(seat.role, seat.model)
    except StandFault:
        raise
    except Exception as e:
        log.warning("pass.seat_unread", role=str(seat.role), error=str(e))
        return None


# the model's own sampler is part of the seat: a budget changed mid-pass answers its next row another way
def _pair(picked) -> tuple[str, str, str] | None:
    if picked is None:
        return None
    return picked.engine.name, picked.name, json.dumps(getattr(picked, "options", None) or {}, sort_keys=True)


def _named(seat: tuple[str, str, str]) -> str:
    engine, model, options = seat
    return f"{engine}/{model}" + ("" if options == "{}" else f" {options}")


def _on_card(picked) -> bool | None:
    try:
        return card.model_on_card(picked.engine, picked.name)
    except StandFault:
        raise
    except Exception as e:
        log.warning("pass.card_unread", model=picked.name, error=str(e))
        return None


# one reading per seat, kept: the row stamps the same reading the pass stopped or went on by
def _read_spills(checked, allow_spill: bool, known: dict | None = None) -> dict:
    seen, off = {}, []
    for seat, picked in checked:
        if picked is None:
            continue
        on = (known or {}).get(seat.role)
        seen[seat.role] = on = _on_card(picked) if on is None else on
        if card.spilled_reading(picked.engine, on):
            off.append(f"{seat.role}={picked.name}")
    if off and not allow_spill:
        raise card.CardNotHanded(
            f"not whole on the card: {', '.join(off)}; the pass stops rather than measure the cpu,"
            " pass allow_cpu if that is what it is for"
        )
    if off:
        log.warning("pass.cpu_allowed", models=off)
    return seen


# half on the processor answers with other kernels, so a measuring pass stops unless the cpu is meant
def refuse_spill(seats, allow_spill: bool) -> None:
    _read_spills([(seat, _resolve(seat)) for seat in seats], allow_spill)


# the guards of a loop over rows with a model, in one place: each loop kept its own copy and missed some
class Pass:
    def __init__(self, job_id: int | None, seats: tuple[Seat, ...], *, allow_spill: bool = False, residency=None):
        self.job_id = job_id
        self.seats = seats
        self.allow_spill = allow_spill
        self._read_residency = residency
        self._seen = None
        self._rows = 0
        self._lock = threading.Lock()
        self._local = threading.local()
        self.ended_on_card = None
        self.seated = {seat: _pair(_resolve(seat)) for seat in seats}

    def cancelled(self) -> bool:
        return self.job_id is not None and job_queue.is_cancelled(self.job_id)

    # before a row's first call: False once cancelled; a moved seat or a spilled model stops the pass
    def before_row(self) -> bool:
        if self.cancelled():
            return False
        now = {seat: _resolve(seat) for seat in self.seats}
        for seat, was in self.seated.items():
            got = _pair(now[seat])
            if None not in (was, got) and got != was:
                raise SeatChanged(
                    f"{seat.role} moved from {_named(was)} to {_named(got)} during the pass; it stops here"
                )
        with self._lock:
            row, self._rows = self._rows, self._rows + 1
        checked = [(s, now[s]) for s in self.seats if s.spill_from is not None and row >= s.spill_from]
        self._local.on_card = _read_spills(checked, self.allow_spill)
        return True

    # read by the row's own calls: before the next row, another role's engine may hold the card and hide this one
    def after_row(self, placed: dict) -> None:
        checked = [(seat, _resolve(seat)) for seat in self.seats if placed.get(seat.role) is not None]
        # one reading per seat per row: `on_card` answers with the call's reading from here on
        self._local.on_card = getattr(self._local, "on_card", {}) | _read_spills(checked, self.allow_spill, known=placed)

    # the card as this thread's last row read it, for the row to stamp
    def on_card(self, role) -> bool | None:
        return getattr(self._local, "on_card", {}).get(role)

    # read once, when first asked: a judge knows its residency only after its first call took the card
    def get(self):
        if self._seen is None and self._read_residency is not None:
            self._seen = self._read_residency()
        return self._seen

    def close(self, *, owed: int, done: int, nothing: type[Exception] = NothingDone) -> None:
        first = next((s for s in self.seats if s.spill_from == 0), None)
        picked = _resolve(first) if first else None
        self.ended_on_card = _on_card(picked) if picked else None
        started = getattr(self._seen, "on_card", None)
        # the rows carry their own reading; this says the pass is not one residency any more
        if None not in (started, self.ended_on_card) and started != self.ended_on_card:
            log.error("pass.residency_moved", job_id=self.job_id, started_on_card=started,
                      ended_on_card=self.ended_on_card)
        if owed and not done and not self.cancelled():
            raise nothing(f"0 of {owed} owed rows done; each row's error is in the worker log")
