import engines
import job_queue
import llm
from engines import card

# a judging pass ends in about a minute; a card nobody is judging on comes back in seconds
BUSY_RETRY_SECONDS = 60
FREE_RETRY_SECONDS = 5


class CardBusy(Exception):
    def __init__(self, status: int, detail: str, retry_after: int | None = None):
        super().__init__(detail)
        self.status, self.detail, self.retry_after = status, detail, retry_after


# the API never hands the card: it asks the queue and names who holds it, or says why it never will
def wait_for_the_card(*roles) -> None:
    picked = [llm.resolve(role).engine for role in roles]
    on_card = {spec.id: spec for spec in picked if spec.placement in engines.CARD}
    if len(on_card) > 1:
        names = ", ".join(sorted(spec.name for spec in on_card.values()))
        raise CardBusy(409, (
            f"this answer needs two engines of the card ({names}), and the chat does not hand the"
            " card between them: ask it through a run, which does"
        ))
    for spec in on_card.values():
        if card.holds_for(spec):
            continue
        if not job_queue.pending_of_type("hand_card", engine_id=spec.id):
            job_queue.enqueue("hand_card", {"engine_id": spec.id, "asked_by": "chat"})
        judging = job_queue.pending_of_type("judge_answers")
        raise CardBusy(
            503,
            f"the card is held by {_holder()}; it is handed to {spec.name} through the queue",
            BUSY_RETRY_SECONDS if judging else FREE_RETRY_SECONDS,
        )


def _holder() -> str:
    judge = llm.resolve("judging").engine
    held = card.on_card()
    if not held:
        return "nobody yet"
    return ", ".join(
        f"the judge on {h.engine.name}" if h.engine.id == judge.id else h.engine.name for h in held
    )
