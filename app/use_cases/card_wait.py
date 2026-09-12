import config
import engines
import job_queue
import llm
from engines import card
from use_cases import agent_policy, chat

# a judging pass ends in about a minute; a card nobody is judging on comes back in seconds
BUSY_RETRY_SECONDS = 60
FREE_RETRY_SECONDS = 5


# why an answer cannot start now, in the stand's terms; each door picks its own way to say it
class CardBusy(Exception):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


# another engine holds the card: asked again after `retry_after` seconds it may be free
class CardHeld(CardBusy):
    def __init__(self, detail: str, retry_after: int):
        super().__init__(detail)
        self.retry_after = retry_after


# this layout cannot put the answer together, however long one waits
class CannotAnswer(CardBusy):
    pass


# every door asks this one: four copies of the gate rule had already parted over `either`
def reranker_needed(rerank_asked: bool | None = None, agent: bool = False,
                    fallback_policy: str | None = None, gate_signal: str | None = None) -> bool:
    if chat.resolve_rerank(rerank_asked):
        return True
    settings = config.settings.agent
    return agent and agent_policy.gates_with_cross_encoder(
        fallback_policy or settings.fallback_policy, gate_signal or settings.gate_signal
    )


def _reranking(**asked) -> tuple[str, ...]:
    return ("reranking",) if reranker_needed(**asked) else ()


def answering_roles(**asked) -> tuple[str, ...]:
    return ("embedding", "generation", *_reranking(**asked))


def retrieving_roles(**asked) -> tuple[str, ...]:
    return ("embedding", *_reranking(**asked))


# the API never hands the card: it asks the queue and names who holds it, or says why it never will
def wait_for_the_card(*roles) -> None:
    try:
        picked = [llm.resolve(role).engine for role in roles]
    except engines.Unnamed as e:
        # the reranker without its weights on a clean machine: a layout fact, not a server error
        raise CannotAnswer(f"{e}; seat it through PUT /v1/role before asking") from e
    on_card = {spec.id: spec for spec in picked if spec.placement in engines.CARD}
    if len(on_card) > 1:
        names = ", ".join(sorted(spec.name for spec in on_card.values()))
        raise CannotAnswer(
            f"this answer needs two engines of the card ({names}), and the chat does not hand the"
            " card between them: ask it through a run, which does"
        )
    for spec in on_card.values():
        if card.holds_for(spec):
            continue
        if not job_queue.pending_of_type("hand_card", engine_id=spec.id):
            job_queue.enqueue("hand_card", {"engine_id": spec.id, "asked_by": "chat"})
        # running, not waiting: the live batch waits five minutes after every answer and read as busy
        judging = job_queue.running_of_type("judge_answers")
        raise CardHeld(
            f"the card is held by {_holder()}; it is handed to {spec.name} through the queue",
            BUSY_RETRY_SECONDS if judging else FREE_RETRY_SECONDS,
        )


def _holder() -> str:
    held = card.on_card()
    if not held:
        return "nobody yet"
    # with no judge seated the holder is still named, not turned into a 500
    try:
        judge_id = llm.resolve("judging").engine.id
    except engines.Unnamed:
        judge_id = None
    return ", ".join(
        f"the judge on {h.engine.name}" if h.engine.id == judge_id else h.engine.name for h in held
    )
