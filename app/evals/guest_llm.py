"""The standard's metrics, asked through our own client rather than through theirs.

A guest axis that reaches ollama by its own path records no seed, meets no card guard and carries
no width stamp, which would make it the only number on this stand that cannot say what produced it.
"""

import asyncio
import os

import llm

# a synchronous POST to the library's own server sat inside every measured call
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
import logging_setup
from evals.guest_axes import MESSAGE_FORMS
from langchain_core.outputs import Generation, LLMResult
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM

log = logging_setup.get_logger(__name__)

# the guest's own seat: its prompts are not our judge's, and a borrowed seat moved with the judge
ROLE = "ragas"

# the standard's own wrapper sends one human message; an empty system switched the template's default off
MESSAGES = MESSAGE_FORMS[0]

# response relevancy compares vectors on a seat of its own: the corpus embedder stays where the corpus is
EMBEDDING_ROLE = "ragas_embedding"


# ragas reads this off the object rather than taking it in, and both adapters set the same one
def _one_try():
    from ragas.run_config import RunConfig

    return RunConfig(max_retries=1, max_wait=1)


def _text_of(prompt) -> str:
    return prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)



class OurClient(BaseRagasLLM):
    def __init__(self, role: str = ROLE, model: str | None = None, messages: str = MESSAGES):
        self.role = role
        self.model = model
        if messages not in MESSAGE_FORMS:
            raise ValueError(f"unknown guest message form {messages!r}; known: {', '.join(MESSAGE_FORMS)}")
        self.messages = messages
        self.set_run_config(_one_try())

    # ragas asks for n samples; our judging sampler is seeded, so n>1 would repeat one answer
    def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None) -> LLMResult:
        if n != 1:
            log.warning("guest_llm.n_capped", asked=n)
        # the old ruler is kept callable, so a bridge can read both on the same rows
        system = "" if self.messages == MESSAGE_FORMS[1] else None
        answer = llm.ask(system, _text_of(prompt), role=self.role, model=self.model)
        return LLMResult(generations=[[Generation(text=answer.text or "")]])

    async def agenerate_text(
        self, prompt, n=1, temperature=None, stop=None, callbacks=None
    ) -> LLMResult:
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)

    # ragas retries while this says no; our client raises instead of returning a half answer
    def is_finished(self, response: LLMResult) -> bool:
        return True


# response relevancy is the one guest that measures with vectors, so it borrows our embedder too
class OurEmbeddings(BaseRagasEmbeddings):
    def __init__(self, role: str = EMBEDDING_ROLE):
        super().__init__()
        self.role = role
        self.set_run_config(_one_try())

    def embed_query(self, text: str) -> list[float]:
        return llm.embed(text, role=self.role)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return llm.request_embeddings_batch(texts, self.role)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


# `ragas` is in neither image, so a guest number that cannot name its process cannot be placed
def _runtime() -> str:
    from pathlib import Path

    return "container" if Path("/.dockerenv").exists() else "host"


# two guest numbers compare only when every field here agrees; a row stamped `role: judging` sat on the judge
def stamp(messages: str = MESSAGES, model: str | None = None) -> dict:
    from importlib.metadata import version

    from engines import answer_parsers

    # the model the calls went to: a bench on the job, or the seat when the job names none
    picked = llm.resolve_for(ROLE, model)
    return {
        "ragas": version("ragas"),
        "model": picked.name,
        "role": ROLE,
        # a row without it sent an empty system beside the prompt: another ruler for the same axis
        "messages": messages,
        "engine": picked.engine.name,
        "sampler": llm.sampler_of(ROLE, picked),
        "parser": answer_parsers.label(picked.parser),
        "cache_key": llm.cache_key_of(picked.engine),
        # one guest measures with vectors, and its embedder never reached the record
        "embedding_model": llm.resolve_name(EMBEDDING_ROLE),
        "embedding_role": EMBEDDING_ROLE,
        # one embedder on two engines is two rulers, and answer relevancy measures with it
        "embedding_engine": llm.resolve(EMBEDDING_ROLE).engine.name,
        "runtime": _runtime(),
    }
