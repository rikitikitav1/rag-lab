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
from langchain_core.outputs import Generation, LLMResult
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM

log = logging_setup.get_logger(__name__)

# the role whose model, sampler and seed the guest borrows: it judges, so it borrows the judge
ROLE = "judging"

# response relevancy compares vectors, so one guest borrows this role beside the judging one
EMBEDDING_ROLE = "embedding"


def _text_of(prompt) -> str:
    return prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)


class OurClient(BaseRagasLLM):
    def __init__(self, role: str = ROLE, model: str | None = None):
        from ragas.run_config import RunConfig

        self.role = role
        self.model = model
        # ragas reads this off the object rather than passing it in, and retries through it
        self.set_run_config(RunConfig(max_retries=1, max_wait=1))

    # ragas asks for n samples; our judging sampler is seeded, so n>1 would repeat one answer
    def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None) -> LLMResult:
        if n != 1:
            log.warning("guest_llm.n_capped", asked=n)
        answer = llm.ask("", _text_of(prompt), role=self.role, model=self.model)
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
        from ragas.run_config import RunConfig

        super().__init__()
        self.role = role
        self.set_run_config(RunConfig(max_retries=1, max_wait=1))

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


def stamp() -> dict:
    from importlib.metadata import version

    return {
        "ragas": version("ragas"),
        "model": llm.resolve_name(ROLE),
        "role": ROLE,
        # one guest measures with vectors, and its embedder never reached the record
        "embedding_model": llm.resolve_name(EMBEDDING_ROLE),
        "runtime": _runtime(),
    }
