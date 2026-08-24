from __future__ import annotations

from collections.abc import Sequence
from operator import itemgetter
from typing import TYPE_CHECKING

import config
import logging_setup

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

log = logging_setup.get_logger(__name__)

_reranker: CrossEncoder | None = None


def device() -> str:
    if _reranker is not None:
        return str(_reranker.model.device).split(":")[0]
    return requested_device()


def requested_device() -> str:
    import os

    import torch

    if os.getenv("RERANK_DEVICE", "auto") == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _model() -> CrossEncoder:
    global _reranker

    if _reranker is None:
        import torch
        from sentence_transformers import CrossEncoder

        target = requested_device()
        kwargs = {"torch_dtype": torch.float16} if target == "cuda" else {}
        _reranker = CrossEncoder(
            config.settings.rerank.model, device=target, model_kwargs=kwargs
        )

    return _reranker


def _predict(pairs: list) -> list:
    global _reranker
    import torch

    try:
        return _model().predict(pairs)
    except torch.cuda.OutOfMemoryError:
        log.warning("rerank.cuda_oom_fallback", pairs=len(pairs))

    from sentence_transformers import CrossEncoder

    unload()
    _reranker = CrossEncoder(config.settings.rerank.model, device="cpu")
    return _reranker.predict(pairs)


def score_pairs(pairs: list) -> list:
    return _predict(pairs) if pairs else []


def unload() -> None:
    global _reranker
    if _reranker is None:
        return
    _reranker = None

    import gc

    import torch

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        log.info("rerank.unloaded", vram_mb=round(torch.cuda.memory_allocated() / 1e6))


def rerank[R: Sequence](
    question: str,
    rows: Sequence[R],
    top: int,
) -> list[R]:
    if not rows:
        return []

    pairs = [(question, row[0]) for row in rows]
    scores = _predict(pairs)
    ranked = sorted(zip(rows, scores, strict=True), key=itemgetter(1), reverse=True)

    return [row for row, _ in ranked[:top]]
