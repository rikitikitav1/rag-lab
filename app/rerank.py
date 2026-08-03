from __future__ import annotations

from collections.abc import Sequence
from operator import itemgetter
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

_reranker: CrossEncoder | None = None


def _device() -> str:
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

        device = _device()
        kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        _reranker = CrossEncoder(
            config.settings.rerank.model, device=device, model_kwargs=kwargs
        )

    return _reranker


def _predict(pairs: list) -> list:
    global _reranker
    import torch

    try:
        return _model().predict(pairs)
    except torch.cuda.OutOfMemoryError:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(config.settings.rerank.model, device="cpu")
        return _reranker.predict(pairs)


def unload() -> None:
    global _reranker
    if _reranker is None:
        return
    _reranker = None
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
