from __future__ import annotations

from collections.abc import Sequence
from operator import itemgetter
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

_reranker: CrossEncoder | None = None


def _model() -> CrossEncoder:
    global _reranker

    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(config.settings.rerank.model)

    return _reranker


def rerank[R: Sequence](
    question: str,
    rows: Sequence[R],
    top: int,
) -> list[R]:
    if not rows:
        return []

    pairs = [(question, row[0]) for row in rows]
    scores = _model().predict(pairs)
    ranked = sorted(zip(rows, scores, strict=True), key=itemgetter(1), reverse=True)

    return [row for row, _ in ranked[:top]]
