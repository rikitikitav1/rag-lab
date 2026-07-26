from __future__ import annotations

from collections.abc import Sequence
from operator import itemgetter
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker

_reranker: FlagReranker | None = None


def _model() -> FlagReranker:
    global _reranker

    if _reranker is None:
        from FlagEmbedding import FlagReranker

        _reranker = FlagReranker(
            config.settings.rerank.model,
            use_fp16=False,
        )

    return _reranker


def rerank[R: Sequence](
    question: str,
    rows: Sequence[R],
    top: int,
) -> list[R]:
    if not rows:
        return []

    pairs = [(question, row[0]) for row in rows]

    scores = _model().compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(zip(rows, scores, strict=True), key=itemgetter(1), reverse=True)

    return [row for row, _ in ranked[:top]]
