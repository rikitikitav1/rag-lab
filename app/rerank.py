from collections.abc import Sequence
from operator import itemgetter

import engines
import llm


def score_pairs(pairs: list) -> list:
    return llm.score_pairs(pairs)


# the placement the reranking engine declares, not a reading of where its weights answered
def device() -> str:
    return "cuda" if llm.resolve("reranking").engine.placement in engines.CARD else "cpu"


def rerank[R: Sequence](
    question: str,
    rows: Sequence[R],
    top: int,
) -> list[tuple[R, float]]:
    if not rows:
        return []

    scores = score_pairs([(question, row[0]) for row in rows])
    ranked = sorted(zip(rows, scores, strict=True), key=itemgetter(1), reverse=True)

    return [(row, float(score)) for row, score in ranked[:top]]
