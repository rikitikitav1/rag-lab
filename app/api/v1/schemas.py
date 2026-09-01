from pydantic import BaseModel


# one shape whichever pipeline answered: two doors declared six identical fields apart
class AnswerSource(BaseModel):
    link: str
    vector_distance: float | None = None
    vector_rank: float | None = None
    keyword_rank: float | None = None
    score: float
    rerank_score: float | None = None

    @classmethod
    def of(cls, source) -> "AnswerSource":
        return cls(
            link=source.source,
            vector_distance=source.vector_distance,
            vector_rank=source.vector_rank,
            keyword_rank=source.keyword_rank,
            score=source.score,
            rerank_score=source.rerank_score,
        )
