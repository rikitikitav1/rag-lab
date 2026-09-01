from sources import base
from sources.base import Base


class InterviewSource(Base):
    language = "eng"
    kind = "git"

    def __init__(self, root, name=None, url=None, language=None):
        self.root = root
        if name is not None:
            self.name = name
        if url is not None:
            self.url = url
        if language is not None:
            self.language = language

    def files(self):
        return [self.root / "README.md"]

    def category_for(self, rel_path):
        topic = self.name.removesuffix("-interview-questions")
        return f"interview.{topic}"

    # the badge above the first question: identical in all 173 repositories and answers nothing
    BOILERPLATE = "You can also find all"

    def postprocess(self, docs, policy=None):
        docs = super().postprocess(docs, policy)
        if not base.hygienic(policy):
            return docs
        kept = [d for d in docs if self.BOILERPLATE not in d.content]
        for i, doc in enumerate(kept):
            doc.chunk_index = i
        return kept
