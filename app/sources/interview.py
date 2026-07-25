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
