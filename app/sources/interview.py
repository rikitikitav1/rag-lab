from sources.base import Base


class InterviewSource(Base):
    reader = "interview"

    # the topic is the repository's name, one bank per repository
    def category_for(self, rel_path):
        topic = self.name.removesuffix("-interview-questions")
        return f"interview.{topic}"
