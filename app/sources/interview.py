from sources.base import Base


class InterviewSource(Base):
    reader = "interview"

    # the topic is the repository's name, one bank per repository
    def tags_for(self, rel_path):
        return ["interview", self.name.removesuffix("-interview-questions")]
