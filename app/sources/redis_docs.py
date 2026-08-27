import frontmatter
import ingest
from sources import base
from sources.base import Base, Parsed


class RedisDocsSource(Base):
    name = "redis-doc"
    kind = "git"
    url = "https://github.com/redis/redis-doc"
    language = "eng"

    def files(self):
        yield from (self.root / "commands").glob("*.md")
        yield from (self.root / "docs").rglob("*.md")

    def category_for(self, rel_path):
        return "databases.redis." + ingest.path_to_category(rel_path)

    # frontmatter was not parsed here before this branch, and the old cut has to keep
    # seeing what it saw: the fence as the first line
    def read(self, file, rel, policy=None):
        if not base.hygienic(policy):
            return super().read(file, rel, policy)
        post = frontmatter.loads(self.text_of(file))
        title = post.metadata.get("title") or self.title_from(post.content)
        return Parsed(post.content, self.category_for(rel), title, [], [])

    # a command page carries no heading at all: the command name lives in the file name
    def section_root_for(self, file, parsed):
        if file.parent.name == "commands":
            return file.stem.replace("-", " ").upper()
        return super().section_root_for(file, parsed)
