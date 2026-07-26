import ingest
from sources.base import Base


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
