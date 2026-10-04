import frontmatter
from sources.base import Base, Parsed
from use_cases import markdown_cleanup


class RedisDocsSource(Base):
    reader = "redis-doc"

    def read(self, file, rel, policy=None):
        post = frontmatter.loads(self.text_of(file))
        title = post.metadata.get("title") or self.title_from(post.content)
        body = markdown_cleanup.without_table_padding(post.content)
        return Parsed(body, self.category_for(rel), title, [], self.tags_for(rel))

    # a command page carries no heading at all: the command name lives in the file name
    def section_root_for(self, file, parsed):
        if file.parent.name == "commands":
            return file.stem.replace("-", " ").upper()
        return super().section_root_for(file, parsed)
