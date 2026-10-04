import frontmatter
from sources.base import Base, Parsed
from use_cases import markdown_cleanup


class CheatsheetsSource(Base):
    reader = "cheatsheets"

    def read(self, file, rel, policy=None):
        post = frontmatter.loads(self.text_of(file))
        if post.metadata.get("category") == "Hidden":
            return None
        # the sheet's own category is a label, not a row of the map
        raw = post.metadata.get("category")
        title = post.metadata.get("title") or self.title_from(post.content)
        labels = [str(raw)] if raw else []
        tags = labels + list(post.metadata.get("tags", []))
        return Parsed(markdown_cleanup.without_table_padding(post.content), self.category_for(rel), title, [], tags)
