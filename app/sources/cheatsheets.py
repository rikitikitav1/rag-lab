import frontmatter
from sources import base
from sources.base import Base, Parsed


class CheatsheetsSource(Base):
    reader = "cheatsheets"

    def read(self, file, rel, policy=None):
        hygienic = base.hygienic(policy)
        post = frontmatter.loads(self.text_of(file) if hygienic else self.legacy_text_of(file))
        if post.metadata.get("category") == "Hidden":
            return None
        # the sheet's own category is a label, not a row of the map
        raw = post.metadata.get("category")
        title = post.metadata.get("title") or (
            self.title_from(post.content) if hygienic else self.legacy_title_from(post.content)
        )
        labels = [str(raw)] if raw else []
        return Parsed(post.content, self.category_for(rel), title, [], labels + list(post.metadata.get("tags", [])))
