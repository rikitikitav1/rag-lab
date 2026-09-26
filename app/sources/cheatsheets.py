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
        raw = post.metadata.get("category")
        tree = self.settings.categories
        category = tree.by_front_matter.get(raw) or tree.by_file.get(file.stem) or "misc"
        title = post.metadata.get("title") or (
            self.title_from(post.content) if hygienic else self.legacy_title_from(post.content)
        )
        return Parsed(post.content, category, title, [], post.metadata.get("tags", []))
