import re

import frontmatter
from sources.base import Base, Parsed

# Hugo shortcodes: `{{< name args >}}` and `{{% name args %}}`, a closing one with a slash before the name
_COMMENT = re.compile(r"\{\{([<%])\s*comment\s*[>%]\}\}.*?\{\{[<%]\s*/comment\s*[>%]\}\}", re.S)
_SHORTCODE = re.compile(r"\{\{[<%]\s*(/?)([\w-]+)((?:[^}]|\}(?!\}))*?)\s*[>%]\}\}")
_QUOTED = re.compile(r'"([^"]*)"')


# a tab's label, a key and an endpoint are words of the page; every other tag is layout and goes
def _shortcode(match: re.Match) -> str:
    closing, name, args = match.group(1), match.group(2), _QUOTED.findall(match.group(3))
    if closing or not args:
        return ""
    if name == "tab":
        return f"\n{args[0]}\n"
    if name == "kbd":
        return args[0]
    if name == "endpoint":
        return f"`{' '.join(args)}`"
    # a Hugo image goes whole, `alt` included: the site's pictures are icons and screenshots, not captions
    return ""


def without_shortcodes(text: str) -> str:
    return _SHORTCODE.sub(_shortcode, _COMMENT.sub("", text))


class ArangoDocsSource(Base):
    reader = "arangodb-docs"

    # a Hugo page carries its title and its lead in the front matter, and no heading of its own
    def read(self, file, rel, policy=None):
        post = frontmatter.loads(self.text_of(file))
        lead = " ".join(str(post.metadata.get("description") or "").split())
        text = (f"{lead}\n\n" if lead else "") + without_shortcodes(post.content)
        title = post.metadata.get("title") or self.title_from(text)
        return Parsed(text, self.category_for(rel), title, [], self.tags_for(rel))
