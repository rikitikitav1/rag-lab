import frontmatter
from sources import base
from sources.base import Base, Parsed


class CheatsheetsSource(Base):
    name = "cheatsheets"
    language = "eng"
    url = "https://github.com/rstacruz/cheatsheets"

    SKIP_FILE_NAMES = {
        "cask-index",
        "cheatsheet-styles",
        "make-assets",
        "AGENTS",
        "README",
        "CONTRIBUTING",
        "index",
        "index@2016",
        "licenses",
        "social-images",
    }

    CATEGORY_TREE = {
        "JavaScript": "languages.javascript",
        "JavaScript libraries": "languages.javascript.libraries",
        "Node.js": "languages.javascript.node",
        "React": "languages.javascript.react",
        "Ruby": "languages.ruby",
        "Ruby libraries": "languages.ruby.libraries",
        "Rails": "languages.ruby.rails",
        "Python": "languages.python",
        "python": "languages.python",
        "Elixir": "languages.elixir",
        "Java & JVM": "languages.java",
        "C-like": "languages.c",
        "CSS": "languages.css",
        "HTML": "markup.html",
        "Markup": "markup",
        "Jekyll": "tools.jekyll",
        "Databases": "databases",
        "Ansible": "devops.ansible",
        "Devops": "devops",
        "CLI": "tools.cli",
        "Vim": "tools.vim",
        "Git": "tools.git",
        "Ledger": "tools.ledger",
        "Analytics": "tools.analytics",
        "Linux": "platforms.linux",
        "macOS": "platforms.macos",
        "apple": "platforms.macos",
        "API": "protocols.api",
        "AI": "llm",
        "Apps": "misc",
        "Bolt": "misc",
        "Development": "misc",
        "Others": "misc",
        "Misc": "misc",
    }

    FILE_TREE = {
        "lua": "languages.lua",
        "less": "languages.css",
        "simple_form": "languages.ruby.libraries",
        "cordova": "languages.javascript",
        "command_line": "tools.cli",
        "google_analytics": "tools.analytics",
        "imagemagick": "tools.imagemagick",
        "plantuml": "tools.plantuml",
        "watchman": "tools.watchman",
        "figlet": "tools.figlet",
        "linux": "platforms.linux",
        "ubuntu": "platforms.linux",
        "firebase": "databases",
        "passenger": "devops",
        "saucelabs": "devops",
    }

    # not our domain by the corpus description, one line of why each
    OFF_DOMAIN_FILES = {
        "vainglory": "items in a mobile MOBA",
        "ph-food-delivery": "delivery phone numbers in Metro Manila",
        "macos-mouse-acceleration": "mouse settings",
        "sketch": "graphics editor",
        "nocode": "a joke repository",
        "flashlight": "spotlight plugins",
        "frequency-separation-retouching": "photo retouching",
        "inkscape": "graphics editor",
    }

    # our domain, but the file is pictures: a general rule for that is still missing
    NOT_TEXT_FILES = {
        "figlet": "samples of every ascii font, one line of knowledge per screen of drawing",
    }

    def files(self):
        return self.root.glob("*.md")

    # a version in the file name means a dead stack, and the unversioned sheet is next to it
    def discover(self, policy=None):
        for file in super().discover(policy):
            if not base.hygienic(policy):
                yield file
                continue
            dropped = self.OFF_DOMAIN_FILES | self.NOT_TEXT_FILES
            if "@" in file.stem or file.stem in dropped:
                continue
            yield file

    def read(self, file, rel, policy=None):
        hygienic = base.hygienic(policy)
        post = frontmatter.loads(
            self.text_of(file) if hygienic else self.legacy_text_of(file)
        )
        if post.metadata.get("category") == "Hidden":
            return None
        raw = post.metadata.get("category")
        category = (
            self.CATEGORY_TREE.get(raw) or self.FILE_TREE.get(file.stem) or "misc"
        )
        title = post.metadata.get("title") or (
            self.title_from(post.content)
            if hygienic
            else self.legacy_title_from(post.content)
        )
        return Parsed(post.content, category, title, [], post.metadata.get("tags", []))
