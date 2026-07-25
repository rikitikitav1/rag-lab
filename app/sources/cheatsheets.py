import frontmatter
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

    def files(self):
        return self.root.glob("*.md")

    def read(self, file, rel):
        post = frontmatter.loads(file.read_text(encoding="utf-8", errors="ignore"))
        if post.metadata.get("category") == "Hidden":
            return None
        raw = post.metadata.get("category")
        category = (
            self.CATEGORY_TREE.get(raw) or self.FILE_TREE.get(file.stem) or "misc"
        )
        title = post.metadata.get("title") or self.title_from(post.content)
        return Parsed(post.content, category, title, [], post.metadata.get("tags", []))
