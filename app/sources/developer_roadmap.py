import re
from pathlib import Path

from sources.base import Base, Parsed


class DeveloperRoadmapSource(Base):
    name = "developer-roadmap"
    language = "eng"
    kind = "git"
    url = "https://github.com/kamranahmedse/developer-roadmap"

    ROADMAP_TREE = {
        # languages
        "c": "languages.c",
        "cpp": "languages.cpp",
        "java": "languages.java",
        "kotlin": "languages.kotlin",
        "golang": "languages.go",
        "rust": "languages.rust",
        "scala": "languages.scala",
        "python": "languages.python",
        "ruby": "languages.ruby",
        "php": "languages.php",
        "javascript": "languages.javascript",
        "typescript": "languages.javascript.typescript",
        "css": "languages.css",
        "swift-ui": "languages.swift",
        # js ecosystem
        "nodejs": "languages.javascript.node",
        "react": "languages.javascript.react",
        "react-native": "languages.javascript.react",
        "vue": "languages.javascript.vue",
        "angular": "languages.javascript.angular",
        "nextjs": "languages.javascript.nextjs",
        # frameworks
        "ruby-on-rails": "languages.ruby.rails",
        "django": "languages.python.django",
        "laravel": "languages.php.laravel",
        "spring-boot": "languages.java.spring",
        "aspnet-core": "languages.csharp",
        "flutter": "languages.dart",
        "wordpress": "languages.php.wordpress",
        # markup
        "html": "markup.html",
        # databases
        "sql": "databases",
        "postgresql-dba": "databases.postgresql",
        "mongodb": "databases.mongodb",
        "redis": "databases.redis",
        "elasticsearch": "databases.elasticsearch",
        # devops
        "docker": "devops.docker",
        "kubernetes": "devops.kubernetes",
        "terraform": "devops.terraform",
        "aws": "devops.aws",
        "cloudflare": "devops.cloudflare",
        "devops": "devops",
        "devops-beginner": "devops",
        "devsecops": "devops.devsecops",
        "mlops": "devops.mlops",
        # tools / platforms / protocols
        "git-github": "tools.git",
        "git-github-beginner": "tools.git",
        "linux": "platforms.linux",
        "android": "platforms.android",
        "ios": "platforms.ios",
        "graphql": "protocols.graphql",
        "api-design": "protocols.api",
        # llm
        "machine-learning": "llm.machine-learning",
        "prompt-engineering": "llm.prompt-engineering",
        "ai-agents": "llm.agents",
    }

    PROFESSIONS = {
        "frontend",
        "frontend-beginner",
        "backend",
        "backend-beginner",
        "full-stack",
        "product-manager",
        "engineering-manager",
        "devrel",
        "technical-writer",
        "qa",
        "ux-design",
        "data-analyst",
        "bi-analyst",
        "data-engineer",
        "network-engineer",
        "software-architect",
        "game-developer",
        "server-side-game-developer",
        "forward-deployed-engineer",
        "ai-engineer",
        "ai-data-scientist",
        "ai-product-builder",
    }

    def files(self):
        return self.root.glob("src/data/roadmaps/*/content/*.md")

    def category_for(self, rel_path):
        slug = Path(rel_path).parts[3]
        if slug in self.ROADMAP_TREE:
            return self.ROADMAP_TREE[slug]
        if slug in self.PROFESSIONS:
            return f"roadmaps.{slug}"
        return "misc"

    def read(self, file, rel):
        content = file.read_text(encoding="utf-8", errors="ignore")
        links = re.findall(r"\]\((https?://[^)\s]+)\)", content)
        body = content.split("Visit the following resources")[0].strip()
        after_title = body.split("\n", 1)[1].strip() if "\n" in body else ""
        if not after_title:  # stub: title only, no body
            return None
        return Parsed(body, self.category_for(rel), self.title_from(content), links, [])
