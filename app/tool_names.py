import re
from enum import StrEnum


# the converter tools the stand has an adapter for; a route, a settings file and an engine name only these
class Tool(StrEnum):
    docling = "docling"
    mineru = "mineru"


# a settings file under converters/<tool>/settings/, named as <tool>/<file>
SETTINGS_NAME = re.compile(r"[a-z0-9_-]+/[a-z0-9_.-]+")


# a settings name that is a file of the named tool, or why not
def settings_refusal(tool: str, name: str) -> str | None:
    if not SETTINGS_NAME.fullmatch(name) or not name.startswith(f"{tool}/"):
        return f"{tool}: {name} is not a settings file of that tool"
    return None
