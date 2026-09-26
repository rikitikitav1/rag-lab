import re
from enum import StrEnum


# the converter tools the stand has an adapter for; a route, a settings file and an engine name only these
class Tool(StrEnum):
    docling = "docling"
    mineru = "mineru"


# a settings file under converters/<tool>/settings/, named as <tool>/<file>
SETTINGS_NAME = re.compile(r"[a-z0-9_-]+/[a-z0-9_.-]+")
