import base64
import html
import re
import zlib
from pathlib import Path
from urllib.parse import unquote


# the labels of the shapes under one parent of a .drawio.svg, read from the source draw.io keeps inside the picture
def cells(path: Path, parent: str) -> list[str]:
    content = html.unescape(re.search(r'content="([^"]*)"', path.read_text()).group(1))
    model = re.search(r"<diagram[^>]*>(.*?)</diagram>", content, re.S).group(1)
    # draw.io desktop saves the model deflated and base64 encoded unless compression is off
    if not model.lstrip().startswith("<"):
        model = unquote(zlib.decompress(base64.b64decode(model), -15).decode())
    # draw.io writes attributes in its own order, alphabetical once the owner saves, so they are read by name
    tags = (dict(re.findall(r'(\w+)="([^"]*)"', tag)) for tag in re.findall(r"<mxCell [^>]*>", model))
    # a free text is a note the drawing carries, not a shape of it
    shapes = (t for t in tags if t.get("parent") == parent and "vertex" in t)
    return [html.unescape(t.get("value", "")) for t in shapes if not t.get("style", "").startswith("text")]
