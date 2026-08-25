import re
import sys
import textwrap
from pathlib import Path

WIDTH = 46
LABEL = re.compile(r'"((?:[^"\\]|\\.)*)"')


# a single colon is prose ("Why phases: ..."), a colon on several lines is a key/value list
def structural(lines: list[str]) -> bool:
    if any(line.strip().startswith(("-", "*")) for line in lines):
        return True
    return sum(": " in line for line in lines) > 1


def rewrap(label: str) -> str:
    if "|md" in label:
        return label
    lines = [line.strip() for line in label.split("\\n") if line.strip()]
    if not lines:
        return label
    wrap = dict(width=WIDTH, break_long_words=False, break_on_hyphens=False)
    if structural(lines):
        # key: value lines carry their own meaning, only the long ones get folded
        out = [folded for line in lines for folded in textwrap.wrap(line, **wrap)]
    else:
        out = textwrap.wrap(" ".join(lines), **wrap)
    return "\\n".join(out)


def process(path: Path) -> bool:
    text = path.read_text()
    out = LABEL.sub(lambda m: f'"{rewrap(m.group(1))}"', text)
    if out == text:
        return False
    path.write_text(out)
    return True


if __name__ == "__main__":
    targets = sys.argv[1:] or [str(p) for p in Path("docs/diagrams").glob("*.d2")]
    changed = [p for p in targets if process(Path(p))]
    print(f"rewrapped {len(changed)} of {len(targets)} diagrams at {WIDTH} chars")
