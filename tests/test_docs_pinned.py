import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_MAP = ROOT / "docs" / "code_map.md"


def _declared(module: str) -> set:
    source = (ROOT / "app" / f"{module}.py").read_text()
    return set(re.findall(r'\.tool\(\s*\n?\s*name="([^"]+)"', source))


def _listed(mount: str) -> set:
    line = next(
        line for line in CODE_MAP.read_text().splitlines()
        if f"(mounted at `{mount}`)" in line
    )
    # the doc names them in backticks separated by slashes, before the prose tail
    return set(re.findall(r"`([a-z_]+)`", line.split(":", 1)[1]))


# the list was maintained by hand and drifted every time a door was added
def test_the_doc_names_every_ops_tool_and_no_tool_that_is_gone():
    assert _listed("/mcp-ops") == _declared("mcp_ops"), (
        "docs/code_map.md and app/mcp_ops.py disagree; run `PYTHONPATH=app uv run python"
        " scripts/surface.py` and copy the tool names"
    )


def test_the_doc_names_every_corpus_tool():
    assert _listed("/mcp") == _declared("mcp_server")
