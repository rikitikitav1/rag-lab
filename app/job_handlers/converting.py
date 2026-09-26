import json
from pathlib import Path

import config
import engines
import logging_setup
from engines import converter
from engines.converter_tools import PIECE, sha256
from models.registry import EngineKind
from tool_names import Tool

from .base import Final
from .card import restart_holder

log = logging_setup.get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "converters"
# a piece of fifty pages takes minutes; this long without a result is a hang, unless the settings bound it closer
CHUNK_CEILING = 1800


# a settings file by its name, with the hash that stands for it in every record
def load_settings(name: str) -> tuple[dict, str]:
    tool, file = name.split("/")
    path = SETTINGS / tool / "settings" / f"{file}.json"
    if not path.is_file():
        raise Final(f"no settings file {path.relative_to(ROOT)}")
    settings = json.loads(path.read_text())
    if settings["tool"] not in PIECE:
        raise Final(f"no adapter for the converter {settings['tool']}")
    return settings, sha256(path)


def fields(settings: dict, language: str) -> list[tuple[str, str]]:
    found = list(settings["fields"].items())
    found += [("ocr_lang", lang) for lang in settings.get("ocr_lang", {}).get(language, [])]
    return found


def ceiling(settings: dict) -> float:
    return settings.get("piece_ceiling_seconds", CHUNK_CEILING)


# a page range in pieces of a size, first and last inclusive
def pieces(first: int, last: int, size: int) -> list[tuple[int, int]]:
    return [(a, min(a + size - 1, last)) for a in range(first, last + 1, size)]


# the engine a tool is declared on; its supervisor is asked only whether it is up and runs that tool
def converter_for(tool: str):
    name = config.settings.intake.engines.get(Tool(tool)) if tool in Tool else None
    if name is None:
        raise Final(f"no converter engine is declared for {tool} in intake.engines")
    spec = next((s for s in engines.card_engines(EngineKind.converter) if s.name == name), None)
    if spec is None:
        raise Final(f"{name} is declared for {tool} and is not a registered converter engine")
    try:
        runs = converter.reading(spec)[1].get("tool")
    except Exception as e:
        raise Final(f"{name} does not answer: {e}") from e
    if runs != tool:
        raise Final(f"{name} runs {runs}, not {tool}")
    return spec


# a piece through its tool's adapter; a piece that hung restarts the child, so the next one gets a fresh tool
def convert(spec, tool: str, path: Path, fields: list[tuple[str, str]], chunk, ceiling: float) -> dict:
    if tool not in PIECE:
        raise Final(f"no adapter for the converter {tool}")
    result = PIECE[Tool(tool)](spec, path, fields, chunk, ceiling)
    if result["status"] == "timeout":
        restart_holder(spec)
        result["errors"] = [*result["errors"], "child restarted"]
    return result
