import os

import config
import yaml
from pydantic import BaseModel, ConfigDict, Field

FORMATS_DIR = "formats"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DocbookFormat(_Strict):
    sections: list[str] = Field(min_length=1)
    titled: list[str] = []


class LatexFormat(_Strict):
    levels: dict[str, int] = Field(min_length=1)


FORMATS = {"docbook": DocbookFormat, "latex": LatexFormat}


# a format's vocabulary, shared by every source of that format
def format_of(name: str):
    with open(os.path.join(config.beside_config(FORMATS_DIR), f"{name}.yaml")) as f:
        return FORMATS[name](**(yaml.safe_load(f) or {}))
