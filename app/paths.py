from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# a declared source's raw markdown and its report, one folder per source, out of git
RAW = ROOT / "datasets" / "raw_sources"
# what an onboarding fetched from outside; a folder placed by hand lives under datasets/inbox instead
FETCHED = RAW / "_fetched"
# a tool's reading of a piece by its file, pages, settings and converter build: a rerun of the rules skips the tool
READINGS = ROOT / "datasets" / "readings"
