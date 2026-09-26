# what of a run goes into git: the question bank and the sources, never the artifacts of a pass

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# what the code reads is an input and lives in git; what a run writes is an artifact and does not
KEPT = ("datasets/questions/", "datasets/sources/", "datasets/panels/", "datasets/converter_gold/")


def _tracked_under(folder: str) -> list[str]:
    said = subprocess.run(["git", "ls-files", folder], cwd=ROOT, capture_output=True, text=True)
    return [line for line in said.stdout.splitlines() if line]


def test_only_the_question_bank_and_the_sources_are_tracked():
    # a measurement is cited by name and read as a table in its journal entry; the file stays out
    stray = [f for f in _tracked_under("datasets") if not f.startswith(KEPT)]
    assert stray == [], f"artifacts of a run in git: {stray[:5]}"
