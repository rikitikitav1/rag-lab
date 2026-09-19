# what a measurement puts in git: the number and its reading, not the rows it was computed from

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAVY = 2 * 1024 * 1024
# the twelve master already carries; the count goes down as they are split, never up
GRANDFATHERED = 12


def _tracked_under(folder: str) -> list[Path]:
    said = subprocess.run(["git", "ls-files", folder], cwd=ROOT, capture_output=True, text=True)
    return [ROOT / line for line in said.stdout.splitlines() if line]


def test_no_new_measurement_carries_its_rows_into_git():
    # the candidate pools are exempt: a recorded number names its pool by file name, so it keeps it
    heavy = [p.name for p in _tracked_under("datasets/measurements")
             if p.exists() and p.stat().st_size > HEAVY]
    assert len(heavy) <= GRANDFATHERED, (
        f"{len(heavy)} tracked measurements over 2 MB against a ceiling of {GRANDFATHERED}: {heavy}"
    )


def test_the_rows_of_a_measurement_are_not_tracked():
    rows = [p.name for p in _tracked_under("datasets") if p.name.endswith("_rows.json.gz")]
    assert rows == [], "the rows live beside the file and stay out of git"
