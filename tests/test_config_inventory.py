import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# a key added to the config, a constant or a variable without a verdict fails here, before anyone reads the table
def test_every_rule_of_the_tree_has_a_verdict_and_the_table_is_current():
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "config_inventory.py"), "--check"], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
