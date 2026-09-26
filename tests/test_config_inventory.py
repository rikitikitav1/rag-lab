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


# a role names the prompt versions the seed activates; a version with no file is caught before the base is seeded
def test_every_prompt_version_a_role_names_has_its_file():
    import seed

    assert seed._declared_versions()


# the cpu overlay names no prompts, and a stand without a card must still seat the versions the gpu one does
def test_an_overlay_does_not_move_the_prompt_versions_the_seed_activates(monkeypatch):
    import config
    import seed

    base = seed._declared_versions()
    monkeypatch.setattr(config, "CONFIG_OVERLAY", "config/roles.cpu.yaml")
    assert seed._declared_versions() == base and base["grade_chunk"] == 1
