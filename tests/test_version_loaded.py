import hashlib
from pathlib import Path

import version


def _a_real_module() -> str:
    return str(Path(version.APP) / "version.py")


def _digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fresh(monkeypatch, loaded_files: dict, loaded_paths: set) -> None:
    monkeypatch.setattr(version, "LOADED_FILES", loaded_files)
    monkeypatch.setattr(version, "_loaded_paths", lambda: loaded_paths)
    # the answer is cached for ten seconds, and a test that reuses it reads the previous case
    monkeypatch.setattr(version, "_LOADED_ASKED_AT", 0.0)
    monkeypatch.setattr(version, "_LOADED_ANSWERED", None)


def test_a_loaded_file_that_moved_is_named(monkeypatch):
    path = _a_real_module()
    _fresh(monkeypatch, {path: "what the tree held at start"}, {path})
    said = version.loaded_differs()
    assert said["count"] == 1
    assert said["files"] == ["app/version.py"]


def test_an_edited_file_the_process_never_imported_stays_silent(monkeypatch):
    path = _a_real_module()
    # the tree remembers it as moved, but this process never loaded it, so it runs none of it
    _fresh(monkeypatch, {path: "moved", "app/never_imported.py": "moved too"}, set())
    assert version.loaded_differs() is None


# the case the first design got wrong: edited, then imported, so it agrees with disk and differs from start
def test_a_file_imported_after_the_edit_still_fires(monkeypatch):
    path = _a_real_module()
    _fresh(monkeypatch, {path: "the bytes the tree held at start"}, {path})
    said = version.loaded_differs()
    # the process now runs bytes that match the disk, and the base is the start, so it is still a finding
    assert said and said["count"] == 1


def test_a_file_absent_from_the_start_tree_counts_as_new_code(monkeypatch):
    path = _a_real_module()
    _fresh(monkeypatch, {}, {path})
    assert version.loaded_differs()["count"] == 1


def test_an_untouched_loaded_file_is_silent(monkeypatch):
    path = _a_real_module()
    _fresh(monkeypatch, {path: _digest(path)}, {path})
    assert version.loaded_differs() is None


def test_the_finding_carries_at_most_five_names(monkeypatch):
    paths = {str(p) for p in sorted(Path(version.APP).rglob("*.py"))[:9]}
    assert len(paths) == 9
    _fresh(monkeypatch, dict.fromkeys(paths, "all of them moved"), paths)
    said = version.loaded_differs()
    assert said["count"] == 9
    assert len(said["files"]) == version.LOADED_NAMES


# the stamp is recorded on thousands of rows, so the split into per-file digests must not move it
def test_the_stamp_is_what_it_was_before_the_digests_were_kept_apart():
    digest = hashlib.sha256()
    for path in sorted(version.APP.rglob("*.py")) + version._config_files():
        try:
            name = path.relative_to(version.APP.parent).as_posix()
        except ValueError:
            name = path.as_posix()
        try:
            digest.update(f"{name}:".encode() + hashlib.sha256(path.read_bytes()).digest())
        except OSError:
            digest.update(f"{name}:gone".encode())
    assert version.tree_stamp() == digest.hexdigest()[:12]
