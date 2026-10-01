# the weights row and the refusal to start a download that cannot finish
import pytest
from engines import disk


def test_a_download_that_would_fill_the_disk_is_refused_before_it_starts(monkeypatch):
    monkeypatch.setattr(disk, "free_bytes", lambda _p: 6 * 2**30)
    with pytest.raises(disk.NotEnoughDisk, match="keeping 5 GiB in reserve"):
        disk.refuse_if_tight(2 * 2**30, "qwen2.5:7b", "/")


def test_a_download_that_fits_with_the_floor_left_over_goes_ahead(monkeypatch):
    monkeypatch.setattr(disk, "free_bytes", lambda _p: 20 * 2**30)
    disk.refuse_if_tight(5 * 2**30, "qwen2.5:7b", "/")


def test_an_unrecorded_size_promises_nothing_rather_than_implying_room(monkeypatch):
    # the image was estimated at 10 GB and turned out to be 21.5: an unknown size must not read as small
    monkeypatch.setattr(disk, "free_bytes", lambda _p: 1)
    disk.refuse_if_tight(None, "unknown-weights", "/")


def test_the_estimate_looks_at_the_cache_and_not_at_the_root(monkeypatch):
    seen = []
    monkeypatch.setattr(disk, "free_bytes", lambda p: seen.append(p) or 100 * 2**30)
    disk.refuse_if_tight(1, "m")
    assert seen == [disk.cache_path()], "weights land in the cache, which may be another filesystem"


def test_the_weights_row_is_the_join_key_and_the_model_points_at_it():
    from models.registry import Model, Weights

    assert Weights.__table__.c.name.unique, "two spellings of one base model would split the join"
    assert Model.__table__.c.weights_id.nullable, "not recorded stays a state of its own"
    assert Model.__table__.c.size_bytes.nullable
    # the artifact, not the weights: one base model is 4.36 GiB in Q4_K_M and 5.19 in AWQ
    assert "size_bytes" in Model.__table__.c and "size_bytes" not in Weights.__table__.c


def test_a_first_pull_borrows_the_size_the_same_name_recorded_on_another_engine(monkeypatch):
    # size lands only after a pull, so on a first pull the estimate had nothing and never refused
    import engines
    from job_handlers import model_ops
    from models.registry import EngineKind, Placement

    spec = engines.EngineSpec(5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu)

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def scalar(self, _stmt):
            return None

        def scalars(self, _stmt):
            class _R:
                def first(self):
                    return 4_683_087_332

            return _R()

    monkeypatch.setattr(model_ops, "Session", _Session)
    seen = model_ops._size_seen_before(engines.Resolved("qwen2.5:7b", spec))

    assert seen == 4_683_087_332, "the sibling row knew the size and the estimate must use it"
