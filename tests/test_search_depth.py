import config
import pytest
from use_cases import search_depth


# the suite runs without a stack, and most of these need none: they hand `resolve` a
# connection it never touches. The one that exercises the real pool says so
def _stack_is_up() -> bool:
    try:
        with search_depth.db.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _clean_cache():
    search_depth.forget()
    yield
    search_depth.forget()


# a connection the caller already holds, so these never reach for a stack. The two that
# do open one are the production paths, and they are exercised by the stack, not here
CONN = object()


def test_a_number_in_the_config_is_served_without_asking_the_planner(monkeypatch):
    monkeypatch.setattr(config.settings.retrieval, "ef_search", 150)
    assert search_depth.resolve("baseline") == 150, "a pinned depth opens nothing"


def test_an_override_beats_both_the_config_and_the_planner(monkeypatch):
    monkeypatch.setattr(config.settings.retrieval, "ef_search", 150)
    assert search_depth.resolve("baseline", 300) == 300


def test_auto_takes_the_deepest_rung_the_plan_still_walks(monkeypatch):
    monkeypatch.setattr(config.settings.retrieval, "ef_search", "auto")
    monkeypatch.setattr(config.settings.retrieval, "ef_ladder", [100, 200, 400])
    monkeypatch.setattr(search_depth, "_shape", lambda conn: (8959, 42668))
    # the plan walks the index at 100 and 200 and sorts at 400, which is what the table
    # actually did on 28.08 with three variants in it
    monkeypatch.setattr(
        search_depth, "uses_index", lambda conn, variant, ef: ef <= 200
    )
    assert search_depth.resolve("clean_1024", conn=CONN) == 200


def test_the_answer_is_re_asked_when_the_row_estimate_moves(monkeypatch):
    monkeypatch.setattr(config.settings.retrieval, "ef_search", "auto")
    monkeypatch.setattr(config.settings.retrieval, "ef_ladder", [100, 200, 400])
    shape = {"pages": 8959, "rows": 42668}
    monkeypatch.setattr(search_depth, "_shape", lambda conn: (shape["pages"], shape["rows"]))
    monkeypatch.setattr(
        search_depth, "uses_index",
        lambda conn, variant, ef: ef <= (200 if shape["pages"] > 6000 else 100),
    )
    assert search_depth.resolve("clean_1024", conn=CONN) == 200
    # the table was rewritten and the pages went with it, so the crossover comes back
    # down. Rows alone would not have noticed: a DELETE leaves the pages where they were
    shape["pages"] = 5000
    assert search_depth.resolve("clean_1024", conn=CONN) == 100


def test_a_table_no_rung_walks_serves_the_floor_and_says_so(monkeypatch, caplog):
    monkeypatch.setattr(config.settings.retrieval, "ef_search", "auto")
    monkeypatch.setattr(config.settings.retrieval, "ef_ladder", [100, 200, 400])
    monkeypatch.setattr(search_depth, "_shape", lambda conn: (1, 1))
    monkeypatch.setattr(search_depth, "uses_index", lambda conn, variant, ef: False)
    assert search_depth.resolve("clean_1024", conn=CONN) == 100


def test_the_probe_asks_for_the_index_back_before_it_looks(monkeypatch):
    # the measuring path turns index scans off on every pooled connection to force exact
    # search. This asks the same pool, so without turning them back on the probe reads
    # "the planner wants a sort" at every rung, falls to the floor and caches it
    issued = []

    class _Conn:
        def execute(self, statement, params=None):
            issued.append(str(statement))

            class _R:
                def scalars(self_inner):
                    return self_inner

                def all(self_inner):
                    return ["Index Scan using data_chunks_embedding_baseline_idx"]

            return _R()

    assert search_depth.uses_index(_Conn(), "baseline", 200) is True
    assert any("enable_indexscan = on" in q for q in issued), (
        "the probe must undo the exact-search mode for its own statement"
    )


@pytest.mark.skipif(not _stack_is_up(), reason="needs the database this probe asks")
def test_the_probe_is_not_poisoned_by_the_exact_search_mode(monkeypatch):
    from use_cases import retrieval_compare as rc

    monkeypatch.setattr(config.settings.retrieval, "ef_search", "auto")
    monkeypatch.setattr(config.settings.retrieval, "ef_ladder", [100, 200, 400])
    rc.prepare(exact=True)
    try:
        assert search_depth.resolve("baseline") == 200
    finally:
        rc.release()


def test_a_table_no_rung_walks_is_not_remembered(monkeypatch):
    # a poisoned session answers "no rung" too, and a poisoned answer that sticks is
    # worse than a slow one asked again
    monkeypatch.setattr(config.settings.retrieval, "ef_search", "auto")
    monkeypatch.setattr(config.settings.retrieval, "ef_ladder", [100, 200, 400])
    monkeypatch.setattr(search_depth, "_shape", lambda conn: (8959, 42668))
    walks = {"any": False}
    monkeypatch.setattr(search_depth, "uses_index", lambda conn, v, ef: walks["any"])
    assert search_depth.resolve("clean_1024", conn=CONN) == 100
    walks["any"] = True
    assert search_depth.resolve("clean_1024", conn=CONN) == 400, "the floor must not have been cached"


def test_an_exact_search_resolves_no_depth(monkeypatch):
    # the switch has to switch something: an arm that turned index scans off must not
    # pay a probe to learn how deep to walk a graph it does not walk
    import db

    monkeypatch.setattr(config.settings.retrieval, "ef_search", "auto")
    monkeypatch.setattr(
        search_depth, "resolve",
        lambda *a, **kw: pytest.fail("an exact arm asked for a depth"),
    )
    issued = []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, statement, params=None):
            issued.append(str(statement))

            class _R:
                def fetchall(self_inner):
                    return []

            return _R()

    monkeypatch.setattr(db.engine, "connect", lambda: _Conn())
    db.hybrid_search("q", "[0]", None, variant="baseline", exact=True)
    assert not any("hnsw.ef_search" in q for q in issued)


def test_an_exact_search_turns_the_index_off_on_its_own_connection(monkeypatch):
    # exactness used to come entirely from a listener another function installs on the
    # shared pool, so a caller that said `exact=True` without it walked the graph at
    # pgvector's default of 40 and every label downstream said "exact"
    import db

    issued = []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, statement, params=None):
            issued.append(str(statement))

            class _R:
                def fetchall(self_inner):
                    return []

            return _R()

    monkeypatch.setattr(db.engine, "connect", lambda: _Conn())
    db.hybrid_search("q", "[0]", None, variant="baseline", exact=True)
    assert any("enable_indexscan = off" in q for q in issued)
    assert not any("hnsw.ef_search" in q for q in issued)
