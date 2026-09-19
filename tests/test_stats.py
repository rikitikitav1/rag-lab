def test_the_raw_verdict_reads_the_boundary_the_way_the_correction_does():
    # `holm` accepts on `p <= alpha/(m - rank)` and the raw flag was strict
    from evals.stats import annotate_holm

    tests = [{"p": 0.05}]
    annotate_holm(tests, "one test", alpha=0.05)

    assert tests[0]["significant_raw"] is True
    assert tests[0]["significant_holm"] is True


def test_a_test_the_step_down_never_reached_is_given_no_bar_to_have_failed():
    # every test got its positional threshold, so a record could name a bar it never faced
    from evals.stats import annotate_holm

    tests = [{"p": 0.02}, {"p": 0.021}, {"p": 0.022}]
    annotate_holm(tests, "three of a kind")

    assert [t["significant_holm"] for t in tests] == [False, False, False]
    assert tests[0]["holm_threshold"] == round(0.05 / 3, 5), "the one that broke it was compared"
    assert [t["holm_threshold"] for t in tests[1:]] == [None, None]


def test_the_interval_does_not_move_with_the_order_the_deltas_arrived_in():
    # the draw was over the caller's order, so a rejudge reshuffling rows moved the interval
    import random

    from evals import stats

    deltas = [3, -1, 0, 2, -2, 5, 1, 0, -3, 4, 2, -1]
    shuffled = deltas[:]
    random.Random(7).shuffle(shuffled)
    assert stats.delta_stats(deltas)["ci95"] == stats.delta_stats(shuffled)["ci95"]
