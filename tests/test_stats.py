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
