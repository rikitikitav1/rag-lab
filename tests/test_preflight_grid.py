def test_a_key_a_row_never_recorded_does_not_count_as_a_value_they_agree_on(preflight):
    # `code_version` is written by the agent path alone and pinned, so single_shot compared null
    pg = preflight

    assert pg._setting("code_version", {"variant": "baseline"}) is pg._ABSENT
    assert pg._setting("code_version", {"code_version": None}) is None
    assert pg._setting("code_version", {"code_version": "abc1234"}) == "abc1234"


def test_every_pinned_setting_is_a_key_the_snapshot_writes(preflight):
    # a renamed key makes `_pinned` skip it and the check print "identical" having stopped
    from use_cases.run_snapshot import KEYS

    unknown = sorted(set(preflight.PINNED) - set(KEYS))

    assert unknown == [], f"pinned but never written: {unknown}"


def test_the_pinned_topic_is_compared_and_not_silently_skipped(preflight):
    # the names match; the shape under `topic` is unpacked by hand and had no holder
    from use_cases.run_snapshot import of_topic

    written = of_topic(0.35, 0.41, {"ru": 0.35, "en": 0.4})

    assert preflight._setting("topic", {"topic": written}) is not preflight._ABSENT
