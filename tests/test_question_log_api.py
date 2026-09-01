def test_a_date_range_given_backwards_is_refused_rather_than_answered_with_nothing(client):
    # only `prompt` checked this; the others returned an empty page, which reads as no rows
    for path in (
        "/v1/question-log?created_from=2026-08-31T00:00:00&created_to=2026-08-01T00:00:00",
        "/v1/job?created_from=2026-08-31T00:00:00&created_to=2026-08-01T00:00:00",
        "/v1/mcp_integration?created_after=2026-08-31T00:00:00&created_before=2026-08-01T00:00:00",
    ):
        out = client.get(path)
        assert out.status_code == 400, path
        assert "must be earlier than" in out.text, path
