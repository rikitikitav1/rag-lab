from dataclasses import dataclass

from timing_wrappers import measure_elapsed


@dataclass
class _Result:
    elapsed: float = 0.0


def test_measure_elapsed_sets_field_on_result():
    @measure_elapsed
    def work():
        return _Result()

    out = work()
    assert isinstance(out.elapsed, float)
    assert out.elapsed >= 0.0


def test_a_prompt_over_the_context_window_is_reported(monkeypatch):
    import llm

    seen = []
    monkeypatch.setattr(llm.log, "warning", lambda event, **kw: seen.append((event, kw)))
    window = llm.config.settings.llm.context_length
    llm._warn_if_truncated(window + 1, "llama3.1:8b")
    llm._warn_if_truncated(window - 1, "llama3.1:8b")

    assert [event for event, _ in seen] == ["llm.prompt_over_context"]
    assert seen[0][1]["context"] == window
