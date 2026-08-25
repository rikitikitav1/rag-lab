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


def test_a_shorter_prompt_than_the_hop_before_means_the_server_trimmed_it():
    from use_cases.agent import AgentResult

    result = AgentResult()
    for tokens in (600, 1800, 3900, 900, 1500):
        result.note_prompt(tokens)

    assert result.truncated_hops == 1
