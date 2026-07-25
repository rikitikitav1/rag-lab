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
