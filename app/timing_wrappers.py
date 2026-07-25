import time
from functools import wraps


def measure_elapsed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        result.elapsed = round(time.perf_counter() - start_time, 3)
        return result

    return wrapper
