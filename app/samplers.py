KEYS = ("temperature", "max_tokens", "seed", "repetition_penalty")
MAX_TOKENS = 65536


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# one rule for the role's options in the config and the model's at its door: only one of them was checked
def check(options: dict) -> dict:
    unknown = sorted(set(options) - set(KEYS))
    if unknown:
        raise ValueError(f"options takes sampler keys only, got {unknown}; known: {sorted(KEYS)}")
    budget = options.get("max_tokens")
    if budget is not None and (not _whole(budget) or not 1 <= budget <= MAX_TOKENS):
        raise ValueError(f"max_tokens is a whole number from 1 to {MAX_TOKENS}, got {budget!r}")
    heat = options.get("temperature")
    if heat is not None and (isinstance(heat, bool) or not isinstance(heat, int | float) or not 0 <= heat <= 2):
        raise ValueError(f"temperature is a number from 0 to 2, got {heat!r}")
    seed = options.get("seed")
    if seed is not None and (not _whole(seed) or seed < 0):
        raise ValueError(f"seed is a whole number from 0, got {seed!r}")
    penalty = options.get("repetition_penalty")
    if penalty is not None and (isinstance(penalty, bool) or not isinstance(penalty, int | float) or not 1 <= penalty <= 2):
        raise ValueError(f"repetition_penalty is a number from 1 to 2, got {penalty!r}")
    return options
