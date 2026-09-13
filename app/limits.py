from typing import Annotated

from pydantic import AfterValidator

# what a door may accept, once: the same value was capped at four lengths across five doors
MAX_RUN_NAME = 200
# every comparison walks every run named: without a cap one call asks for the table
MAX_RUNS = 32
MAX_QUESTION_IDS = 10000
# k is the LIMIT of the search and the number of chunks joined into the prompt
MAX_K = 100
# how many rows one guest pass may take: its axes cost several model calls per row each
MAX_GUEST_ROWS = 2000

# a family a reader declares by hand: six arms on three axes is 45 tests, and MAX_RUNS is 32
MAX_TESTS = 200


# a repeated id runs its question once and records the list as named
def refuse_repeated_ids(ids: list[int] | None) -> list[int] | None:
    repeated = sorted({i for i in ids or () if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"question ids repeat: {repeated[:20]}")
    return ids


QuestionIds = Annotated[list[int] | None, AfterValidator(refuse_repeated_ids)]
