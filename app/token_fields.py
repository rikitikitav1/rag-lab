# the counts a call adds to its job, in the order the tally keeps them; one list for the writer and the readers
FIELDS = ("prompt", "completion", "calls", "uncounted", "max_prompt", "cut_by_length", "input_cut")
MAXED = ("max_prompt",)
SUMMED = tuple(name for name in FIELDS if name not in MAXED)
# written only when they happened: a zero there is the ordinary case, not a count to carry
OPTIONAL = ("uncounted", "cut_by_length", "input_cut")
JUDGE_PROMPT = "judge_prompt_tokens"
JUDGE_COMPLETION = "judge_completion_tokens"
JUDGE_CUT = "judge_cut_by_length"


# the output limit stopped the answer, the one reason every engine spells the same way
def cut(finish_reason: str | None) -> bool:
    return finish_reason == "length"
