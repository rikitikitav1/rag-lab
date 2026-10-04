from enum import StrEnum
from typing import Literal

# the words every layer names a job's options by; this module imports nothing of the stand, so anyone may import it

Language = Literal["en", "ru"]
# the rules that pick a query's language: by its stopwords in each full-text config, by its alphabet, by langdetect
QueryLanguageRule = Literal["function_words", "cyrillic_ratio", "langdetect"]
# the markup families a source may declare, each rendered by its own table before the cut
Markup = Literal["hugo", "mdn"]
SOURCE_NAME = r"^[a-z0-9][a-z0-9_-]{0,62}$"

# the axes our own judge scores an answer on
JUDGE_AXES = ("faithfulness", "relevance", "completeness")

# how the guest's prompt reaches the model; the second is the ruler every older guest number was taken with
MESSAGE_FORMS = ("user_only", "empty_system")


# the ceiling a hop budget may name, on every door that takes one
MAX_HOPS = 10

class FallbackPolicy(StrEnum):
    corpus_first = "corpus_first"
    corpus_first_weak = "corpus_first_weak"
    agent_choice = "agent_choice"


class GateSignal(StrEnum):
    cross_encoder = "cross_encoder"
    distance = "distance"
    either = "either"


class Orchestrator(StrEnum):
    # gone, and the values stay queryable: 422 logs carry one arm and 222 the other
    handrolled = "agent"
    langgraph_middleware = "langgraph_middleware"
    langgraph_ported = "langgraph_ported"
    langgraph_idiomatic = "langgraph_idiomatic"
    # measured and refused: it filled a reasoning schema instead of calling a tool
    schema_guided = "schema_guided"


# retired implementations, declared here so a fourth retirement is one line, not three
GONE = frozenset(
    {Orchestrator.handrolled, Orchestrator.langgraph_middleware, Orchestrator.schema_guided}
)


class FallbackReason(StrEnum):
    none = "none"
    empty = "empty"
    weak = "weak"
    off_topic = "off_topic"
    # the grader threw out every chunk: the search found something, so this is not `empty`
    graded_out = "graded_out"


# which edge ended the graph: `final` meant three things, and the reader re-derived one of them
class FinishedBy(StrEnum):
    answer = "answer"
    hops_exhausted = "hops_exhausted"
    # the loop stopped without the model producing text, and the ceiling was not the reason
    no_answer = "no_answer"
    # a bare `create_agent` has no edge of ours; the emptiness is named rather than silent
    unrecorded = "unrecorded"
