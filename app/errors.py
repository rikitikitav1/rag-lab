from enum import StrEnum

ERROR_PREFIX = "error: "


# every later row fails the same way; not a RuntimeError, which the loops forgive as a failed call
class StandFault(Exception):
    pass


# the same options fail the same way: a retry only wakes a server and takes the card again
class Final(Exception):
    pass


# why a request was refused; each door says it in its own terms, the API by `server.REFUSAL_STATUS`
class RefusalKind(StrEnum):
    invalid = "invalid"
    malformed = "malformed"
    missing = "missing"
    taken = "taken"
    busy = "busy"
    conflict = "conflict"


class Refusal(Exception):
    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        # a misspelt kind fails where it is raised, not as a 500 at the one door that maps it
        self.kind = RefusalKind(kind)
