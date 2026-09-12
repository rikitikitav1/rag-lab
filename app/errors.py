ERROR_PREFIX = "error: "


# every later row fails the same way; not a RuntimeError, which the loops forgive as a failed call
class StandFault(Exception):
    pass
