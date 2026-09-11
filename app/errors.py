ERROR_PREFIX = "error: "


# every later row fails the same way; not a RuntimeError, which the loops forgive a failed call as
class StandFault(Exception):
    pass
