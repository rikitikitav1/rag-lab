ERROR_PREFIX = "error: "


# every later row fails the same way; not a RuntimeError, which the loops forgive as a failed call
class StandFault(Exception):
    pass


# the same options fail the same way: a retry only wakes a server and takes the card again
class Final(Exception):
    pass
