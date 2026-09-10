from dataclasses import dataclass

NAMED, DANGLING, UNNAMED = "named", "dangling", "unnamed"


@dataclass(frozen=True)
class Stamped:
    name: str | None
    address: str | None
    state: str


# three outcomes: an engine deleted since the run must not read as one that was never recorded
def engine_of(stamp: dict, live: set[str] | None) -> Stamped:
    name = (stamp or {}).get("engine_name")
    address = (stamp or {}).get("engine")
    if not name:
        return Stamped(None, address, UNNAMED)
    # an unreadable table cannot tell a live engine from a deleted one, so it says neither
    if live is None:
        return Stamped(name, address, NAMED)
    return Stamped(name, address, NAMED if name in live else DANGLING)
