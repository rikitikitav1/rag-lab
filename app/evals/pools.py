import outcomes

POOLS = ("in_corpus", "out_of_corpus", "off_domain", "rejected")


def kind(ql) -> str:
    marked = ql.question.marked_sources if ql.question else None
    declared = ql.question.kind if ql.question else None
    if declared in POOLS:
        return declared
    return "in_corpus" if marked else "out_of_corpus"


def outcome(ql) -> str:
    metrics = ql.metrics or {}
    recorded = metrics.get("outcome")
    if recorded in ("narrated_call", "exhausted"):
        return recorded
    config = metrics.get("config") or {}
    exhausted = (
        metrics.get("hops") is not None
        and metrics["hops"] >= config.get("max_hops", 4)
        and not metrics.get("failed")
    )
    if recorded == "error":
        return "exhausted" if exhausted else "error"
    return outcomes.classify(
        ql.answer,
        bool(ql.sources),
        prefixes=[f"{name}__" for name in config.get("mcp_configured") or []],
        exhausted=exhausted,
    )


def has_remote_evidence(ql) -> bool:
    return any(s["source"].startswith("mcp:") for s in (ql.sources or []))


# in_corpus without marked sources cannot be scored against the corpus, so it lands outside
def split(logs) -> dict[str, list]:
    pools: dict[str, list] = {name: [] for name in POOLS}
    for ql in logs:
        name = kind(ql)
        if name == "in_corpus" and not (ql.question and ql.question.marked_sources):
            name = "out_of_corpus"
        pools[name].append(ql)
    return pools
