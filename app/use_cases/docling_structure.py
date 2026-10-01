def page_of(item: dict) -> int | None:
    return (item.get("prov") or [{}])[0].get("page_no")


# the body in reading order, furniture (running heads and feet) left out
def reading_order(structure: dict) -> list[tuple[str, dict]]:
    kinds = ("texts", "tables", "pictures", "groups")
    items = {f"#/{kind}/{i}": t for kind in kinds for i, t in enumerate(structure.get(kind, []))}

    def walk(node):
        for child in node.get("children", []):
            ref = child.get("$ref", "")
            item = items.get(ref)
            if item is None or item.get("content_layer") == "furniture":
                continue
            if ref.startswith("#/groups/"):
                yield from walk(item)
            else:
                yield ref, item

    return list(walk(structure.get("body", {})))


# for each code item and each table, whether the next one goes on from it over the page break, nothing between
def continued(structure: dict | None) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {"code": [], "table": []}
    # a lone pipe between two halves is a running foot's rule and is stepped over; a code block reading `|` still counts
    order = [
        (ref, item)
        for ref, item in reading_order(structure or {})
        if item.get("label") in out or item.get("text", "").strip() != "|"
    ]
    for (_, item), (_, after) in zip(order, [*order[1:], ("", {})], strict=True):
        label = item.get("label")
        if label in out:
            page, next_page = page_of(item), page_of(after)
            same_kind = after.get("label") == label
            out[label].append(same_kind and page is not None and next_page == page + 1)
    return out
