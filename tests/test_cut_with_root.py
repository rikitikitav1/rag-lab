import ingest

DOC = """# Ignored, the root is declared

intro text

## First question

first answer

## Second question

second answer
"""


def cuts(content=DOC, root="Redis Streams", ceiling=1000):
    return ingest.cut_with_root(content, root, ceiling=ceiling)


def test_every_piece_carries_the_root():
    assert all(c.prefix.startswith("# Redis Streams\n") for c in cuts())


def test_the_section_is_the_path_in_the_baseline_format():
    assert [c.section for c in cuts()] == [
        "Redis Streams",
        "Redis Streams > First question",
        "Redis Streams > Second question",
    ]


def test_the_body_never_repeats_the_prefix():
    for c in cuts():
        assert not c.body.startswith("#")
        assert c.prefix not in c.body


def test_the_leading_heading_of_the_file_does_not_survive_into_the_body():
    assert cuts()[0].body == "intro text"


def test_a_runaway_heading_is_capped_instead_of_riding_on_every_piece():
    long_root = "R" * 4000
    pieces = ingest.cut_with_root("## S\n" + "b" * 300, long_root, ceiling=100)
    assert all(p.body.strip() for p in pieces)
    assert all(p.prefix == f"# {'R' * ingest.HEADING_CAP}\n## S\n" for p in pieces)


def test_a_heading_is_collapsed_to_one_line():
    pieces = ingest.cut_with_root("## S\n\nbody", "  Redis\t Streams  ")
    assert pieces[0].prefix.startswith("# Redis Streams\n")


def test_the_ceiling_is_spent_on_the_body_unless_the_variant_says_otherwise():
    body = "word " * 60
    on_body = ingest.cut_with_root("## S\n" + body, "Root", ceiling=100)
    assert len(on_body) > 1
    assert all(len(p.body) <= 100 for p in on_body)
    assert any(len(p.prefix) + len(p.body) > 100 for p in on_body)

    on_content = ingest.cut_with_root(
        "## S\n" + body, "Root", ceiling=100, ceiling_on=ingest.CONTENT
    )
    assert all(len(p.prefix) + len(p.body) <= 100 for p in on_content)


# over all 1001 files: 15652 headings, the longest 177, so the cap touches nothing real
def test_a_capped_heading_does_not_shorten_the_recorded_section():
    long_heading = "H" * (ingest.HEADING_CAP - 10)
    pieces = ingest.cut_with_root(f"## {long_heading}\n\nbody", "Root")
    assert pieces[0].section == f"Root > {long_heading}"
    assert f"## {long_heading}\n" in pieces[0].prefix


def test_the_recorded_path_is_bounded_too():
    pieces = ingest.cut_with_root("## " + "H" * 4000 + "\n\nbody", "Root")
    assert len(pieces[0].section) == ingest.SECTION_CAP


def test_a_file_without_a_root_still_cuts():
    pieces = ingest.cut_with_root("## S\n\nbody", None)
    assert [p.section for p in pieces] == ["S"]
    assert pieces[0].prefix == "## S\n"


def test_empty_input_gives_nothing():
    assert ingest.cut_with_root("", "Root") == []
    assert ingest.cut_with_root("   \n ", "Root") == []


def test_a_section_with_no_body_is_dropped():
    pieces = ingest.cut_with_root("## Empty\n\n## Real\n\nbody", "Root")
    assert [p.section for p in pieces] == ["Root > Real"]


LONG = "word " * 300


def structured(content, root="Root", ceiling=200):
    return ingest.cut_structured(content, root, ceiling=ceiling)


def test_a_section_that_fits_is_not_split_by_its_subheadings():
    pieces = structured("## S\n\nshort body")
    assert len(pieces) == 1
    assert pieces[0].section == "Root > S"


def test_a_long_section_is_cut_at_its_subheadings():
    content = f"## S\n\n### One\n\n{LONG}\n\n### Two\n\n{LONG}"
    pieces = structured(content)
    prefixes = {p.prefix for p in pieces}
    assert any("### One" in p for p in prefixes)
    assert any("### Two" in p for p in prefixes)


def test_the_section_stays_at_the_question_level_whatever_the_prefix_carries():
    content = f"## S\n\n### One\n\n{LONG}\n\n### Two\n\n{LONG}"
    assert {p.section for p in structured(content)} == {"Root > S"}


def test_a_sliver_joins_its_neighbour_and_keeps_its_heading():
    content = f"## S\n\n### Big\n\n{LONG}\n\n### Tiny\n\nx"
    pieces = structured(content)
    assert not any(p.body == "x" for p in pieces), "a sliver must not be a chunk of its own"
    assert any("### Tiny" in p.body for p in pieces), "and its heading must survive the merge"


def test_a_sliver_is_kept_apart_when_merging_would_break_the_ceiling():
    body = "y" * 195
    content = f"## S\n\n### Big\n\n{body}\n\n### Tiny\n\nx"
    pieces = structured(content)
    assert any(p.body == "x" for p in pieces)


def test_a_subheading_with_nothing_under_it_rides_along_instead_of_vanishing():
    content = f"## S\n\n### Empty\n### Real\n\n{LONG}"
    pieces = structured(content)
    assert any("### Empty" in p.body for p in pieces)


def test_the_structured_cut_honours_the_ceiling_semantics_too():
    content = f"## S\n\n### One\n\n{LONG}\n\n### Two\n\n{LONG}"
    pieces = ingest.cut_structured(content, "Root", ceiling=200, ceiling_on=ingest.CONTENT)
    assert all(len(p.prefix) + len(p.body) <= 200 for p in pieces)


def test_nothing_merges_across_a_section_boundary():
    content = f"## One\n\n### A\n\n{LONG}\n\n## Two\n\n### B\n\nx"
    pieces = structured(content)
    for p in pieces:
        assert p.body.count("###") == 0 or p.section in ("Root > One", "Root > Two")
    assert {p.section for p in pieces} == {"Root > One", "Root > Two"}


FENCED = """## Real section

before

```bash
### this is a shell comment
## and so is this
```

after
"""


def test_a_heading_inside_a_code_fence_is_not_a_heading():
    pieces = ingest.cut_with_root(FENCED, "Root", ceiling=1000)
    assert [p.section for p in pieces] == ["Root > Real section"]
    assert "### this is a shell comment" in pieces[0].body


def test_the_tilde_fence_counts_too():
    content = "## S\n\n~~~yaml\n### not a heading\n~~~\n\ntail"
    assert [p.section for p in ingest.cut_with_root(content, "Root")] == ["Root > S"]


def test_a_heading_indented_inside_the_line_is_still_a_heading():
    content = "## S\n\nbody\n\n  ## Slightly indented\n\nmore"
    sections = [p.section for p in ingest.cut_with_root(content, "Root")]
    assert sections == ["Root > S", "Root > Slightly indented"]


def test_a_file_whose_fences_do_not_close_is_read_without_them():
    # the missing bracket cannot be put back, and one code block would swallow every heading
    content = "## One\n\n```python\nprint(1)\n\n## Two\n\nbody\n\n## Three\n\nmore"
    sections = [p.section for p in ingest.cut_with_root(content, "Root")]
    assert sections == ["Root > One", "Root > Two", "Root > Three"]


def test_a_tab_inside_a_heading_does_not_lose_the_file():
    # the parser reports headings with non-printables removed, so a raw comparison missed
    content = "## one\ttwo\n\nbody\n\n## plain\n\nmore"
    sections = [p.section for p in ingest.cut_with_root(content, "Root")]
    assert sections == ["Root > one\ttwo", "Root > plain"]


def test_the_file_title_does_not_ride_into_a_structured_intro():
    # an over-ceiling intro emitted the declared root and the file's H1 stacked
    content = "# File Title\n\n" + "intro " * 60 + "\n\n### Sub\n\n" + "body " * 60
    pieces = ingest.cut_structured(content, "Root", ceiling=200)
    assert not any("# File Title" in p.body for p in pieces)
    assert any("intro" in p.body for p in pieces), "and the intro text itself survives"


def test_a_leading_sliver_joins_the_piece_ahead_when_the_ceiling_allows():
    content = "## S\n\n### Tiny\n\nx\n\n### Big\n\n" + "y " * 40
    pieces = structured(content)
    assert not any(p.body.strip() == "x" for p in pieces), "it had nothing behind it"
    joined = next(p for p in pieces if "x" in p.body)
    assert "### Tiny" in joined.body, "its heading rides with its text"
    assert len(joined.prefix) + len(joined.body) <= 200, "and the ceiling still holds"


def test_a_leading_sliver_that_cannot_fit_stands_on_its_own_rather_than_overflow():
    # a short piece is a poor chunk; a chunk over the ceiling is a broken one
    content = "## S\n\n### Tiny\n\nx\n\n### Big\n\n" + "y " * 120
    pieces = structured(content)
    assert any(p.body.strip() == "x" for p in pieces)
    assert all(len(p.prefix) + len(p.body) <= 200 for p in pieces if p.cut_by != "size")


def test_a_piece_with_no_text_joins_the_text_beside_it():
    content = "## S\n\n### Empty\n### Real\n\n" + "z " * 60
    pieces = structured(content)
    assert not any(not ingest._has_text(p.body) for p in pieces)
    assert any("### Empty" in p.body for p in pieces)


def test_a_textless_piece_beside_a_full_one_is_bounded_rather_than_absorbed():
    # the join would break the ceiling, so the heading stands alone
    content = "## S\n\n### Empty\n### Real\n\n" + "z " * 120
    pieces = structured(content)
    assert all(len(p.body) <= 200 for p in pieces), "nothing over the ceiling"
    assert any("### Empty" in p.body for p in pieces), "and nothing written is lost"


def test_a_fenced_heading_stays_fenced_even_when_the_same_text_is_a_real_heading():
    # the parser reports which texts are headings, not which lines, so a fenced copy slipped
    content = "## Setup\n\nreal\n\n```bash\n## Setup\n```\n\ntail"
    pieces = ingest.cut_with_root(content, "Root")
    assert [p.section for p in pieces] == ["Root > Setup"]
    assert any("```bash" in p.body for p in pieces)


def test_an_empty_subheading_between_two_full_pieces_stays_under_the_ceiling():
    # neither neighbour has room, so the heading stands alone rather than break the ceiling
    content = "## S\n\n### Big\n\n" + "y " * 100 + "\n\n### Empty\n### Real\n\n" + "z " * 100
    pieces = structured(content)
    assert all(len(p.body) <= 200 for p in pieces)
    assert any("### Empty" in p.body for p in pieces), "and its heading is not lost"


def test_an_empty_subheading_after_a_full_piece_joins_when_there_is_room():
    content = "## S\n\n### Big\n\n" + "y " * 40 + "\n\n### Empty\n### Real\n\n" + "z " * 20
    pieces = structured(content)
    assert not any(not ingest._has_text(p.body) for p in pieces)


def test_a_run_of_empty_headings_is_bounded_by_the_ceiling_on_both_cutters():
    # textless pieces rode past the budget and sixty headings grew one chunk to 11944 chars
    heads = "".join(f"### Subheading number {i} with an ordinary length\n\n" for i in range(60))
    content = f"# Root\n\n## S\n\n{heads}### Real\n\n" + "word " * 300
    for cut in (ingest.cut_with_root, ingest.cut_structured):
        pieces = cut(content, "Root", ceiling=1024)
        assert all(len(p.body) <= 1024 for p in pieces), cut.__name__
        assert any("Subheading number 0" in p.body for p in pieces), "and none of them is lost"


def test_the_rooted_cutter_folds_a_textless_slice_too():
    # the rule lived on `_merge_slivers`, which `cut_with_root` never calls
    heads = "".join(f"### Heading {i}\n\n" for i in range(6))
    content = f"# Root\n\n## S\n\n{heads}### Real\n\nreal body text under the last one.\n"
    pieces = ingest.cut_with_root(content, "Root", ceiling=1024)
    assert not any(not ingest._has_text(p.body) for p in pieces)
