import pytest
from use_cases.ingest_quality import (
    Sample,
    gate_breaches,
    judged_by,
    measure,
    score,
    verdict,
)

CEILING = 100

# read from the config the production code reads, not copied: four of five hand-written
# gates had drifted from `config.yaml`, and a test that asserts against a dict it wrote
# itself cannot be falsified by changing the thing it is about
def _gates(kind: str) -> dict:
    import config

    cfg = getattr(config.settings.ingest_quality, kind)
    return {
        name: {k: v for k, v in vars(gate).items() if v is not None}
        for name, gate in vars(cfg).items()
        if gate is not None
    }


HARD = _gates("hard_gates")
SOFT = _gates("soft_gates")


BODY = "body text long enough to outweigh its own heading"


def _body_of(content):
    # what the cut would have left: the text under the heading lines it glued on top
    lines = content.split("\n")
    while lines and lines[0].lstrip().startswith("#"):
        lines.pop(0)
    return "\n".join(lines)


def chunk(
    file="src/a.md",
    content="# T\n## S\n" + BODY,
    body=None,
    section="T > S",
    root="T",
    i=0,
):
    return Sample(
        file=file,
        content=content,
        chunk_index=i,
        body=_body_of(content) if body is None else body,
        section=section,
        root=root,
    )


def test_counts_chunks_and_files():
    m = measure([chunk(file="src/a.md"), chunk(file="src/a.md"), chunk(file="src/b.md")], CEILING)
    assert m.chunks == 3
    assert m.files == 2


def test_section_coverage_is_the_share_with_a_heading_path():
    m = measure([chunk(section="T > S"), chunk(section=None)], CEILING)
    assert m.section_coverage == 0.5


def test_a_cut_that_cannot_record_sections_abstains_instead_of_reading_zero():
    # the legacy cut only records a section when the file opens with an H1 then an H2, so
    # a source carrying its title in frontmatter would read 0.0 forever
    m = measure([chunk(section=None), chunk(section=None)], CEILING, records_sections=False)
    assert m.section_coverage is None


def test_prefix_dominates_when_the_prefix_outweighs_the_body():
    m = measure(
        [
            chunk(content="x" * 90 + "\nshort", body="short"),
            chunk(content="# T\n" + "b" * 90, body="b" * 90),
        ],
        CEILING,
    )
    assert m.prefix_dominates == 0.5


def test_duplicates_inside_one_file_and_across_the_source_are_different_numbers():
    same = "# T\n## S\nsame body text that is long enough"
    samples = [
        chunk(file="src/a.md", content=same, i=0),
        chunk(file="src/a.md", content=same, i=1),
        chunk(file="src/b.md", content=same, i=0),
    ]
    m = measure(samples, CEILING)
    # one row of three is a repeat within its own file
    assert m.dup_in_file == pytest.approx(1 / 3)
    # two of three are repeats once the whole source is looked at together
    assert m.dup_in_source == pytest.approx(2 / 3)


def test_tiny_counts_chunks_too_short_to_answer_anything():
    m = measure([chunk(content="ok"), chunk(content="a proper chunk of text here")], CEILING)
    assert m.tiny == 0.5


def test_size_cut_is_what_the_cutter_said_not_what_a_length_suggests():
    # a body just under the ceiling looks the same whether structure or the counter put
    # it there; only the cut knows, so the cut is asked
    by_size = Sample(
        file="src/a.md", content="# T\n## S\n" + "x" * 90, chunk_index=0,
        body="x" * 90, section="T > S", root="T", cut_by="size",
    )
    by_structure = Sample(
        file="src/a.md", content="# T\n## S\n" + "x" * 90, chunk_index=1,
        body="x" * 90, section="T > S", root="T", cut_by="subsection",
    )
    m = measure([by_size, by_structure], CEILING)
    assert m.size_cut == 0.5


def test_size_cut_abstains_where_the_cut_is_not_recorded():
    m = measure([chunk()], CEILING)
    assert m.size_cut is None


def test_orphans_are_chunks_that_do_not_open_with_a_heading():
    m = measure([chunk(content="# T\n## S\nbody"), chunk(content="just body")], CEILING)
    assert m.orphans == 0.5


def test_soup_is_descriptive_only_and_never_a_gate():
    m = measure([chunk(content="#$%^&*()_+{}|:<>?~" * 5), chunk(content="plain words here")], CEILING)
    assert m.soup == 0.5
    assert not any("soup" in b for b in gate_breaches(m, HARD) + gate_breaches(m, SOFT))


def test_code_only_counts_chunks_without_prose():
    m = measure(
        [chunk(content="```\nx = 1\n```"), chunk(content="a sentence about the thing")],
        CEILING,
    )
    assert m.code_only == 0.5


def test_boilerplate_is_the_same_block_standing_in_most_files_of_a_source():
    banner = "# T\n## S\nsubscribe to the channel and smash the like button"
    samples = [
        chunk(file="src/a.md", content=banner),
        chunk(file="src/b.md", content=banner),
        chunk(file="src/c.md", content=banner),
        chunk(file="src/d.md", content="# T\n## S\n" + BODY),
    ]
    m = measure(samples, CEILING)
    assert m.boilerplate == 0.75


def test_a_source_without_any_section_breaches_a_hard_gate_and_is_broken():
    m = measure([chunk(section=None), chunk(section=None)], CEILING)
    hard = gate_breaches(m, HARD)
    assert "section_coverage.min" in hard
    assert verdict(hard, gate_breaches(m, SOFT)) == "broken"


def test_a_soft_breach_alone_is_dirty_not_broken():
    same = "# T\n## S\n" + BODY
    m = measure([chunk(content=same, i=0), chunk(content=same, i=1)], CEILING)
    assert gate_breaches(m, HARD) == []
    assert "dup_in_file.max" in gate_breaches(m, SOFT)
    assert verdict([], gate_breaches(m, SOFT)) == "dirty"


def test_boilerplate_is_not_measurable_under_three_files():
    m = measure([chunk(file="src/a.md"), chunk(file="src/b.md", i=1)], CEILING)
    assert m.boilerplate is None
    assert gate_breaches(m, {"boilerplate": {"max": 0.0}}) == []


def test_a_clean_source_breaches_nothing():
    other = "# T\n## S2\n" + BODY.replace("body", "other")
    m = measure([chunk(), chunk(i=1, content=other, body=BODY.replace("body", "other"), section="T > S2")], CEILING)
    assert gate_breaches(m, HARD) == []
    assert gate_breaches(m, SOFT) == []
    assert verdict([], []) == "ok"


def test_verdict_is_one_of_three():
    m = measure([chunk()], CEILING)
    assert verdict(gate_breaches(m, HARD), gate_breaches(m, SOFT)) in (
        "ok", "dirty", "broken",
    )


def test_without_weights_there_is_no_score_but_the_verdict_still_works():
    m = measure([chunk(section=None)], CEILING)
    assert score(m, {}) is None
    assert verdict(gate_breaches(m, HARD), gate_breaches(m, SOFT)) == "broken"


def test_score_is_an_integer_on_a_hundred_point_scale():
    m = measure([chunk()], CEILING)
    s = score(m, {"section_coverage": 1.0})
    # perfect coverage under the only metric where more is better: emptying
    # HIGHER_IS_BETTER turns this into 0, and `0 <= s <= 100` would not notice
    assert s == 100


def test_a_metric_with_nothing_to_measure_is_none_not_zero():
    # indexed rows of a frozen variant carry no root and no body: the gates behind those
    # metrics must abstain rather than pass on a number nobody took
    bare = [
        Sample(file="src/a.md", content="# T\n## S\nbody", chunk_index=0, section="T > S")
    ]
    m = measure(bare, CEILING)
    assert m.prefix_dominates is None
    assert m.dup_in_source is None
    assert gate_breaches(m, HARD) == []
    assert score(m, {"prefix_dominates": 25}) is None


def test_an_unmeasured_metric_neither_scores_nor_penalises():
    m = measure([chunk()], CEILING)
    m.dup_in_source = None
    # section_coverage is 1.0 and it is the only metric left with a weight
    assert score(m, {"section_coverage": 40, "dup_in_source": 25}) == 100


def test_defects_are_measured_on_the_body_so_the_prefix_cannot_hide_them():
    prefix = "# T\n## S\n"
    same = prefix + "identical answer text"
    samples = [
        Sample(file="src/a.md", content=same, chunk_index=0, body="identical answer text",
               section="T > S", root="T"),
        Sample(file="src/b.md", content=same, chunk_index=0, body="identical answer text",
               section="T > S", root="T"),
    ]
    m = measure(samples, CEILING)
    assert m.dup_in_source == 0.5


def test_nothing_to_measure_gives_no_verdict_rather_than_broken():
    # an unknown variant selects no rows. "the cut is broken" and "there was no cut here"
    # must not come out as the same word
    m = measure([], CEILING)
    assert m.section_coverage is None
    assert judged_by(m, HARD) == []
    assert gate_breaches(m, HARD) == []
    assert verdict([], [], judged=False) is None


def test_boilerplate_counts_one_population_on_both_sides():
    # the floor used to count every file while the ratio counted only those with a body,
    # so a source with two measurable files out of four read 1.0 against a gate of 0.1
    same = "the same block in both"
    samples = [
        Sample(file="a.md", content=same, chunk_index=0, body=same, section="T > S"),
        Sample(file="b.md", content=same, chunk_index=0, body=same, section="T > S"),
        Sample(file="c.md", content="x", chunk_index=0, section="T > S"),
        Sample(file="d.md", content="y", chunk_index=0, section="T > S"),
    ]
    assert measure(samples, CEILING).boilerplate is None, (
        "two files with a body is under the floor, so the metric abstains"
    )


def test_boilerplate_reads_the_same_population_above_the_floor():
    # the floor used to count every file while the ratio counted only those with a body:
    # reverting the denominator to `files` reads 1.0 here instead of the share below
    # four measurable files against ten in all: the block is in every measurable one
    # (4/4 = 1.0, over the 0.5 share) and in under half of all of them (4/10), so the two
    # denominators disagree instead of both reading 1.0 on the boundary
    same = "the same block in every measurable file"
    with_body = [
        Sample(file=f"{n}.md", content=same, chunk_index=0, body=same, section="T > S")
        for n in "abcd"
    ]
    bodyless = [
        Sample(file=f"{n}.md", content="x", chunk_index=0, section="T > S")
        for n in "efghij"
    ]
    m = measure(with_body + bodyless, CEILING)
    assert m.boilerplate == 1.0, "counted over the files a body can be drawn from"
