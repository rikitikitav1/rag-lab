from evals import gold_classes


def _row(**over):
    base = {
        "marked_sources": ["notes/databases/postgresql/mvcc-vacuum.md"],
        "gold_heading": "How does **MVCC** work in Postgres?",
        "candidates": [],
    }
    return {**base, **over}


def _chunk(source, section):
    return {"source": source, "section": section, "address": f"{source}#0"}


def test_the_gold_section_is_the_marked_file_and_the_heading_the_question_came_from():
    row = _row(candidates=[_chunk("notes/databases/postgresql/mvcc-vacuum.md",
                                  "mvcc-vacuum > How does MVCC work in Postgres?")])
    assert gold_classes.of_row(row) == [gold_classes.GOLD]


def test_another_section_of_the_gold_file_is_neither_gold_nor_a_stranger():
    # the mark is on the file, and reading a neighbour as foreign inflates what the filter drops
    row = _row(candidates=[_chunk("notes/databases/postgresql/mvcc-vacuum.md",
                                  "mvcc-vacuum > What does vacuum do?")])
    assert gold_classes.of_row(row) == [gold_classes.NEIGHBOUR]


def test_a_chunk_of_an_unmarked_file_is_a_stranger():
    row = _row(candidates=[_chunk("notes/kafka/partitions.md", "partitions > How many?")])
    assert gold_classes.of_row(row) == [gold_classes.STRANGER]


def test_the_heading_is_read_the_way_the_stand_reads_it():
    # numbering and markup live in the corpus, and the question text carries neither
    row = _row(candidates=[_chunk("notes/databases/postgresql/mvcc-vacuum.md",
                                  "mvcc-vacuum > 12. How does _MVCC_ work in Postgres?")])
    assert gold_classes.of_row(row) == [gold_classes.GOLD]


def test_the_denominators_say_how_many_rows_could_keep_anything():
    rows = [
        _row(candidates=[_chunk("notes/databases/postgresql/mvcc-vacuum.md",
                                "mvcc-vacuum > How does MVCC work in Postgres?")]),
        # the file reached the pool and the section did not: retention is not defined on this row
        _row(candidates=[_chunk("notes/databases/postgresql/mvcc-vacuum.md",
                                "mvcc-vacuum > What does vacuum do?")]),
        _row(candidates=[_chunk("notes/kafka/partitions.md", "partitions > How many?")]),
    ]
    got = gold_classes.report(rows)

    assert got["rows_with_the_gold_section"] == 1
    assert got["rows_with_the_gold_file"] == 2
    assert got["by_class"] == {"gold_section": 1, "gold_file_other_section": 1, "stranger": 1}
