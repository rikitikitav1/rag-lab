from pathlib import Path


def test_a_config_comment_and_its_provenance_lines_stay_within_three():
    # config.yaml had run to 235 comment lines of 519 before the rule was applied

    root = Path(__file__).resolve().parent.parent
    run, worst, where = 0, 0, 0
    for i, line in enumerate(root.joinpath("config.yaml").read_text().splitlines(), 1):
        run = run + 1 if line.strip().startswith("#") else 0
        if run > worst:
            worst, where = run, i
    assert worst <= 3, f"comment block of {worst} lines ending at config.yaml:{where}"

# a comment longer than one line is a discussion, and its home is the arc log
LONG_COMMENT_BLOCKS = 0


# every file whose comments are written by hand: `db/schema.sql` is a pg_dump and is not
COMMENTED = (
    "app/**/*.py", "scripts/**/*.py", "tests/**/*.py", "scripts/**/*.sh",
    "db/migrations/*.sql", ".github/workflows/*.yml",
)
COMMENTED_FILES = (
    "Dockerfile", "docker-compose.yml", "config.yaml", ".env.example", "pyproject.toml",
)
# sql says it with two dashes, and a ratchet that looks for `#` there reads nothing at all
MARKERS = {".sql": "--"}
# `-- migrate:up` is dbmate telling the file where to split, not a comment about anything
DIRECTIVES = ("#!", "-- migrate:")


def _comment_lines(source) -> set[int]:
    # tokenised for python, so a `#` inside a fixture string is not read as a comment
    if source.suffix != ".py":
        marker = MARKERS.get(source.suffix, "#")
        return {
            i
            for i, line in enumerate(source.read_text().splitlines(), 1)
            if line.strip().startswith(marker)
            and not line.strip().startswith(DIRECTIVES) and "tuned: file=" not in line
        }
    import io
    import tokenize

    text = source.read_text()
    return {
        tok.start[0]
        for tok in tokenize.generate_tokens(io.StringIO(text).readline)
        if tok.type == tokenize.COMMENT and "tuned: file=" not in tok.string
    }


def _long_comment_blocks() -> list[str]:

    root = Path(__file__).resolve().parent.parent
    sources = [f for pattern in COMMENTED for f in sorted(root.glob(pattern))]
    sources += [root / name for name in COMMENTED_FILES]
    found = []
    for source in sources:
        run: list[int] = []
        for line in sorted(_comment_lines(source)) + [0]:
            if run and line == run[-1] + 1:
                run.append(line)
                continue
            if len(run) > 1:
                found.append(f"{source.relative_to(root)}:{run[0]} ({len(run)} lines)")
            run = [line]
    return found


def test_the_ratchet_reads_every_file_it_says_it_reads():
    # `db/**/*.sql` was declared and read nothing at all: sql says it with `--`, not `#`

    root = Path(__file__).resolve().parent.parent
    blind = []
    for pattern in COMMENTED:
        files = sorted(root.glob(pattern))
        if not files:
            blind.append(f"{pattern}: matches no file")
        elif not any(_comment_lines(f) for f in files):
            blind.append(f"{pattern}: {len(files)} files and not one comment found")
    for name in COMMENTED_FILES:
        source = root / name
        if not source.exists():
            blind.append(f"{name}: missing")
        elif not _comment_lines(source):
            blind.append(f"{name}: not one comment found")
    assert not blind, f"the ratchet is blind to {blind}"


def test_no_comment_block_runs_past_one_line():
    blocks = _long_comment_blocks()
    assert len(blocks) <= LONG_COMMENT_BLOCKS, (
        f"{len(blocks)} comment blocks over one line against a ceiling of "
        f"{LONG_COMMENT_BLOCKS}: {blocks[-5:]}"
    )


def test_no_script_puts_the_app_on_the_path_by_hand():
    # `app` is installed editable by `uv sync` here and by the second sync in the image

    root = Path(__file__).resolve().parent.parent
    # spelled in two halves so this file is not its own first offender
    needle = "sys.path" + ".insert"
    tree = sorted(root.glob("scripts/**/*.py")) + sorted(root.glob("tests/**/*.py"))
    guilty = [str(f.relative_to(root)) for f in tree if needle in f.read_text()]
    # the same mechanism as an environment variable, which the first guard could not see
    carriers = (".github/workflows/ci.yml", "scripts/keyword_switch_grid.sh")
    guilty += [
        f"{name}: {line.strip()}"
        for name in carriers
        for line in (root / name).read_text().splitlines()
        if "PYTHONPATH" in line
    ]

    assert guilty == [], f"the package is installed; these add it again: {guilty}"


def test_one_value_is_capped_the_same_at_every_door_that_names_it():
    # a run name was capped at four lengths across five doors, and one of them at nothing
    import limits
    from api.v1 import eval as eval_door
    from api.v1 import experiment as experiment_door

    def cap(model, field):
        return next(
            (m.max_length for m in model.model_fields[field].metadata
             if getattr(m, "max_length", None) is not None),
            None,
        )

    run_name = [
        (eval_door.EvalRunRequest, "run_name"),
        (eval_door.ExperimentRequest, "run_name"),
        (eval_door.RejudgeRequest, "run_name"),
        (eval_door.RejudgeRequest, "source"),
    ]
    for model, field in run_name:
        assert cap(model, field) == limits.MAX_RUN_NAME, f"{model.__name__}.{field}"

    ids = [
        (eval_door.EvalRunRequest, "question_ids"),
        (eval_door.ExperimentRequest, "question_ids"),
        (experiment_door.ExperimentCreate, "question_ids"),
    ]
    for model, field in ids:
        assert cap(model, field) == limits.MAX_QUESTION_IDS, f"{model.__name__}.{field}"
