import json

import config
import corpus_keys
from sources.converted import Converted
from sources.declaration import SourceFile
from use_cases import raw_quality

BOOK = """# Chapter 2 Processes

Intro words about processes and what the chapter covers.

## 2.1 Creation

A process is created by fork, and the child runs the same program as its parent.

```
pid = fork()
```

## 2.2 Exit

A process ends by exit, and its parent reads the status with wait.
"""


# the index reads an onboarded folder under the files' own paths and cuts it exactly as the source's report did
def test_a_converted_folder_is_read_by_its_paths_and_cut_as_its_report(tmp_path):
    (tmp_path / "files").mkdir()
    rel = "books/ostep/cpu-api.pdf"
    (tmp_path / "files" / f"{corpus_keys.file_stem(rel)}.md").write_text(BOOK)
    record = {"units": {f"{rel}#1-10": {"file": rel}, f"{rel}#11-20": {"file": rel}}}
    (tmp_path / "record.json").write_text(json.dumps(record))
    source = SourceFile(name="ostep", language="en", licence="MIT", folder=str(tmp_path), reader="converted")
    policy = config.settings.corpus.policy()

    docs = Converted(tmp_path, source).documents(policy)

    assert {d.source for d in docs} == {f"ostep/{rel}"}
    samples = raw_quality._samples(BOOK, rel, policy)
    assert [d.content for d in docs] == [s.content for s in samples]
    assert [d.section for d in docs] == [s.section for s in samples]


# a markdown an earlier run left in the folder is not read: the record names the files the index reads
def test_a_markdown_the_record_does_not_name_is_not_read(tmp_path):
    (tmp_path / "files").mkdir()
    rel = "b8472833/Eloquent_JavaScript.pdf"
    (tmp_path / "files" / f"{corpus_keys.file_stem(rel)}.md").write_text(BOOK)
    (tmp_path / "files" / f"{corpus_keys.file_stem('eloquent-javascript.pdf')}.md").write_text(BOOK)
    (tmp_path / "record.json").write_text(json.dumps({"units": {f"{rel}#1-10": {"file": rel}}}))
    source = SourceFile(name="eloquent", language="en", licence="MIT", folder=str(tmp_path), reader="converted")

    docs = Converted(tmp_path, source).documents(config.settings.corpus.policy())

    assert {d.source for d in docs} == {f"eloquent/{rel}"}
