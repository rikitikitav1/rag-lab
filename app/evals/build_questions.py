import json
import re
from pathlib import Path

import config

QUESTION_RE = re.compile(r"^## \d+\.\s+(.+)$", re.MULTILINE)


def _clean(text):
    return re.sub(r"[_*`]", "", text).strip()


def _readmes():
    repos_dir = Path(config.settings.repos_dir)
    for repo in config.settings.sources.interview.repos:
        readme = repos_dir / repo / "README.md"
        if readme.exists():
            yield f"{repo}/README.md", readme.read_text(encoding="utf-8", errors="ignore")


def _qa_pairs(text):
    headers = list(QUESTION_RE.finditer(text))
    for i, match in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        yield _clean(match.group(1)), text[match.end() : end].strip()


def build(out_path="questions_interview.md"):
    lines = [
        f"{question} | {source}"
        for source, text in _readmes()
        for question, _ in _qa_pairs(text)
    ]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def build_answers(out_path="answers_interview.jsonl"):
    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for source, text in _readmes():
            for question, answer in _qa_pairs(text):
                record = {
                    "question": question,
                    "reference_answer": answer,
                    "source": source,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def _interview_records(path="answers_interview.jsonl"):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        source = d.get("source")
        yield {
            "original_text": d["question"],
            "set_name": "interview",
            "language": "eng",
            "marked_sources": [source] if source else [],
            "reference_answer": d.get("reference_answer"),
            "kind": "in_corpus",
        }


def _curated_records(path="questions.md"):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        question, _, raw = line.partition("|")
        marked = [] if raw.strip() == "NONE" else [x.strip() for x in raw.split(",") if x.strip()]
        yield {
            "original_text": question.strip(),
            "set_name": "curated",
            "language": "ru",
            "marked_sources": marked,
            "reference_answer": None,
            "kind": "in_corpus" if marked else "out_of_corpus",
        }


DATASET_COLUMNS = ["set_name", "language", "kind", "marked_sources", "original_text"]


def build_dataset(out_path="questions.tsv"):
    records = list(_curated_records()) + list(_interview_records())
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("\t".join(DATASET_COLUMNS) + "\n")
        for r in records:
            out.write(
                "\t".join(
                    [
                        r["set_name"],
                        r["language"] or "",
                        r["kind"] or "",
                        ",".join(r["marked_sources"]),
                        r["original_text"],
                    ]
                )
                + "\n"
            )
    return len(records)


if __name__ == "__main__":
    print("questions:", build())
    print("answers:", build_answers())
    print("dataset:", build_dataset())
