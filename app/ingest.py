import re
from pathlib import Path

import config

MAX_CHUNK_SIZE = config.settings.ingestion.chunk_max_size


def chunk_markdown(content, separator="\n## "):
    if not content.strip():
        return []

    parts = content.split(separator)

    intro = parts[0]
    h1 = content.splitlines()[0]
    chunks = [intro] + [h1 + separator + part for part in parts[1:]]

    return split_all_by_size(chunks)


def split_all_by_size(chunks):
    result = []
    for chunk in chunks:
        result.extend(split_by_size(chunk))
    return result


def split_by_size(text, separators=("\n\n", "\n", ". ", " ")):
    if len(text) <= MAX_CHUNK_SIZE:
        return [text]

    if not separators:
        return [
            text[i : i + MAX_CHUNK_SIZE] for i in range(0, len(text), MAX_CHUNK_SIZE)
        ]

    separator, rest = separators[0], separators[1:]
    result, current_chunk = [], ""

    for part in text.split(separator):
        candidate = f"{current_chunk}{separator}{part}" if current_chunk else part
        if len(candidate) <= MAX_CHUNK_SIZE:
            current_chunk = candidate
        else:
            if current_chunk:
                result.append(current_chunk)
            if len(part) > MAX_CHUNK_SIZE:
                result.extend(split_by_size(part, rest))
                current_chunk = ""
            else:
                current_chunk = part

    if current_chunk:
        result.append(current_chunk)

    return result


def path_to_category(rel_path):
    parts = Path(rel_path).with_suffix("").parts
    return ".".join(re.sub(r"[^\w-]", "_", p) for p in parts)

    # for part in parts:
    #     if not re.fullmatch(r"[\w-]+", part):
    #         raise ValueError(f"Invalid path part: {part}")

    # return ".".join(parts)


if __name__ == "__main__":
    for file in Path("/notes").rglob("*.md"):
        print(path_to_category(file.relative_to("/notes")))
        print(chunk_markdown(file.read_text(encoding="utf-8")))
