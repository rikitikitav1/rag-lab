import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "datasets" / "inbox" / "books"
MANIFEST = BOOKS / "books.yaml"
API = os.environ.get("RAG_LAB_API", "http://localhost:8000/v1")


def _books(only: set[str], with_held: bool) -> list[dict]:
    books = yaml.safe_load(MANIFEST.read_text())["books"]
    return [b for b in books if (not only or b["name"] in only) and (with_held or not b.get("hold"))]


def _call(method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{API}{path}", data=data, method=method)
    request.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as e:
        body = e.read()
        # a proxy or a crash answers in plain text; the batch reports it and goes on
        try:
            return e.code, json.loads(body or b"null")
        except json.JSONDecodeError:
            return e.code, {"detail": body.decode(errors="replace")[:300]}


# each book into books/<name>/, a loose file moved, a folder renamed
def layout(books: list[dict]) -> None:
    for book in books:
        if "urls" in book:
            continue
        target = BOOKS / book["name"]
        if len(book["files"]) == 1 and (BOOKS / book["files"][0]).is_dir():
            if book["files"][0] != book["name"] and not target.exists():
                (BOOKS / book["files"][0]).rename(target)
        else:
            target.mkdir(exist_ok=True)
            for name in book["files"]:
                if (BOOKS / name).is_file():
                    shutil.move(BOOKS / name, target / name)
        files = sorted(p.name for p in target.iterdir()) if target.is_dir() else []
        print(book["name"], len(files), "files")


# the door answers a page at a time, so the whole list is read page by page
def _ids() -> dict[str, int]:
    ids, offset = {}, 0
    while True:
        status, page = _call("GET", f"/source?limit=1000&offset={offset}")
        if status != 200:
            sys.exit(f"GET /source: {status} {page}")
        ids |= {s["name"]: s["id"] for s in page}
        if len(page) < 1000:
            return ids
        offset += len(page)


# declared through the stand's own door; a name already there is left as it is
def declare(books: list[dict]) -> None:
    known = _ids()
    for book in books:
        if book["name"] in known:
            print(book["name"], "exists", known[book["name"]])
            continue
        body = {"name": book["name"], "language": book["language"], "licence": book["licence"]}
        if "urls" in book:
            body["urls"] = book["urls"]
        else:
            body["folder"] = str((BOOKS / book["name"]).relative_to(ROOT))
        status, out = _call("POST", "/source", body)
        print(book["name"], status, out.get("id") if status == 201 else out)


def onboard(books: list[dict]) -> None:
    known = _ids()
    for book in books:
        if book["name"] not in known:
            print(book["name"], "not declared")
            continue
        status, out = _call("POST", f"/source/{known[book['name']]}/onboard", {})
        print(book["name"], status, out.get("id") if status == 200 else out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lay the store's books out, declare them and queue their onboarding")
    parser.add_argument("command", choices=["layout", "declare", "onboard"])
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--with-held", action="store_true")
    args = parser.parse_args()
    books = _books(set(args.only), args.with_held)
    {"layout": layout, "declare": declare, "onboard": onboard}[args.command](books)


if __name__ == "__main__":
    main()
