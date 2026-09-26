import hashlib
import mimetypes
import time
from pathlib import Path

import requests
from tool_names import Tool

from . import converter

POST_TIMEOUT = 120
POLL_WAIT = 10
# a long poll holds the request for its wait, so the timeout is the wait plus room for the answer
POLL_TIMEOUT = POLL_WAIT + 30

# the terminal task statuses docling-serve reports, and the stand's word for each
DOCLING_STATUSES = {
    "success": "success",
    "partial_success": "partial_success",
    "failure": "failure",
    "skipped": "failure",
}
# the pages a tool lost are what the metrics must see, so a partial result is kept with its errors
KEPT_STATUSES = {"success", "partial_success"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# submitted, polled, fetched: an hour of conversion is never one open request
def _docling_piece(spec, path: Path, fields: list[tuple[str, str]], chunk, ceiling: float) -> dict:
    base = converter.base_url(spec)
    if chunk is not None:
        fields = fields + [("page_range", str(chunk[0])), ("page_range", str(chunk[1]))]
    started = time.monotonic()
    with path.open("rb") as f:
        files = {"files": (path.name, f, mimetypes.guess_type(path.name)[0] or "application/octet-stream")}
        seen = requests.post(base + "/v1/convert/file/async", data=fields, files=files, timeout=POST_TIMEOUT)
    seen.raise_for_status()
    task = seen.json()["task_id"]
    deadline = time.monotonic() + ceiling
    while True:
        polled = requests.get(f"{base}/v1/status/poll/{task}", params={"wait": POLL_WAIT}, timeout=POLL_TIMEOUT)
        # a 503 is the supervisor saying its card went, which a KeyError on the body would hide
        polled.raise_for_status()
        status = polled.json()
        if status["task_status"] in DOCLING_STATUSES:
            break
        if time.monotonic() > deadline:
            return {
                "status": "timeout",
                "tool_status": status["task_status"],
                "seconds": round(time.monotonic() - started, 2),
                "errors": [f"no result in {ceiling} s"],
                "markdown": None,
            }
    body = (
        requests.get(f"{base}/v1/result/{task}", timeout=POST_TIMEOUT).json()
        if status["task_status"] != "failure"
        else {}
    )
    document = body.get("document") or {}
    said = body.get("status") or status["task_status"]
    stand = DOCLING_STATUSES.get(said, "failure")
    return {
        "status": stand,
        "tool_status": said,
        "seconds": round(time.monotonic() - started, 2),
        "errors": body.get("errors") or [x for x in (status.get("error_message"),) if x],
        "markdown": document.get("md_content") if stand in KEPT_STATUSES else None,
        # Docling's own structure, when `to_formats` asked for json beside the markdown
        "structure": document.get("json_content") if stand in KEPT_STATUSES else None,
    }


# MinerU's job and file words, and the stand's word for each
MINERU_STATUSES = {"completed": "success", "partial": "partial_success", "failed": "failure", "canceled": "failure"}
# an uploaded file lives in the tool's process: a restart forgets it, so the process's start is in the key
_MINERU_FILES: dict = {}


def _mineru_file(base: str, path: Path, residency) -> str:
    key = (base, residency, str(path), sha256(path))
    if key in _MINERU_FILES:
        return _MINERU_FILES[key]
    blob = path.read_bytes()
    up = requests.post(
        base + "/v1/uploads",
        json={
            "filename": path.name,
            "bytes": len(blob),
            "purpose": "parse",
            "sha256sum": key[3],
            "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        },
        timeout=POST_TIMEOUT,
    )
    up.raise_for_status()
    upload = up.json()
    if not (upload.get("file") or {}).get("id"):
        requests.put(
            f"{base}/v1/uploads/{upload['id']}/content",
            data=blob,
            headers={"Content-Type": "application/octet-stream"},
            timeout=POST_TIMEOUT,
        ).raise_for_status()
        done = requests.post(
            f"{base}/v1/uploads/{upload['id']}/complete", json={"sha256sum": key[3]}, timeout=POST_TIMEOUT
        )
        done.raise_for_status()
        upload = done.json()
    _MINERU_FILES[key] = upload["file"]["id"]
    return _MINERU_FILES[key]


# uploaded once, a job a piece by page range, polled, the markdown fetched by its own file id
def _mineru_piece(spec, path: Path, fields: list[tuple[str, str]], chunk, ceiling: float) -> dict:
    base = converter.base_url(spec)
    residency = converter.reading(spec)[1].get("started_at")
    started = time.monotonic()
    entry = {"source": {"type": "file_id", "file_id": _mineru_file(base, path, residency)}}
    if chunk is not None:
        entry["page_range"] = f"{chunk[0]}-{chunk[1]}"
    posted = requests.post(base + "/v1/parse/jobs", json={**dict(fields), "files": [entry]}, timeout=POST_TIMEOUT)
    if posted.status_code == 404:
        _MINERU_FILES.clear()
        entry["source"]["file_id"] = _mineru_file(base, path, residency)
        posted = requests.post(base + "/v1/parse/jobs", json={**dict(fields), "files": [entry]}, timeout=POST_TIMEOUT)
    posted.raise_for_status()
    job = posted.json()["job_id"]
    deadline = time.monotonic() + ceiling
    while True:
        polled = requests.get(f"{base}/v1/parse/jobs/{job}", timeout=POST_TIMEOUT)
        polled.raise_for_status()
        state = polled.json()
        if state["status"] in MINERU_STATUSES:
            break
        if time.monotonic() > deadline:
            return {
                "status": "timeout",
                "tool_status": state["status"],
                "seconds": round(time.monotonic() - started, 2),
                "errors": [f"no result in {ceiling} s"],
                "markdown": None,
            }
        time.sleep(POLL_WAIT / 5)
    result = state["files"][0]
    stand = MINERU_STATUSES.get(result.get("status"), "failure")
    markdown_ref = (result.get("output_files") or {}).get("markdown") or {}
    markdown = None
    if stand in KEPT_STATUSES and markdown_ref.get("file_id"):
        fetched = requests.get(f"{base}/v1/files/{markdown_ref['file_id']}/content", timeout=POST_TIMEOUT)
        fetched.raise_for_status()
        markdown = fetched.text
    return {
        "status": stand,
        "tool_status": result.get("status"),
        "seconds": round(time.monotonic() - started, 2),
        "errors": [result["error"]] if result.get("error") else [],
        "markdown": markdown,
    }


# one adapter a tool: each speaks its tool's API and answers in the stand's four statuses
PIECE = {Tool.docling: _docling_piece, Tool.mineru: _mineru_piece}
# where each tool says its own version
VERSION_ROUTE = {Tool.docling: "/version", Tool.mineru: "/v1/health"}


def tool_version(spec, tool: str):
    route = VERSION_ROUTE.get(tool)
    return requests.get(converter.base_url(spec) + route, timeout=10).json() if route else None
