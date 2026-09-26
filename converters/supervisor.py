"""Hold a converter's process on the card only while asked: start it on take, kill it on let go."""

import http.client
import http.server
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PORT = int(os.environ.get("SUPERVISOR_PORT", "5001"))
# a tool may need a server of its own beside it; the first child is the one requests are proxied to
CHILDREN = json.loads(os.environ["CONVERTER_CHILDREN"]) if os.environ.get("CONVERTER_CHILDREN") else [{
    "cmd": os.environ["CONVERTER_CHILD"], "port": int(os.environ.get("CONVERTER_CHILD_PORT", "5002")),
    "health": os.environ.get("CONVERTER_CHILD_HEALTH", "/health")}]
CHILD_PORT = CHILDREN[0]["port"]
REQUIRED = Path(os.environ.get("CONVERTER_REQUIRED_MODELS", "/opt/converter/required_models.txt"))
BUILD = Path("/opt/converter/build.json")
STOP_DEADLINE = 20
# a conversion of a long document outlives any sane socket default
PROXY_TIMEOUT = 3600

lock = threading.Lock()
children: list[subprocess.Popen] = []
started_at: float | None = None
stopping = False
in_flight = 0
reason: str | None = None


def _missing_models() -> list[str]:
    if not REQUIRED.exists():
        return []
    wanted = [line.strip() for line in REQUIRED.read_text().splitlines() if line.strip()]
    return [path for path in wanted if not Path(path).is_file()]


def _ready(port: int, health: str) -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", health)
        return conn.getresponse().status == 200
    except OSError:
        return False


# ready only when every child answers: the tool is no use while the server it leans on still loads
def _child_ready() -> bool:
    return all(_ready(c["port"], c["health"]) for c in CHILDREN)


def _alive() -> bool:
    return bool(children) and all(p.poll() is None for p in children)


# a child that dies on its own takes the supervisor down with it, so compose and the state agree
def _watch(proc: subprocess.Popen) -> None:
    code = proc.wait()
    with lock:
        if proc in children and not stopping:
            print(f"supervisor: child exited on its own with {code}", flush=True)
            os._exit(1)


def take() -> dict:
    global children, started_at, reason
    with lock:
        if stopping:
            return {"ready": False, "reason": "a stop is in progress"}
        missing = _missing_models()
        if missing:
            reason = f"models missing: {', '.join(missing)}"
            return {"ready": False, "reason": reason, "permanent": True}
        if not _alive():
            for proc in children:
                if proc.poll() is None:
                    proc.kill()
            children = [subprocess.Popen(shlex.split(c["cmd"])) for c in CHILDREN]
            started_at = time.time()
            reason = None
            for proc in children:
                threading.Thread(target=_watch, args=(proc,), daemon=True).start()
    return {"ready": _child_ready(), "started_at": started_at}


# the card belongs to the queue: a stop wins over a conversion in flight, which fails on its side
def let_go() -> dict:
    global children, started_at, stopping, reason
    with lock:
        procs, flying = [p for p in children if p.poll() is None], in_flight
        if not procs:
            children, started_at = [], None
            return {"held": False}
        stopping = True
    if flying:
        print(f"supervisor: stopped with {flying} requests in flight", flush=True)
    for proc in procs:
        proc.terminate()
    for proc in procs:
        try:
            proc.wait(STOP_DEADLINE)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(STOP_DEADLINE)
            except subprocess.TimeoutExpired:
                with lock:
                    reason = f"child unkillable since {time.strftime('%H:%M:%S')}"
                    stopping = False
                return {"held": True, "reason": reason}
    with lock:
        children, started_at, stopping = [], None, False
    return {"held": False}


def state() -> dict:
    missing = _missing_models()
    with lock:
        alive = _alive()
        if missing and not alive:
            label = "down"
        else:
            label = "unknown" if reason and alive else "holds" if alive else "free"
        return {
            "state": label,
            "missing_models": missing,
            "tool": os.environ.get("CONVERTER_TOOL"),
            "started_at": started_at if alive else None,
            "ready": alive and _child_ready(),
            "in_flight": in_flight,
            "reason": reason,
            "build": json.loads(BUILD.read_text()) if BUILD.exists() else None,
        }


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/supervisor":
            return self._json(200, state())
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        self._proxy()

    def do_POST(self):
        if self.path == "/supervisor/take":
            return self._json(200, take())
        if self.path == "/supervisor/let_go":
            return self._json(200, let_go())
        self._proxy()

    # a tool that takes an upload by PUT or drops a file by DELETE is proxied the same way
    def do_PUT(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    # kept dumb: the body streams through unparsed, a chunked one arrives empty, the reply is buffered whole
    def _proxy(self):
        global in_flight
        with lock:
            alive = _alive() and not stopping
            if alive:
                in_flight += 1
        if not alive:
            return self._json(503, {"detail": "the converter does not hold the card; take it first"})
        if not _child_ready():
            with lock:
                in_flight -= 1
            return self._json(503, {"detail": "the converter is starting"})
        conn = None
        try:
            length = int(self.headers.get("Content-Length") or 0)
            conn = http.client.HTTPConnection("127.0.0.1", CHILD_PORT, timeout=PROXY_TIMEOUT)
            conn.putrequest(self.command, self.path, skip_accept_encoding=True)
            for key, value in self.headers.items():
                if key.lower() != "host":
                    conn.putheader(key, value)
            conn.endheaders()
            left = length
            while left > 0:
                block = self.rfile.read(min(left, 1 << 20))
                if not block:
                    break
                conn.send(block)
                left -= len(block)
            reply = conn.getresponse()
            body = reply.read()
            self.send_response(reply.status)
            for key, value in reply.getheaders():
                if key.lower() not in ("transfer-encoding", "content-length", "connection"):
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError as e:
            self._json(502, {"detail": f"the converter went away mid-request: {e}"})
        finally:
            # a poll a few seconds for hours: an unclosed socket a poll piles up until the collector runs
            if conn is not None:
                conn.close()
            with lock:
                in_flight -= 1

    def log_message(self, fmt, *args):
        return None


def _stop(signum, frame):
    let_go()
    sys.exit(0)


def main():
    signal.signal(signal.SIGTERM, _stop)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"supervisor: listening on {PORT}, children {[c['cmd'] for c in CHILDREN]}, proxied to {CHILD_PORT}",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
