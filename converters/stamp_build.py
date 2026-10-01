"""Write what the image was built from, so a job can tell an image that lags behind the tree."""

import hashlib
import json
import sys
import time
from pathlib import Path


def main(target, tool, base_digest):
    # a tool's pinned dependencies are part of what it was built from, when it has them
    names = [n for n in ("supervisor.py", "required_models.txt", "Dockerfile", "constraints.txt")
             if n != "constraints.txt" or Path("/opt/converter/constraints.txt").exists()]
    files = {name: hashlib.sha256(Path(f"/opt/converter/{name}").read_bytes()).hexdigest() for name in names}
    stamp = {"tool": tool, "base": base_digest, "files": files,
             "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(target).write_text(json.dumps(stamp, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
