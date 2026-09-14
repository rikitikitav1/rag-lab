import subprocess

import logging_setup

log = logging_setup.get_logger(__name__)


# asked of the driver: torch opened a CUDA context that held 128 MiB of the card for the API's life
def memory_mb() -> tuple[int, int] | None:
    try:
        seen = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("gpu.memory_unread", error=str(e))
        return None
    if seen.returncode or not seen.stdout.strip():
        return None
    # the card with the most room: a model does not span two cards here, so a sum would overpromise
    try:
        rows = [tuple(int(v) for v in line.split(",")) for line in seen.stdout.splitlines() if line.strip()]
    # a driver without the numbers prints `[N/A]`, and a bare int() raised past every caller
    except ValueError:
        log.warning("gpu.memory_unparsed", said=seen.stdout.strip()[:80])
        return None
    return max(rows)
