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
    free, total = (int(v) for v in seen.stdout.splitlines()[0].split(","))
    return free, total
