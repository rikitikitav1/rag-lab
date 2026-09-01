import logging_setup

log = logging_setup.get_logger(__name__)


# three readers asked torch this and two rounded it differently
def memory_mb() -> tuple[int, int] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return free // 2**20, total // 2**20
