from use_cases import mcp_integration

from .base import register


@register("check_mcp_health")
def check_mcp_health(options: dict) -> None:
    mcp_integration.check_health(options["integration_id"])
