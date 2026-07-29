import asyncio
from datetime import datetime, timezone

import logging_setup
import mcp_client
from models.mcp_integration import TOOL_NAME_RE, McpIntegration, McpStatus
from orm.sync_db import Session

log = logging_setup.get_logger(__name__)

_MAX_DESCRIPTION_CHARS = 500


def check_health(integration_id: int) -> dict | None:
    snapshot = _load(integration_id)
    if snapshot is None:
        return None

    elapsed, error = _run_probe(mcp_client.ping, snapshot)

    with Session() as session:
        integration = session.get(
            McpIntegration, integration_id, with_for_update=True
        )
        if integration is None:
            return None
        mark_probe(integration, error)
        session.commit()

        return {
            "id": integration.id,
            "name": integration.name,
            "alive": error is None,
            "status": integration.status,
            "elapsed": elapsed,
            "error": error,
        }


def discover(integration_id: int) -> dict | None:
    snapshot = _load(integration_id)
    if snapshot is None:
        return None

    tools, error = _run_probe(mcp_client.list_tools, snapshot)

    with Session() as session:
        integration = session.get(
            McpIntegration, integration_id, with_for_update=True
        )
        if integration is None:
            return None
        mark_probe(integration, error)
        if tools is not None:
            integration.tool_schemas = tool_cache(tools)
        session.commit()

        return {
            "id": integration.id,
            "name": integration.name,
            "error": error,
            "tools": [
                {"name": name, "description": schema["description"]}
                for name, schema in integration.tool_schemas.items()
            ],
        }


def _load(integration_id: int) -> McpIntegration | None:
    with Session() as session:
        return session.get(McpIntegration, integration_id)


def mark_probe(integration, error: str | None) -> None:
    integration.last_checked_at = datetime.now(timezone.utc)
    integration.last_error = error
    if error is None and integration.status == McpStatus.unreachable:
        integration.status = McpStatus.active
    elif error is not None and integration.status == McpStatus.active:
        integration.status = McpStatus.unreachable


def tool_cache(tools) -> dict:
    cache = {}
    for tool in tools:
        if not TOOL_NAME_RE.match(tool.name):
            log.warning("mcp.tool_name_skipped", tool=tool.name[:80])
            continue
        cache[tool.name] = {
            "description": (tool.description or "")[:_MAX_DESCRIPTION_CHARS],
            "parameters": tool.inputSchema or {},
        }
    return cache


def _run_probe(op, integration):
    try:
        return asyncio.run(op(integration)), None
    except Exception as e:
        log.warning("mcp.probe_failed", integration=integration.name, error=str(e))
        return None, f"{type(e).__name__}: {e}"[:500]
