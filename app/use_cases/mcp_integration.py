import asyncio
import json
from datetime import datetime, timezone

import logging_setup
import mcp_client
from models.mcp_integration import TOOL_NAME_RE, McpIntegration, McpStatus
from orm.sync_db import Session

log = logging_setup.get_logger(__name__)

_MAX_DESCRIPTION_CHARS = 500
_MAX_TOOL_NAME = 128
_MAX_TOOLS = 64
# the whole blob is stored, returned unauthenticated and read into the generation prompt
_MAX_SCHEMA_CHARS = 4000


# both doors are the same walk; only the operation and the answer differ
def _probed(integration_id: int, operation, answer):
    snapshot = _load(integration_id)
    if snapshot is None:
        return None

    outcome, error = _run_probe(operation, snapshot)

    with Session() as session:
        integration = session.get(McpIntegration, integration_id, with_for_update=True)
        if integration is None:
            return None
        mark_probe(integration, error)
        reply = answer(integration, outcome, error)
        session.commit()
        return reply


def check_health(integration_id: int) -> dict | None:
    def answer(integration, elapsed, error):
        return {
            "id": integration.id,
            "name": integration.name,
            "alive": error is None,
            "status": integration.status,
            "elapsed": elapsed,
            "error": error,
        }

    return _probed(integration_id, mcp_client.ping, answer)


def discover(integration_id: int) -> dict | None:
    def answer(integration, tools, error):
        if tools is not None:
            integration.tool_schemas = tool_cache(tools)
        return {
            "id": integration.id,
            "name": integration.name,
            "error": error,
            "tools": [
                {"name": name, "description": schema["description"]}
                for name, schema in integration.tool_schemas.items()
            ],
        }

    return _probed(integration_id, mcp_client.list_tools, answer)


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
    tools = list(tools)
    cache = {}
    for tool in tools[:_MAX_TOOLS]:
        if len(tool.name) > _MAX_TOOL_NAME or not TOOL_NAME_RE.fullmatch(tool.name):
            log.warning("mcp.tool_name_skipped", tool=tool.name[:80])
            continue
        schema = json.dumps(tool.inputSchema or {})
        if len(schema) > _MAX_SCHEMA_CHARS:
            log.warning("mcp.tool_schema_dropped", tool=tool.name[:80], chars=len(schema))
            schema = None
        cache[tool.name] = {
            "description": (tool.description or "")[:_MAX_DESCRIPTION_CHARS],
            "parameters": (tool.inputSchema or {}) if schema is not None else {},
        }
    if len(tools) > _MAX_TOOLS:
        log.warning("mcp.tools_capped", offered=len(tools), kept=_MAX_TOOLS)
    return cache


def _run_probe(op, integration):
    try:
        return asyncio.run(op(integration)), None
    except Exception as e:
        log.warning("mcp.probe_failed", integration=integration.name, error=str(e))
        return None, f"{type(e).__name__}: {e}"[:500]
